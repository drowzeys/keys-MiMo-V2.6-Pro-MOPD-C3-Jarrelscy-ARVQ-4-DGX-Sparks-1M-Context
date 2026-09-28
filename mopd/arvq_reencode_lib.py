"""Fixed-codebook ARVQ re-encoder (v4: rvq256_256x8_expert_fp16block).

Reuses Jarrelscy's pack/unpack/decode (recipe/arvq88/pack.py), the NVFP4 hot
quantizer (hybrid.quantize/decode) and the MXFP4 source reader (source.Source).
Only the per-expert code assignment + block-scale LS refit is new: the
per-expert 8+8 FP4 books and the projection global scale are kept FIXED.

W[n, 8g:8g+8] = glob * s[n, g//16] * (c0[a[n,g]] + c1[b[n,g]])
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

TOOLS = Path("/home/keyspark/mimo26-arvq-tp4/runtime/tools/mimo_arvq")
for p in (TOOLS, TOOLS / "recipe"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from arvq88 import pack as P  # noqa: E402  (Jarrelscy pack/unpack/decode)
from hybrid import quantize as nvfp4_quantize, decode as nvfp4_decode  # noqa: E402
from source import Source, decode_mxfp4  # noqa: E402

LEVELS = torch.tensor(P.LEVELS, dtype=torch.float32)
DIMS = {"w13": (4096, 6144), "w2": (6144, 2048)}
TAG = {"w13": "gateup", "w2": "down"}


def prefix(layer, proj):
    return f"model.layers.{layer}.mlp.experts.arvq_{proj}_"


# ------------------------------------------------------------------ decode side
def books_from_u32(cb_row: torch.Tensor):
    """uint32 [512] -> c0 [256,8], c1 [256,8] fp32 (same nibble rule as P.decode)."""
    cb = cb_row.long()
    vals = LEVELS.to(cb.device)[(cb[:, None] >> (torch.arange(8, device=cb.device) * 4)) & 15]
    return vals[:256].contiguous(), vals[256:].contiguous()


def unpack_expert(packed_e, scales_e, N, K):
    """Stored per-expert tensors -> a,b uint8 [N,K/8], s fp32 [N,K/128]."""
    a, b = P.unpack(packed_e, N, K)
    s = scales_e.permute(0, 2, 1).contiguous().reshape(N, K // 128)
    s = s.float() if s.dtype == torch.float16 else s.view(torch.float8_e4m3fn).float()
    return a, b, s


def reconstruct(a, b, s, c0, c1, glob):
    N, G = a.shape
    cb = c0[a.long()] + c1[b.long()]
    # same op order as Jarrelscy P.decode -> bit-exact
    return (cb * s.repeat_interleave(16, 1)[..., None] * glob).reshape(N, G * 8)


def source_expert(src: Source, layer, expert, device):
    """Dense fp32 (w13=[gate;up] [4096,6144], w2=down [6144,2048])."""
    g = src.expert(layer, expert, "gate_proj", device)
    u = src.expert(layer, expert, "up_proj", device)
    d = src.expert(layer, expert, "down_proj", device)
    return torch.cat((g, u)), d, (g, u, d)


# ------------------------------------------------------------------ encoder
def assign(W, s, c0, c1, glob, *, beam=8, refine=2, chunk=1 << 17):
    """Best (a,b) per 8-group given FIXED scales. Beam over first-stage atoms,
    exact best residual atom per beam entry, then alternating refinement.
    Error in normalized space == real space per group (scale is per group)."""
    N, K = W.shape
    sden = (glob * s).clamp_min(1e-30).repeat_interleave(16, 1)          # [N,K/8]
    tn_all = (W.reshape(N, K // 8, 8) / sden[..., None]).reshape(-1, 8)
    c0n, c1n = c0.square().sum(1), c1.square().sum(1)
    A = torch.empty(tn_all.shape[0], dtype=torch.long, device=W.device)
    B = torch.empty_like(A)
    for i in range(0, tn_all.shape[0], chunk):
        tn = tn_all[i:i + chunk]
        m = tn.shape[0]
        d0 = c0n[None] - 2 * tn @ c0.t()                                   # [m,256]
        cand = d0.topk(beam, dim=1, largest=False).indices                 # [m,B]
        r = tn[:, None, :] - c0[cand]                                       # [m,B,8]
        d1 = c1n[None, None] - 2 * r @ c1.t()                               # [m,B,256]
        v1, j1 = d1.min(-1)
        tot = r.square().sum(-1) + v1                                       # [m,B]
        k = tot.argmin(1)
        ar = torch.arange(m, device=W.device)
        a = cand[ar, k]
        b = j1[ar, k]
        for _ in range(refine):
            a = (c0n[None] - 2 * (tn - c1[b]) @ c0.t()).argmin(1)
            b = (c1n[None] - 2 * (tn - c0[a]) @ c1.t()).argmin(1)
        A[i:i + m] = a
        B[i:i + m] = b
    return A.view(N, K // 8).to(torch.uint8), B.view(N, K // 8).to(torch.uint8)


def refit_scales_fp16(W, a, b, c0, c1, glob, hdiag=None):
    """LS per-(row,128block) scale given codes (Jarrelscy refit_scales with an
    identity/diagonal Hessian), rounded to FP16 (v4 stores FP16 block scales)."""
    N, K = W.shape
    u = (c0[a.long()] + c1[b.long()]).reshape(N, K // 8, 8) * glob
    Wg = W.reshape(N, K // 8, 8)
    hd = (torch.ones(K, device=W.device) if hdiag is None else hdiag).reshape(K // 8, 8)[None]
    num = (hd * Wg * u).reshape(N, K // 128, 128).sum(-1)
    den = (hd * u * u).reshape(N, K // 128, 128).sum(-1).clamp_min(1e-30)
    s = (num / den).clamp(0, 65504.0)
    return s.half().float()


def encode_expert(W, c0, c1, glob, s_init, *, passes=3, beam=8, refine=2, hdiag=None):
    """Alternate assignment <-> scale LS starting from s_init; final (a,b,s)
    are mutually consistent (assignment done with the final scales)."""
    s = s_init.clone()
    best = None
    for p in range(passes):
        a, b = assign(W, s, c0, c1, glob, beam=beam, refine=refine)
        err = (W - reconstruct(a, b, s, c0, c1, glob)).square().sum().item()
        if best is None or err < best[0]:
            best = (err, a, b, s)
        s = refit_scales_fp16(W, a, b, c0, c1, glob, hdiag)
        err = (W - reconstruct(a, b, s, c0, c1, glob)).square().sum().item()
        if err < best[0]:
            best = (err, a, b, s)
    return best[1], best[2], best[3]


def init_scales_rms(W, glob):
    """Jarrelscy Phase-A init: block RMS / glob (FP16 here)."""
    N, K = W.shape
    return (W.reshape(N, K // 128, 128).square().mean(-1).sqrt() / glob).half().float()


def pack_expert(a, b, s, N, K):
    """-> packed uint32 [N/16,K/64,64] (Jarrelscy P.pack) and fp16 scales
    [N/16,K/128,16] (same reshape/permute as P.export_layer)."""
    packed = P.pack(a, b, N, K)
    scales = s.half().reshape(N // 16, 16, K // 128).permute(0, 2, 1).contiguous()
    return packed, scales


def rel(num, den):
    return (num / max(den, 1e-30)) ** 0.5


def drop_cache(path):
    """Drop clean page cache of a file (GB10 cudaMalloc fails when MemFree is
    low even though page cache is reclaimable)."""
    import os
    try:
        fd = os.open(str(path), os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass


def drop_source_expert(src: Source, layer, expert):
    drop_cache(src.directory / src.index[f"model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"])


# ------------------------------------------------------------------ delta mode
def _group_err(T, a, b, s, c0, c1, glob):
    """Per-group squared error [N,K/8] of codes (a,b) at scales s vs target T."""
    N, K = T.shape
    return (T - reconstruct(a, b, s, c0, c1, glob)).reshape(N, K // 8, 8).square().sum(-1)


def encode_delta(W_new, W_old, a0, b0, s0, c0, c1, glob, *, beam=8, refine=2,
                 refit_scales=False):
    """Transfer a sparse source update onto the published encoding.

    Target T = Q_pub + (W_new - W_old): keeps the published (activation-Hessian /
    PV-trained) code+scale choices everywhere the source did not change and adds
    the update on top. Only 8-groups whose source changed are re-assigned (fixed
    books, fixed published scales); a new code is accepted only if it is strictly
    closer to T. With refit_scales, touched 128-blocks get an LS scale refit on T
    (accepted per block only if the block error drops).
    Returns a, b, s, stats."""
    N, K = W_new.shape
    Qp = reconstruct(a0, b0, s0, c0, c1, glob)
    D = W_new - W_old
    T = Qp + D
    touched = (D.reshape(N, K // 8, 8) != 0).any(-1)                       # [N,K/8]
    a, b, s = a0.clone(), b0.clone(), s0.clone()
    if touched.any():
        an, bn = _assign_masked(T, s, c0, c1, glob, touched, beam, refine)
        e_old = _group_err(T, a0, b0, s, c0, c1, glob)
        e_new = _group_err(T, an, bn, s, c0, c1, glob)
        take = touched & (e_new < e_old)
        a = torch.where(take, an, a0)
        b = torch.where(take, bn, b0)
        if refit_scales:
            blk = touched.reshape(N, K // 128, 16).any(-1)                  # [N,K/128]
            s_ls = refit_scales_fp16(T, a, b, c0, c1, glob)
            s_try = torch.where(blk, s_ls, s)
            bm = blk.repeat_interleave(16, 1)
            an2, bn2 = _assign_masked(T, s_try, c0, c1, glob, bm, beam, refine)
            a2 = torch.where(bm, an2, a)
            b2 = torch.where(bm, bn2, b)
            eb_old = _group_err(T, a, b, s, c0, c1, glob).reshape(N, K // 128, 16).sum(-1)
            eb_new = _group_err(T, a2, b2, s_try, c0, c1, glob).reshape(N, K // 128, 16).sum(-1)
            acc = blk & (eb_new < eb_old)
            accg = acc.repeat_interleave(16, 1)
            a = torch.where(accg, a2, a)
            b = torch.where(accg, b2, b)
            s = torch.where(acc, s_try, s)
    Q = reconstruct(a, b, s, c0, c1, glob)
    dn = D.square().sum().item()
    moved = Q - Qp
    stats = {
        "groups_touched": int(touched.sum()),
        "groups": touched.numel(),
        "idx_changed": int(((a != a0) | (b != b0)).sum()),
        "scales_changed": int((s != s0).sum()),
        "delta_energy": dn,
        # fraction of the update transferred (projection of the code movement on D)
        "delta_dot": float((moved * D).sum()),
        "delta_resid": float((moved - D).square().sum()),
    }
    return a, b, s, stats


def _assign_masked(T, s, c0, c1, glob, mask, beam, refine):
    """assign() only on masked groups; others returned as 0 (caller masks)."""
    N, K = T.shape
    rows, gs = mask.nonzero(as_tuple=True)
    a = torch.zeros(N, K // 8, dtype=torch.uint8, device=T.device)
    b = torch.zeros_like(a)
    if rows.numel() == 0:
        return a, b
    tg = T.reshape(N, K // 8, 8)[rows, gs]                                  # [M,8]
    sg = s[rows, gs // 16]                                                  # [M]
    # reuse assign() on a [M,8] "matrix" with one 8-group per row
    M = tg.shape[0]
    tn = tg / (glob * sg).clamp_min(1e-30)[:, None]
    c0n, c1n = c0.square().sum(1), c1.square().sum(1)
    A = torch.empty(M, dtype=torch.long, device=T.device)
    B = torch.empty_like(A)
    ch = 1 << 17
    for i in range(0, M, ch):
        x = tn[i:i + ch]
        m = x.shape[0]
        cand = (c0n[None] - 2 * x @ c0.t()).topk(beam, 1, largest=False).indices
        r = x[:, None, :] - c0[cand]
        v1, j1 = (c1n[None, None] - 2 * r @ c1.t()).min(-1)
        k = (r.square().sum(-1) + v1).argmin(1)
        ar = torch.arange(m, device=T.device)
        aa, bb = cand[ar, k], j1[ar, k]
        for _ in range(refine):
            aa = (c0n[None] - 2 * (x - c1[bb]) @ c0.t()).argmin(1)
            bb = (c1n[None] - 2 * (x - c0[aa]) @ c1.t()).argmin(1)
        A[i:i + m], B[i:i + m] = aa, bb
    a[rows, gs] = A.to(torch.uint8)
    b[rows, gs] = B.to(torch.uint8)
    return a, b
