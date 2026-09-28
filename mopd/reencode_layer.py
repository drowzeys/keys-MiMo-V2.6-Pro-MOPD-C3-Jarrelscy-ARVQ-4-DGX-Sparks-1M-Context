#!/usr/bin/env python3
"""Re-encode one MoE layer of MiMo-V2.6-Pro (e.g. MOPD) into the served
ARVQ/NVFP4 hybrid format (v4 rvq256_256x8_expert_fp16block), REUSING the
published per-expert 8+8 FP4 books and projection globals of an existing tree.

Writes <out>/arvq-layer-NNN-{gateup,down}.safetensors and roster-layer-NNN.safetensors
with the exact keys/dtypes/shapes (asserted) of the originals.

Cold modes:
  delta (default): T = Q_pub(RL) + (W_mopd - W_rl); only 8-groups whose source
      changed are re-assigned (fixed books + published PV scales). Preserves the
      activation-Hessian/PV choices of the published fit. Needs --src-rl-dir.
  full: from-scratch fixed-book encode of W_mopd (beam assign <-> FP16 LS scale
      refit, warm-started from published scales). Weight-L2 optimal, but
      discards the output-aware (PV) code/scale choices.
Hot (NVFP4) experts: requantized from the new source with Jarrelscy's
hybrid.quantize (bit-exact reproduction of the published hot tensors on RL).

Experts whose new-source shard is not present on disk are copied through
unchanged (reported), unless --require-all.

Example:
  python3 reencode_layer.py --layer 64 \
    --src-rl-dir "/media/keyspark/My Book1/Local Models Archive/MiMo-V2.6-Pro-RL" \
    --src-mopd-dir "/media/keyspark/My Book1/Local Models Archive/MiMo-V2.6-Pro-MOPD" \
    --arvq-tree ./arvq-src --out ./out
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from arvq_reencode_lib import (DIMS, TAG, P, Source, books_from_u32, drop_source_expert,
                               encode_delta, encode_expert, nvfp4_quantize, pack_expert,
                               prefix, reconstruct, rel, unpack_expert)


def parse_experts(spec):
    if spec is None:
        return None
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def load_all(path):
    with safe_open(path, "pt", device="cpu") as h:
        return {k: h.get_tensor(k) for k in h.keys()}, h.metadata() or {}


def signature(path):
    with safe_open(path, "pt", device="cpu") as h:
        return {k: (tuple(h.get_slice(k).get_shape()), h.get_slice(k).get_dtype()) for k in h.keys()}


def _complete(path):
    """True only if the safetensors file is fully written (size == header + data)."""
    import struct
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
        return os.path.getsize(path) == 8 + n + end
    except Exception:  # noqa: BLE001
        return False


def available(src: Source, layer, e, allow_shards=None):
    rel_name = src.index[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"]
    if allow_shards is not None and rel_name not in allow_shards:
        return False
    f = src.directory / rel_name
    return f.exists() and _complete(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--src-rl-dir", required=False)
    ap.add_argument("--src-mopd-dir", required=True, help="new source (MXFP4, official layout)")
    ap.add_argument("--arvq-tree", required=True, help="local copy of the served ARVQ tree (config.json + layer files)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("delta", "full"), default="delta")
    ap.add_argument("--experts", default=None, help="restrict, e.g. 0-47,100 (others copied)")
    ap.add_argument("--require-all", action="store_true")
    ap.add_argument("--hot-only", action="store_true",
                    help="requantize only the NVFP4 hot experts; write only roster-layer-NNN")
    ap.add_argument("--only-shards", default=None,
                    help="comma list of new-source shard filenames allowed to be read (e.g. while a download is running)")
    ap.add_argument("--refit-scales", action="store_true", help="delta mode: LS-refit touched blocks")
    ap.add_argument("--beam", type=int, default=8)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    L, dev = args.layer, args.device
    tree, out = Path(args.arvq_tree), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == "delta" and not args.src_rl_dir:
        ap.error("--mode delta needs --src-rl-dir")
    books = json.loads((tree / "config.json").read_text())["quantization_config"]["aqlm_layer_books"][str(L)]
    cold = books.get("cold_expert_ids") or list(range(384))
    roster_path = tree / f"roster-layer-{L:03d}.safetensors"
    roster_sig = signature(roster_path)
    with safe_open(roster_path, "pt", device="cpu") as h:
        kind = h.get_tensor(f"model.layers.{L}.mlp.experts.hyb_kind")
    hot = (kind == 0).nonzero().flatten().tolist()
    assert sorted(set(cold) | set(hot)) == list(range(384)) and not set(cold) & set(hot)
    assert cold == sorted(cold), "non-ascending cold order unsupported here"
    new = Source(args.src_mopd_dir)
    old = Source(args.src_rl_dir) if args.src_rl_dir else None
    want = set(hot) if args.hot_only else parse_experts(args.experts)
    todo = [e for e in range(384) if (want is None or e in want)]
    allow = set(args.only_shards.split(",")) if args.only_shards else None
    avail = {e for e in todo if available(new, L, e, allow)}
    missing = sorted(set(todo) - avail)
    if missing and args.require_all:
        raise SystemExit(f"new-source shards missing for {len(missing)} experts: {missing[:10]}...")
    report = {"layer": L, "mode": args.mode, "cold": {}, "hot": {}, "copied_unchanged": missing,
              "not_requested": sorted(set(range(384)) - set(todo))}
    t_start = time.time()
    # All three layer files resident on CPU (<= ~6 GB); experts streamed one at a time.
    files, sigs, metas, T = {}, {}, {}, {}
    for proj in () if args.hot_only else ("w13", "w2"):
        files[proj] = tree / f"arvq-layer-{L:03d}-{TAG[proj]}.safetensors"
        sigs[proj] = signature(files[proj])
        T[proj], metas[proj] = load_all(files[proj])
    R, rmeta = load_all(roster_path)
    rp = f"model.layers.{L}.mlp.experts.nvfp4_"
    acc = {proj: {"pub_old": [0.0, 0.0], "new_vs_src": [0.0, 0.0], "pub_vs_newsrc": [0.0, 0.0],
                  "idx_changed": 0, "idx_total": 0, "scales_changed": 0, "scales_total": 0,
                  "delta_dot": 0.0, "delta_energy": 0.0, "delta_resid": 0.0, "experts": 0}
           for proj in ("w13", "w2")}
    hot_changed, nhot, t_io, t_cmp = {}, 0, 0.0, 0.0
    for e in sorted(avail):
        t0 = time.time()
        gn, un, dn = (new.expert(L, e, p, dev) for p in ("gate_proj", "up_proj", "down_proj"))
        drop_source_expert(new, L, e)
        if old is not None:
            go, uo, do = (old.expert(L, e, p, dev) for p in ("gate_proj", "up_proj", "down_proj"))
            drop_source_expert(old, L, e)
        torch.cuda.synchronize()
        t1 = time.time()
        t_io += t1 - t0
        if e in hot:
            slot = hot.index(e)
            vals = [nvfp4_quantize(x) for x in (gn, un, dn)]
            newt = {"w13_packed": torch.cat((vals[0][0], vals[1][0])),
                    "w13_bscale": torch.cat((vals[0][1], vals[1][1])),
                    "w13_scale2": torch.stack((vals[0][2], vals[1][2])),
                    "w2_packed": vals[2][0], "w2_bscale": vals[2][1], "w2_scale2": vals[2][2][None]}
            for k, v in newt.items():
                v = v.cpu().to(R[rp + k].dtype)
                hot_changed[k] = hot_changed.get(k, 0) + int((R[rp + k][slot] != v).sum())
                R[rp + k][slot] = v
            nhot += 1
        else:
            slot = cold.index(e)
            for proj in ("w13", "w2"):
                N, K = DIMS[proj]
                t, pre, a_ = T[proj], prefix(L, proj), acc[proj]
                glob = float(t[pre + "global"][0])
                Wn = torch.cat((gn, un)) if proj == "w13" else dn
                c0, c1 = books_from_u32(t[pre + "codebooks"][slot].to(dev))
                a0, b0, s0 = unpack_expert(t[pre + "packed"][slot].to(dev),
                                           t[pre + "scales"][slot].to(dev), N, K)
                Qp = reconstruct(a0, b0, s0, c0, c1, glob)
                Wo = None
                if old is not None:
                    Wo = torch.cat((go, uo)) if proj == "w13" else do
                    a_["pub_old"][0] += (Wo - Qp).square().sum().item()
                    a_["pub_old"][1] += Wo.square().sum().item()
                if args.mode == "delta":
                    a, b, s, st = encode_delta(Wn, Wo, a0, b0, s0, c0, c1, glob, beam=args.beam,
                                               refit_scales=args.refit_scales)
                    for k in ("delta_dot", "delta_energy", "delta_resid"):
                        a_[k] += st[k]
                else:
                    a, b, s = encode_expert(Wn, c0, c1, glob, s0, passes=args.passes, beam=args.beam)
                packed, scales = pack_expert(a, b, s, N, K)
                # round-trip through Jarrelscy unpack/decode (bit-exact)
                ua, ub = P.unpack(packed, N, K)
                assert torch.equal(ua, a) and torch.equal(ub, b)
                one = {pre + "packed": packed[None], pre + "scales": scales[None],
                       pre + "codebooks": t[pre + "codebooks"][slot][None].to(dev),
                       pre + "global": t[pre + "global"].to(dev)}
                Q = P.decode(one, L, proj, 0, N, K)
                assert torch.equal(Q, reconstruct(a, b, s, c0, c1, glob))
                den = Wn.square().sum().item()
                a_["new_vs_src"][0] += (Wn - Q).square().sum().item()
                a_["new_vs_src"][1] += den
                a_["pub_vs_newsrc"][0] += (Wn - Qp).square().sum().item()
                a_["pub_vs_newsrc"][1] += den
                a_["idx_changed"] += int(((a != a0) | (b != b0)).sum())
                a_["idx_total"] += a.numel()
                a_["scales_changed"] += int((s != s0).sum())
                a_["scales_total"] += s.numel()
                a_["experts"] += 1
                t[pre + "packed"][slot] = packed.cpu()
                t[pre + "scales"][slot] = scales.cpu().to(t[pre + "scales"].dtype)
        torch.cuda.synchronize()
        t_cmp += time.time() - t1
    for proj in () if args.hot_only else ("w13", "w2"):
        name = files[proj].name
        meta = dict(metas[proj])
        meta["reencode"] = json.dumps({"mode": args.mode, "source": str(args.src_mopd_dir),
                                       "fixed_books_from": str(tree), "experts": acc[proj]["experts"]})
        tmp = out / (name + ".tmp")
        save_file(T[proj], str(tmp), metadata=meta)
        os.replace(tmp, out / name)
        assert signature(out / name) == sigs[proj], f"{name}: key/dtype/shape mismatch vs original"
        a_ = acc[proj]
        r = {"experts": a_["experts"],
             "rel_l2_new_vs_newsrc": rel(*a_["new_vs_src"]),
             "rel_l2_pub_vs_newsrc": rel(*a_["pub_vs_newsrc"]),
             "rel_l2_pub_vs_oldsrc": rel(*a_["pub_old"]) if a_["pub_old"][1] else None,
             "idx_changed_frac": a_["idx_changed"] / max(a_["idx_total"], 1),
             "scales_changed_frac": a_["scales_changed"] / max(a_["scales_total"], 1)}
        if args.mode == "delta" and a_["delta_energy"]:
            r["delta_rel_l2_vs_src"] = rel(a_["delta_energy"], a_["new_vs_src"][1])
            r["delta_transfer_proj"] = a_["delta_dot"] / a_["delta_energy"]
            r["delta_transfer_resid_rel"] = rel(a_["delta_resid"], a_["delta_energy"])
        report["cold"][proj] = r
    T.clear()
    tmp = out / (roster_path.name + ".tmp")
    save_file(R, str(tmp), metadata=rmeta or None)
    os.replace(tmp, out / roster_path.name)
    assert signature(out / roster_path.name) == roster_sig, "roster key/dtype/shape mismatch"
    del R
    report["hot"] = {"experts": nhot, "elements_changed": hot_changed}
    report["seconds_source_read"] = t_io
    report["seconds_compute"] = t_cmp
    report["seconds"] = time.time() - t_start
    report["experts_reencoded"] = len(avail)
    (out / f"reencode-layer-{L:03d}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("not_requested",)}, indent=1)[:4000])


if __name__ == "__main__":
    main()
