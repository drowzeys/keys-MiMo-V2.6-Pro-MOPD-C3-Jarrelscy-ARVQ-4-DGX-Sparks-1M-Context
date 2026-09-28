#!/usr/bin/env python3
"""Stage 3: widen the NVFP4 hot set to 5,560 experts (21%) and fill it from MOPD.

Selection = Jarrelscy's prepare_hybrid.select_hot (global routing-weighted benefit
score from the tree's allocation_scores, per-layer cap 192) with count 5560 instead
of 1325; it is a strict superset of the published hot set.

Per changed layer (hot count differs from the published one):
  roster-layer-NNN      every hot expert requantized from MOPD with hybrid.quantize
                        (same as stage 2), slots in ascending expert order, hyb_kind 0/2
  arvq-layer-NNN-*      the published cold rows (RL, PV-fitted) for the experts that stay
                        cold, rows selected in ascending expert order; globals unchanged
config.json            aqlm_layer_books[L] n_nvfp4 / n_cold / cold_expert_ids updated
Unchanged layers keep the stage 2 roster (MOPD hot) and the published cold files.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/keyspark/mimo26-arvq-tp4/mopd/reencode")
from arvq_reencode_lib import Source, drop_source_expert, nvfp4_quantize  # noqa: E402

M = Path("/home/keyspark/mimo26-arvq-tp4/mopd")
OUT = M / "stage3/out"
TMP = M / "stage3/tmp"
STOCK = "10.100.10.2:/home/keyspark/NewModels/MiMo-V2.6-Pro-RL-ARVQ-hybrid-63430f7"
MOPD = "/media/keyspark/My Book1/Local Models Archive/MiMo-V2.6-Pro-MOPD"
COUNT, CAP = int(os.environ.get("HOT_COUNT", 5560)), 192
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)


def select_hot(scores, count, cap=CAP):
    selected = {layer: [] for layer in range(1, 70)}
    for _, layer, e in sorted((-float(s), layer, e) for layer, v in scores.items() for e, s in enumerate(v)):
        if len(selected[layer]) < cap:
            selected[layer].append(e)
            count -= 1
            if count == 0:
                return {k: sorted(v) for k, v in selected.items()}
    raise ValueError("insufficient slots")


scores = {L: json.load(open(M / f"stage1/allocation_scores/layer{L}.json"))["scores"] for L in range(1, 70)}
pub = select_hot(scores, 1325)
new = select_hot(scores, COUNT)
cfg = json.load(open(M / "stage1/c1-out/config.json"))
books = cfg["quantization_config"]["aqlm_layer_books"]
for L in range(1, 70):
    b = books[str(L)]
    old_cold = b.get("cold_expert_ids") or list(range(384))
    assert sorted(set(range(384)) - set(old_cold)) == pub[L], f"published hot set mismatch L{L}"
    assert set(pub[L]) <= set(new[L])
changed = [L for L in range(1, 70) if len(new[L]) != len(pub[L])]
print(f"hot {COUNT}: {len(changed)} changed layers {changed[0]}..{changed[-1]}", flush=True)

src = Source(MOPD)
dev = "cuda"
only = [int(x) for x in sys.argv[1:]] or changed
for L in only:
    t0 = time.time()
    b = books[str(L)]
    old_cold = b.get("cold_expert_ids") or list(range(384))
    hot = new[L]
    cold = [e for e in range(384) if e not in set(hot)]
    done = OUT / f"done-{L:03d}.json"
    if not done.exists():
        # cold: keep the published rows for experts that stay cold
        rows = torch.tensor([old_cold.index(e) for e in cold])
        for tag in ("gateup", "down"):
            name = f"arvq-layer-{L:03d}-{tag}.safetensors"
            subprocess.run(["rsync", "-a", f"{STOCK}/{name}", str(TMP / name)], check=True)
            with safe_open(TMP / name, "pt") as f:
                meta = f.metadata()
                T = {}
                for k in f.keys():
                    t = f.get_tensor(k)
                    T[k] = t.index_select(0, rows).contiguous() if t.shape[0] == len(old_cold) and not k.endswith("_global") else t
            meta = dict(meta or {})
            meta["stage3"] = json.dumps({"n_cold": len(cold), "from_n_cold": len(old_cold)})
            save_file(T, str(OUT / (name + ".tmp")), metadata=meta)
            os.replace(OUT / (name + ".tmp"), OUT / name)
            (TMP / name).unlink()
            del T
        # hot: requantize every hot expert from MOPD
        rp = f"model.layers.{L}.mlp.experts."
        acc = {k: [] for k in ("w13_packed", "w13_bscale", "w13_scale2", "w2_packed", "w2_bscale", "w2_scale2")}
        for e in hot:
            g, u, d = (src.expert(L, e, p, dev) for p in ("gate_proj", "up_proj", "down_proj"))
            drop_source_expert(src, L, e)
            v = [nvfp4_quantize(x) for x in (g, u, d)]
            acc["w13_packed"].append(torch.cat((v[0][0], v[1][0])).cpu())
            acc["w13_bscale"].append(torch.cat((v[0][1], v[1][1])).cpu())
            acc["w13_scale2"].append(torch.stack((v[0][2], v[1][2])).cpu())
            acc["w2_packed"].append(v[2][0].cpu())
            acc["w2_bscale"].append(v[2][1].cpu())
            acc["w2_scale2"].append(v[2][2][None].cpu())
        kind = torch.full((384,), 2, dtype=torch.int8)
        kind[hot] = 0
        R = {rp + "hyb_kind": kind}
        ref = safe_open(M / f"stage2/rosters/roster-layer-{L:03d}.safetensors", "pt")
        for k, lst in acc.items():
            dt = ref.get_slice(rp + "nvfp4_" + k).get_dtype()
            t = torch.stack(lst)
            R[rp + "nvfp4_" + k] = t.view(torch.uint8) if dt == "U8" and t.dtype != torch.uint8 and t.element_size() == 1 else t.to({"U8": torch.uint8, "F32": torch.float32}[dt])
        # consistency: experts hot in both stage 2 and stage 3 must be bit-identical
        s2hot = pub[L]
        for k in acc:
            a = ref.get_tensor(rp + "nvfp4_" + k)
            idx = [hot.index(e) for e in s2hot]
            assert a.shape[1:] == R[rp + "nvfp4_" + k].shape[1:], (k, a.shape, R[rp + "nvfp4_" + k].shape)
            if idx:
                assert torch.equal(a, R[rp + "nvfp4_" + k][idx]), f"L{L} {k}: stage2 hot rows differ"
        save_file(R, str(OUT / f"roster-layer-{L:03d}.safetensors.tmp"))
        os.replace(OUT / f"roster-layer-{L:03d}.safetensors.tmp", OUT / f"roster-layer-{L:03d}.safetensors")
        done.write_text(json.dumps({"layer": L, "n_nvfp4": len(hot), "n_cold": len(cold), "seconds": round(time.time() - t0)}))
    print(f"L{L} hot {len(pub[L])}->{len(hot)} {json.loads(done.read_text())['seconds']}s", flush=True)
    torch.cuda.empty_cache()

for L in changed:
    hot = new[L]
    books[str(L)].update(n_nvfp4=len(hot), n_base=0, n_cold=384 - len(hot),
                         cold_expert_ids=[e for e in range(384) if e not in set(hot)])
cfg["quantization_config"]["aqlm_layer_books"] = books
json.dump(cfg, open(OUT / "config.json", "w"), indent=2)
json.dump({"hot_count": COUNT, "per_layer_cap": CAP, "method": "prepare_hybrid.select_hot extended (stage 3)",
           "layers": {str(L): {"hot": new[L]} for L in range(1, 70)}}, open(OUT / "allocation.json", "w"), indent=1)
print("STAGE3 BUILD DONE", flush=True)
