#!/usr/bin/env python3
"""C1 = stock RL-ARVQ backbone shards with MOPD's non-expert tensors swapped in.

backbone-001 (575 decoder non-expert tensors) <- MOPD model_pp0_ep0_shard0
backbone-002 (48 MTP-head tensors)           <- MOPD model_mtp
The stock tensors are byte-identical to the RL source (checked), so this is the
exact MOPD non-expert state on top of Jarrelscy's RL expert quantization.
Every swapped tensor must match name, dtype and shape; relative L2 change is logged.
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MOPD = "/media/keyspark/My Book1/Local Models Archive/MiMo-V2.6-Pro-MOPD"
S = "/home/keyspark/mimo26-arvq-tp4/mopd/stage1"
PAIRS = [("backbone-001.safetensors", f"{S}/stock-backbone-001.safetensors", f"{MOPD}/model_pp0_ep0_shard0.safetensors"),
         ("backbone-002.safetensors", f"{S}/stock-backbone-002.safetensors", f"{MOPD}/model_mtp.safetensors")]
only = sys.argv[1:] or [p[0] for p in PAIRS]
report = {}
for name, stock, mopd in PAIRS:
    if name not in only:
        continue
    out, rows = {}, []
    with safe_open(stock, "pt") as s, safe_open(mopd, "pt") as m:
        meta = s.metadata()
        mk = set(m.keys())
        for k in s.keys():
            t = s.get_tensor(k)
            if k in mk:
                n = m.get_tensor(k)
                assert n.dtype == t.dtype and n.shape == t.shape, (k, n.dtype, n.shape, t.dtype, t.shape)
                if n.dtype.is_floating_point and n.element_size() > 1:
                    rel = ((n.float() - t.float()).norm() / t.float().norm().clamp_min(1e-12)).item()
                else:
                    rel = float(not torch.equal(n.view(torch.uint8), t.view(torch.uint8)))
                rows.append((k, rel))
                t = n
            out[k] = t
    save_file(out, f"{S}/src-c1/{name}", metadata=meta)
    changed = [r for r in rows if r[1] > 0]
    report[name] = {"swapped": len(rows), "changed": len(changed),
                    "top": sorted(changed, key=lambda r: -r[1])[:10]}
    print(name, json.dumps(report[name]), flush=True)
json.dump(report, open(f"{S}/c1-swap-report.json", "w"), indent=1)
