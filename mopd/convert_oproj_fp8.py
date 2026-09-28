#!/usr/bin/env python3
"""Quantize the target's BF16 attention o_proj to FP8 e4m3 with 128x128 block
scales (the format qkv_proj already uses) and write a new backbone shard.

Xiaomi's source_fp8.ignored_layers lists every model.layers.N.self_attn.o_proj,
so these stayed BF16 (6144 x 16384 each, 70 layers). On the TP4 serve that is
~3.5 GB of BF16 reads per rank per decode step (profile: ~17 ms of ~93 ms).

Writes to --out:
  backbone-001.safetensors  o_proj.weight -> float8_e4m3fn, + o_proj.weight_scale_inv [48,128] fp32
  config.json               the decoder o_proj entries removed from source_fp8.ignored_layers
  model.safetensors.index.json  the new scale keys mapped to backbone-001
MTP heads (model.mtp.*), audio and vision o_proj stay BF16.
"""
import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, "/home/keyspark/mimo26-arvq-tp4/dflash-ft/train")
from mtp_ref import dequant_block, quant_block  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/home/keyspark/mimo26-arvq-tp4/dflash-ft/src-ckpt")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

OPROJ = re.compile(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")
idx = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
wm = idx["weight_map"]
targets = sorted((k for k in wm if OPROJ.match(k)), key=lambda k: int(OPROJ.match(k).group(1)))
files = sorted({wm[k] for k in targets})
assert files == ["backbone-001.safetensors"], files
print(f"{len(targets)} decoder o_proj tensors in {files}")

src = os.path.join(a.src, "backbone-001.safetensors")
out, worst = {}, 0.0
with safe_open(src, "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        t = f.get_tensor(k)
        if OPROJ.match(k):
            assert t.dtype == torch.bfloat16 and tuple(t.shape) == (6144, 16384), (k, t.dtype, t.shape)
            q, s = quant_block(t.float())
            rel = ((dequant_block(q, s) - t.float()).norm() / t.float().norm()).item()
            worst = max(worst, rel)
            out[k] = q
            out[k + "_scale_inv"] = s
            wm[k + "_scale_inv"] = "backbone-001.safetensors"
        else:
            out[k] = t
print(f"quantized {len(targets)} o_proj; worst relative Frobenius error {worst:.4f}")
save_file(out, os.path.join(a.out, "backbone-001.safetensors"), metadata=meta)

cfg = json.load(open(os.path.join(a.src, "config.json")))
ign = cfg["quantization_config"]["source_fp8"]["ignored_layers"]
keep = [x for x in ign if not re.match(r"^model\.layers\.\d+\.self_attn\.o_proj$", x)]
print(f"ignored_layers {len(ign)} -> {len(keep)} (kept: {keep})")
cfg["quantization_config"]["source_fp8"]["ignored_layers"] = keep
if "text_config" in cfg and "quantization_config" in cfg["text_config"]:
    tq = cfg["text_config"]["quantization_config"].get("source_fp8")
    if tq and "ignored_layers" in tq:
        tq["ignored_layers"] = [x for x in tq["ignored_layers"]
                                if not re.match(r"^model\.layers\.\d+\.self_attn\.o_proj$", x)]
json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
json.dump(idx, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
print("wrote", a.out)
