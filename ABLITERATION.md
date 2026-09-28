# Abliteration status

Status as of **2026-09-28 UTC: live serve is MOPD C3-ablit**. Two gated weight repos:

| Tree | HF | Thinking off |
|---|---|---|
| **MOPD C3-ablit** (live) | [drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-Abliterated) | **30/32 · 22/22** |
| RL dealign-op (non-MOPD) | [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated) | **32/32 · 22/22** |

Both copy decoder `o_proj` from [dealignai/MiMo-V2.6-Pro-RL-UNCENSORED](https://huggingface.co/dealignai/MiMo-V2.6-Pro-RL-UNCENSORED) v3 (25 of 29 layers: 32–45, 48–54, 64–67; DFlash/pad/front anchors left stock). C3-ablit applies that map as FP8 onto the MOPD hybrid-21 backbone. Stock C3 (not gated): 5/32 · 6/22.

## Gate (heuristic classifier)

| Mode | Refusal 32 | Cyber 22 |
|---|---:|---:|
| **MOPD C3-ablit, thinking off** (greedy, 192) | **30/32** bypass · 2 refuse (items 27, 29) | **22/22** bypass |
| MOPD C3 stock, thinking off | 5/32 | 6/22 |
| RL dealign-op, thinking **off** | **32/32** bypass | **22/22** bypass |
| RL dealign-op, thinking **on** (1024, visible content) | **25/32** bypass · 7 refuse | **16/22** bypass · 1 refuse · 2 garble · 3 empty |
| Prior l68t leftover-SRA (thinking off) | 11/32 | 15/22 |
| Stock RL ARVQ | 5/32 | 9/22 |

Thinking-off and thinking-on are the same weights. Choose per request with `chat_template_kwargs.enable_thinking`. A bypass label means the reply starts delivering the requested content. It does not certify correctness.

Canonical snapshot: [serve/verification/current-status.json](serve/verification/current-status.json). Gate logs on the lab host: `~/mimo26-arvq-tp4/ablit/work-dealign/`.

## Scope

These results describe this Abliterated Pro ARVQ tree, not stock Xiaomi MXFP4 or MiMo-V2.6-Flash. A bypass label does not certify correctness.

The four-node serve retains 1M context, MTP=2, four sequences, eager execution, BF16 KV, GPU memory fraction **0.85**, and MiMo tool parsers. Hermes executes `write_file` / `terminal` / `execute_code` so a prompt can build and run code.

**Weights:** gated [MOPD C3-ablit](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-Abliterated) or gated [RL ablit](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated). This GitHub repo is the Spark recipe; set `HOSTPATH` to the download. Image v4 is runtime-only. GPU memory fraction stays **0.85**.
