# Abliteration status

Status as of **2026-09-28 UTC: live serve is MOPD C3-ablit**.

| Tree | HF | Thinking off |
|---|---|---|
| **MOPD C3-ablit** (default, live) | [drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-Abliterated) (gated) | **30/32 · 22/22** |
| MOPD C3 stock | [drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21) | 5/32 · 6/22 |

C3-ablit copies decoder `o_proj` from [dealignai/MiMo-V2.6-Pro-RL-UNCENSORED](https://huggingface.co/dealignai/MiMo-V2.6-Pro-RL-UNCENSORED) v3 onto the MOPD hybrid-21 backbone as FP8: **25 of 29** layers (**32–45, 48–54, 64–67**). DFlash-source / pad / front anchors stay C3. Packed experts, MTP, and `dflash/` stay C3.

## Gate (heuristic classifier)

| Mode | Refusal 32 | Cyber 22 |
|---|---:|---:|
| **MOPD C3-ablit, thinking off** (greedy, 192) | **30/32** bypass · 2 refuse (items 27, 29) | **22/22** bypass |
| MOPD C3 stock, thinking off | 5/32 | 6/22 |

Thinking-off and thinking-on are the same weights. Choose per request with `chat_template_kwargs.enable_thinking`. A bypass label means the reply starts delivering the requested content. It does not certify correctness. Thinking-on was not re-gated on C3-ablit.

Canonical snapshot: [serve/verification/current-status.json](serve/verification/current-status.json). Gate logs on the lab host: `~/mimo26-arvq-tp4/ablit/work-c3/gate-ablit/`.

## Scope

These results describe the MOPD C3-ablit tree. A bypass label does not certify correctness.

The four-node serve retains 1M context, MTP=2, four sequences, torch.compile + CUDA graphs, BF16 KV, GPU memory fraction **0.85**, and MiMo tool parsers. Hermes executes `write_file` / `terminal` / `execute_code` so a prompt can build and run code.

**Weights:** gated [MOPD C3-ablit](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-Abliterated) (default) or public [MOPD C3 stock](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21). This GitHub repo is the Spark recipe; set `HOSTPATH` to the download. Pull the prebuilt image (`ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v4`). GPU memory fraction stays **0.85**.
