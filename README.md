# keys-MiMo-V2.6-Pro-RL Jarrelscy ARVQ Abliterated — 4 DGX Sparks, 1M context

Serving recipe for the **abliterated** [Jarrelscy ARVQ / NVFP4 hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) of **[XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL)** on **four NVIDIA DGX Spark (GB10)** nodes, tensor-parallel 4, **1,048,576-token context**.

Gated weights (automatic approval after terms): **[drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated)**.

The launcher enables MiMo tool calling on the server (`--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo`). Hermes then **executes** those calls (`write_file`, `terminal`, `execute_code`, `read_file`, …) so a prompt can write a project, run it, and iterate. See [HERMES.md](HERMES.md) and [`serve/verify-tools-and-build.sh`](serve/verify-tools-and-build.sh).

## New 2026-09-28: MOPD build (recommended for agents and tool use)

Xiaomi's [MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) fixes tool-call repetition. We carried it into the ARVQ stack with a 21% NVFP4 hot set: **[drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21)**. It uses the same image and launcher. It is **not abliterated**. Details are in [MOPD.md](MOPD.md).

| | Turns with a repeated tool call | Flooding turns (32+ calls) | GSM8K | HumanEval | MMLU-Pro | Prose / code tok/s |
|---|---:|---:|---:|---:|---:|---:|
| RL abliterated (this repo's weights) | 53% | 9.0% | — | — | — | 24.5 / 34.1 |
| **MOPD hybrid-21** | **7.4%** | **0.9%** | 96.8% | 93.9% | 77.4% | 23.2 / 31.3 |

The abliterated RL build below stays available for uncensored use. An abliterated MOPD build is not published yet.

## Current status — 2026-09-27 UTC (image v4 + FP8 o_proj weights)

- **Decode: 34.1 tok/s on code, 24.5 tok/s on prose**, single stream (up from 18.6 prose on the old eager build). The latest step is FP8 attention `o_proj` weights: +12% decode with NLL +0.28%. See [SPARK-PORT.md §9](SPARK-PORT.md#9-attention-o_proj-in-fp8-weights-update-2026-09-27).
- **All three MTP draft heads now run**, non-chain, in the V2 runner. Before v4, only head 0 ever drafted. On code, drafting accepts 2.6 tokens per pass at k=2 and 3.2–3.35 at k=3.
- **Prefill: ~950–1,290 tok/s** (was 128). 38K-token time to first token: **~36 s** (was 302 s). NCCL now drives both PCIe paths of the one cabled CX-7 port; see [SPARK-PORT.md §10](SPARK-PORT.md#10-nccl-over-both-pcie-paths-of-the-cabled-cx-7-port-2026-09-27).
- **✅ Tool-call loop fixed.** Truncated tool batches return `finish_reason: "length"`, and the output cap is 8192. See [HERMES.md](HERMES.md#fixed-2026-09-26-never-ending-tool-call-loop).
- **Safer defaults for clients that send nothing** (for example Pi): temperature 0.7 and thinking off unless requested. See [Clients](#clients).
- **[Abliteration](ABLITERATION.md):** live `dealign-op` tree. Thinking **off** **32/32** refusal and **22/22** cyber; thinking **on** 25/32 and 16/22 (visible content). Gated HF: [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated).
- **Vision:** the live serve is text-only.

Canonical snapshot: [serve/verification/current-status.json](serve/verification/current-status.json) (thinking-off 32/32 · 22/22, thinking-on 25/32 · 16/22, live `write_file`+`terminal` build stdout 42). The earlier [2026-09-25-status.json](serve/verification/2026-09-25-status.json) is the pre-champion tool-parser check on l68t.

## Credit

The quantization is Jarrelscy's. Official MiMo-V2.6-Pro images read the source MXFP4 / FP8 expert layout. They do not load this checkpoint. Jarrelscy's hybrid keeps a small hot-expert set in NVFP4 and the remaining routed experts in ARVQ codebooks, which is what fits the model across four 128 GB Sparks.

| Piece | Author | Where |
|---|---|---|
| ARVQ / NVFP4 hybrid checkpoint | Jarrelscy | [jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) @ `63430f7b9c1b13f4bfca9e3bc3969ec0115d1a88` |
| vLLM fork that loads `nvfp4_arvq_hybrid` | Jarrelscy | [jarrelscy/vllm-mimo-v26-arvq-sm120](https://github.com/jarrelscy/vllm-mimo-v26-arvq-sm120) @ `88c94233120247f275ec94baf21638321a930469` |
| Base model | Xiaomi MiMo | [XiaomiMiMo/MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) |
| Abliterated ARVQ weights | Keys | [drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated) |
| MOPD base model | Xiaomi MiMo | [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) |
| MOPD ARVQ hybrid-21 weights | Keys | [drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21) |
| Four-Spark serve, MTP measurement, Spark port notes, tool/build loop | Keys | this repo |

Jarrelscy marks full-model quality and SM120 / 1M serving as **unqualified**. The numbers below are a serving measurement on GB10 (SM121), not a quality claim.

## Champion

**MTP = 2** draft tokens: the best setting for prose and for concurrent requests. The draft is the checkpoint's own three-head MTP stack (`model.mtp.layers.0-2`). v4 runs the heads **non-chain**, the way Xiaomi's SGLang deploy does: head k reads the target's hidden state plus the token k+1 ahead. The fork as published ran head 0 for every draft step; see [SPARK-PORT.md §6](SPARK-PORT.md#6-all-three-mtp-heads-non-chain-on-the-v2-runner-image-v4-2026-09-27). **MTP = 3** is faster on code (32–34 tok/s) but slower on prose.

| | |
|---|---|
| Nodes | 4× DGX Spark GB10, one GPU each, TP4, PP1 |
| Context | `1048576` |
| KV | BF16 (`--kv-cache-dtype` left at auto). FP8 KV is not used |
| GPU memory fraction | **0.85** (do not raise this on GB10) |
| Scheduler | `max-num-seqs 4`, `max-num-batched-tokens 5120`, chunked prefill, prefix caching |
| Execution | torch.compile + CUDA graphs (`FULL_AND_PIECEWISE`). Compile cache persisted in `/var/tmp/mimo-arvq-vllm-cache` |
| ARVQ prefill | **expert-batched CUDA prefill** (v3: `VLLM_ARVQ_BATCHED_PREFILL=1`, chunks ≥32 tokens), fused decode activation pack (image defaults) |
| Modalities | `--language-model-only` so the startup profile does not spend the KV budget on a video |
| Draft | `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` |
| Served name | `MiMo-V2.6-Pro-ARVQ` |
| Tool calls | `--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo` (required; without these Hermes `tool_choice: auto` is HTTP 400) |
| Server defaults | `--override-generation-config '{"max_new_tokens": 8192, "temperature": 0.7, "top_p": 0.95}'` and `--default-chat-template-kwargs '{"enable_thinking": false}'`. These apply only when a client sends no values of its own; see [Clients](#clients). |
| Checkpoint | abliterated tree `…-ablit-dealign-op` / gated HF repo above |

Measured KV pool on the champion boot: about **2.07M tokens** (three MTP heads now hold KV). Weights about **73.2 GiB per rank**.

### Speed, current champion (image v4, 2026-09-27)

512 new tokens, temperature 1.0, top_p 0.95, thinking off, one request at a time. Prose is the mean of three literary stories (beekeeper, lighthouse, nurse). Code is the mean of two tasks (a red-black tree, a lexer and parser). Reproduce with [`serve/bench/speedbench.py`](serve/bench/speedbench.py).

| Draft tokens | **Prose** | **Code** | Prose tokens/pass | Code tokens/pass |
|---:|---:|---:|---:|---:|
| **2 (default), FP8 o_proj weights** | **24.5 tok/s** | **34.1 tok/s** | 1.89 | 2.67 |
| 2, BF16 o_proj (weights before 2026-09-27) | 21.8 tok/s | 30.9 tok/s | 1.83 | 2.64 |
| 3, BF16 o_proj | 19.9 tok/s | 32.3–34.0 tok/s | 1.93 | 3.19–3.35 |
| old build (head 0 only, k=2) | 20.4 tok/s | — | 1.82 | — |

Requests overlapped (`max_num_seqs 4`), MTP=2, prose:

| Requests | Aggregate | Per request |
|---:|---:|---:|
| 1 | 24.5 tok/s | 24.5 tok/s |
| 2 | 36.0 tok/s | 18.0 tok/s |
| 4 | **48.5 tok/s** | 12.1 tok/s |

Creative stories at temperature 1.0 are the hardest text to predict. On ordinary held-out prose, the heads reach 2.32 tokens/pass at k=3 (0.67 / 0.41 / 0.24 per position), and code reaches 2.98.

Uncached prefill (nonce prompt, `max_tokens` 1):

| Prompt | Old eager recipe | Grouped prefill (superseded) | **Now: batched prefill** | Time to first token, now |
|---:|---:|---:|---:|---:|
| 9.5K tokens | 128 tok/s | 505 tok/s | **951–1,291 tok/s** | **7.4–10.0 s** |
| 38K tokens | 126 tok/s | 527 tok/s | **1,038–1,052 tok/s** | **~36 s** |
| 152K tokens | — | — | 615 tok/s | 247 s |

Prefill slows as prompts grow because the 10 full-attention layers grow with context length. The batched ARVQ kernels give the big prefill gain. Driving both CX-7 PCIe paths adds 14–25% on BF16 weights; FP8 `o_proj` gives some of that back at prefill, where the math is compute-bound, while speeding up decode by 12%. Single runs vary by roughly ±10% on GB10 (unified-memory page migration), so the table shows ranges.

## Image

**`ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:latest`**, the same image as `:63430f7-sm121-v4`. It is public, needs no login, and is the only supported image. `serve/launch-rank.sh` pins `:63430f7-sm121-v4`.

Digest `sha256:6c6b6aed088da63452ccc280cc0b88e3931ae7de5c27126053d0332b196769cf`.

The image contains:
- Jarrelscy's fork compiled for GB10 (`sm_121a`), plus the Spark loader fixes.
- **All three MTP heads running non-chain on the V2 `MTPSpeculator`**, inside the draft-prefill CUDA graph.
- **Expert-batched ARVQ prefill kernels** (`grouped.cu`, built during the image build).
- The tool-call loop fix and torch.compile-clean abliteration hooks.

The recipe is [`serve/image/Dockerfile`](serve/image/Dockerfile), built on [`Dockerfile.base`](serve/image/Dockerfile.base). A rebuild reproduces the published image file for file (17 checks). The image does not contain the weights.

## Bring-up (best setup = the defaults)

You need four DGX Sparks on the 200G RoCE fabric. The launcher's defaults **are** the champion configuration:
- v4 image, MTP k=2 across all three heads, CUDA graphs
- batched prefill, 1M context, 4 sequences, GPU memory fraction 0.85
- tool-call loop fix and server sampling defaults

```bash
# 1. Weights, on storage all four nodes can read. Pick one:
#    MOPD hybrid-21 (tool-call repetition fixed, not abliterated):
hf download drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21 --local-dir /path/to/mimo-arvq
#    or RL abliterated (gated repo: accept the terms once):
hf download drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated --local-dir /path/to/mimo-arvq

# 2. Recipe, on each node
git clone https://github.com/drowzeys/keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated-4-DGX-Sparks-1M-Context
cd keys-MiMo-V2.6-Pro-RL-Jarrelscy-ARVQ-Abliterated-4-DGX-Sparks-1M-Context
export MASTER_ADDR=<rank-0 IP>

# 3. Start ranks 1-3 first (headless), then rank 0 (the API on :8888)
bash serve/launch-rank.sh <this-node-IP> <1|2|3> <RoCE-GID-index> /path/to/mimo-arvq headless
bash serve/launch-rank.sh <this-node-IP> 0 <RoCE-GID-index> /path/to/mimo-arvq api
```

`NCCL_IB_HCA` defaults to both RoCE devices of the cabled port (`rocep1s0f1,roceP2p1s0f1`). Check the names with `ibdev2netdev`: both should show the same port Up. The only per-node value you must set is the **RoCE GID index**: the IPv4 RoCE entry for the HCA, from `show_gids`. It was 3 on three of our Sparks and 7 on one. For code-heavy use, add `SPEC='{"method":"mtp","num_speculative_tokens":3}'`. NCCL uses the 200G RoCE NIC (`NCCL_NET=IB`). See [SPARK-PORT.md](SPARK-PORT.md) for the port notes.

## Integration and experiments

- **[MOPD](MOPD.md)** — MOPD integration: candidates C0–C3, tool-call repetition, quality and speed; build scripts in [`mopd/`](mopd/).
- **[Hermes](HERMES.md)** — parsers, Hermes execution, and build-from-prompt (`write_file` + `terminal`).
- **[DFlash](DFLASH.md)** — measured on the old eager build (13.0 tok/s prose). Slower than MTP. Not the champion.
- **[Abliteration](ABLITERATION.md)** — live dealign-op: thinking-off 32/32 · 22/22; thinking-on 25/32 · 16/22.

## Clients

**Use `/v1/chat/completions`.** The raw `/v1/completions` endpoint skips the chat template, so an instruct model just continues the prompt text: echoing, garbling, and no tool calls. That is expected behaviour, not a model fault.

Server defaults, which a client's own values always override:

| Setting | Default | Why |
|---|---|---|
| `max_tokens` | 8192 | The checkpoint's 2048 truncated parallel tool batches, which caused the tool-call loop. |
| `temperature` / `top_p` | **0.7** / 0.95 | Some clients (for example Pi) send no sampling parameters. At the checkpoint's 1.0, about 0.3% of samples lock into a repetition loop (about 2% for long thinking-on outputs). See the measurement below. |
| `enable_thinking` | **false** | The chat template turns thinking **on** when the kwarg is missing. Pass `"chat_template_kwargs": {"enable_thinking": true}` to opt in. |

Measured on image v4, 2026-09-27. We took the 19 prompts whose temperature-1.0 samples had looped in our 6,000-sample on-policy run and sampled each 3 times:

| Setting | Samples that looped |
|---|---:|
| Old behaviour (temperature 1.0, thinking on) | 5 / 57 |
| **New defaults** (client sends nothing) | **2 / 57** |

A request sent without `chat_template_kwargs` returned no reasoning block, and one with `enable_thinking: true` did. One-shot validation of v4 (GHCR pull plus `serve/launch-rank.sh` with defaults):

| Measure | Result |
|---|---:|
| Prose | 22.4 tok/s |
| Code | 30.5 tok/s |
| 4 requests, aggregate | 48.0 tok/s |
| 38K prefill | 1,014 tok/s |

## Build from a prompt

After the four ranks are up and Hermes points at `http://<rank-0>:8888/v1` model `MiMo-V2.6-Pro-ARVQ`:

```bash
# server + Hermes execution (write a file, run it, check output)
bash serve/verify-tools-and-build.sh http://127.0.0.1:8888/v1

# or a free-form build:
hermes chat -q "Create /tmp/demo/app.py that prints hello and run it. Use write_file then terminal." --oneshot --yolo
```

The model must emit tool calls. Hermes runs them on the host (`terminal.backend: local`). Do not leave `tool_use_enforcement` off for this checkpoint — MiMo otherwise narrates the build instead of calling tools.

## License

Recipe text in this repo is MIT. The checkpoint and the vLLM fork keep their own licenses (the uploaded snapshot's card is MIT; the fork follows upstream vLLM).
