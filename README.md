# keys-MiMo-V2.6-Pro-MOPD C3 stock — 4 DGX Sparks, 1M context

Serving recipe for **[XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD)** in [Jarrelscy's ARVQ / NVFP4 hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) on **four NVIDIA DGX Spark (GB10)** nodes, tensor-parallel 4, **1,048,576-token context**.

**Weights: MOPD C3 stock (hybrid-21):**
**[drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21)**.

Pull the **prebuilt** image. Do not rebuild it:
`ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:latest` (`:63430f7-sm121-v4`).

The launcher enables MiMo tool calling on the server (`--enable-auto-tool-choice --tool-call-parser mimo --reasoning-parser mimo`). Hermes then **executes** those calls (`write_file`, `terminal`, `execute_code`, `read_file`, …) so a prompt can write a project, run it, and iterate. See [HERMES.md](HERMES.md) and [`serve/verify-tools-and-build.sh`](serve/verify-tools-and-build.sh).

## MOPD C3 (2026-09-28)

Xiaomi's MOPD build fixes tool-call repetition. We carried it into the ARVQ stack with a 21% NVFP4 hot set. Same prebuilt image and `serve/launch-rank.sh` defaults (MTP k=2, compile + CUDA graphs, 1M context, GPU util **0.85**). Details: [MOPD.md](MOPD.md).

| Weights | HF | Tool-call dup / flood | Refusal / cyber (thinking off) | Prose / code tok/s |
|---|---|---:|---:|---:|
| **MOPD C3 stock** | [hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21) | **7.4% / 0.9%** | 5/32 · 6/22 | **24.2 / 32.2** |

GSM8K 96.8 / HumanEval 93.9 / MMLU-Pro 77.4 are on this stock C3 tree.

## Current status — 2026-09-28 UTC (image v4 + MOPD C3 stock)

- **Decode on live C3 stock: 32.2 tok/s on code, 24.2 tok/s on prose**, single stream. See [SPARK-PORT.md §9](SPARK-PORT.md#9-attention-o_proj-in-fp8-weights-update-2026-09-27) for FP8 `o_proj`.
- **All three MTP draft heads run**, non-chain, in the V2 runner. On C3 stock code, drafting accepts 2.60 tokens per pass at k=2.
- **Prefill (C3 stock): 883 tok/s at 9.5K, 1074 tok/s at 38K**. NCCL drives both PCIe paths of the cabled CX-7 port; see [SPARK-PORT.md §10](SPARK-PORT.md#10-nccl-over-both-pcie-paths-of-the-cabled-cx-7-port-2026-09-27).
- **✅ Tool-call loop fixed.** Truncated tool batches return `finish_reason: "length"`, and the output cap is 8192. See [HERMES.md](HERMES.md#fixed-2026-09-26-never-ending-tool-call-loop).
- **Safer defaults for clients that send nothing** (for example Pi): temperature 0.7 and thinking off unless requested. See [Clients](#clients).
- **Vision:** the live serve is text-only. This repo publishes stock MOPD C3 only.

Canonical snapshot: [serve/verification/current-status.json](serve/verification/current-status.json).

## Credit

The quantization is Jarrelscy's. Official MiMo-V2.6-Pro images read the source MXFP4 / FP8 expert layout. They do not load this checkpoint. Jarrelscy's hybrid keeps a small hot-expert set in NVFP4 and the remaining routed experts in ARVQ codebooks, which is what fits the model across four 128 GB Sparks.

| Piece | Author | Where |
|---|---|---|
| ARVQ / NVFP4 hybrid format | Jarrelscy | [jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid](https://huggingface.co/jarrelscy/MiMo-V2.6-Pro-RL-ARVQ-hybrid) @ `63430f7b9c1b13f4bfca9e3bc3969ec0115d1a88` |
| vLLM fork that loads `nvfp4_arvq_hybrid` | Jarrelscy | [jarrelscy/vllm-mimo-v26-arvq-sm120](https://github.com/jarrelscy/vllm-mimo-v26-arvq-sm120) @ `88c94233120247f275ec94baf21638321a930469` |
| MOPD base model | Xiaomi MiMo | [XiaomiMiMo/MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) |
| MOPD C3 stock (hybrid-21) | Keys | [drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21) |
| Four-Spark serve, MTP measurement, Spark port notes, tool/build loop | Keys | this repo |

Jarrelscy marks full-model quality and SM120 / 1M serving as **unqualified**. The numbers below are a serving measurement on GB10 (SM121), not a quality claim.

## Champion

**MTP = 2** draft tokens: the best setting for prose and for concurrent requests. The draft is the checkpoint's own three-head MTP stack (`model.mtp.layers.0-2`). v4 runs the heads **non-chain**, the way Xiaomi's SGLang deploy does: head k reads the target's hidden state plus the token k+1 ahead. The fork as published ran head 0 for every draft step; see [SPARK-PORT.md §6](SPARK-PORT.md#6-all-three-mtp-heads-non-chain-on-the-v2-runner-image-v4-2026-09-27). **MTP = 3** is faster on code but slower on prose.

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
| Checkpoint | **MOPD C3 stock** (hybrid-21). |

Measured KV pool on C3: **1,282,005 tokens**. Weights about **83.2 GiB per rank**. Still covers the 1M context.

### Speed, live MOPD C3 (image v4, 2026-09-28)

512 new tokens, temperature 1.0, top_p 0.95, thinking off, one request at a time. Prose is the mean of three literary stories (beekeeper, lighthouse, nurse). Code is the mean of two tasks (a red-black tree, a lexer and parser). Reproduce with [`serve/bench/speedbench.py`](serve/bench/speedbench.py).

| Checkpoint | **Prose** | **Code** | Prose tok/pass | Code tok/pass | 4-wide agg | Prefill 9.5K / 38K |
|---|---:|---:|---:|---:|---:|---:|
| **C3 stock** | **24.2 tok/s** | **32.2 tok/s** | 1.91 | 2.60 | **46.3** | 883 / 1074 |

Creative stories at temperature 1.0 are the hardest text to predict. Prefill slows as prompts grow because the 10 full-attention layers grow with context length. The batched ARVQ kernels give the big prefill gain. Driving both CX-7 PCIe paths adds 14–25% on BF16 weights; FP8 `o_proj` speeds decode. Single runs vary by roughly ±10% on GB10 (unified-memory page migration).

## Image (prebuilt — pull, do not rebuild)

**`ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:latest`**, the same image as `:63430f7-sm121-v4`. Public, no login. This is the runtime for MOPD C3 stock. `serve/launch-rank.sh` pins `:63430f7-sm121-v4`.

Digest `sha256:6c6b6aed088da63452ccc280cc0b88e3931ae7de5c27126053d0332b196769cf`.

The image contains:
- Jarrelscy's fork compiled for GB10 (`sm_121a`), plus the Spark loader fixes.
- **All three MTP heads running non-chain on the V2 `MTPSpeculator`**, inside the draft-prefill CUDA graph.
- **Expert-batched ARVQ prefill kernels** (`grouped.cu`, built during the image build).
- The tool-call loop fix and torch.compile-clean capture hooks.

The recipe is [`serve/image/Dockerfile`](serve/image/Dockerfile), built on [`Dockerfile.base`](serve/image/Dockerfile.base). A rebuild reproduces the published image file for file (17 checks). The image does not contain the weights — download hybrid-21 separately.

## Bring-up (one-shot = C3 stock + prebuilt v4)

You need four DGX Sparks on the 200G RoCE fabric. The launcher's defaults **are** the champion configuration:
- prebuilt v4 image, MTP k=2 across all three heads, CUDA graphs
- batched prefill, 1M context, 4 sequences, GPU memory fraction 0.85
- tool-call loop fix and server sampling defaults

```bash
# 1. Weights, on storage all four nodes can read. MOPD C3 stock:
hf download drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21 --local-dir /path/to/mimo-arvq

# 2. Recipe, on each node
git clone https://github.com/drowzeys/keys-MiMo-V2.6-Pro-MOPD-C3-Jarrelscy-ARVQ-4-DGX-Sparks-1M-Context
cd keys-MiMo-V2.6-Pro-MOPD-C3-Jarrelscy-ARVQ-4-DGX-Sparks-1M-Context
export MASTER_ADDR=<rank-0 IP>

# 3. Prebuilt image is pulled by launch-rank.sh (ghcr.io/drowzeys/mimo-v26-pro-arvq-spark:63430f7-sm121-v4)
#    Start ranks 1-3 first (headless), then rank 0 (the API on :8888)
bash serve/launch-rank.sh <this-node-IP> <1|2|3> <RoCE-GID-index> /path/to/mimo-arvq headless
bash serve/launch-rank.sh <this-node-IP> 0 <RoCE-GID-index> /path/to/mimo-arvq api
```

`NCCL_IB_HCA` defaults to both RoCE devices of the cabled port (`rocep1s0f1,roceP2p1s0f1`). Check the names with `ibdev2netdev`: both should show the same port Up. The only per-node value you must set is the **RoCE GID index**: the IPv4 RoCE entry for the HCA, from `show_gids`. It was 3 on three of our Sparks and 7 on one. For code-heavy use, add `SPEC='{"method":"mtp","num_speculative_tokens":3}'`. NCCL uses the 200G RoCE NIC (`NCCL_NET=IB`). See [SPARK-PORT.md](SPARK-PORT.md) for the port notes.

One-shot validation of this recipe on live C3 stock (prebuilt GHCR image + `serve/launch-rank.sh` defaults, 2026-09-28):

| Measure | Result |
|---|---:|
| Prose | 24.2 tok/s |
| Code | 32.2 tok/s |
| 4 requests, aggregate | 46.3 tok/s |
| 38K prefill | 1,074 tok/s |

## Integration and experiments

- **[MOPD](MOPD.md)** — MOPD integration: candidates C0–C3, tool-call repetition, quality and speed; build scripts in [`mopd/`](mopd/).
- **[Hermes](HERMES.md)** — parsers, Hermes execution, and build-from-prompt (`write_file` + `terminal`).
- **[DFlash](DFLASH.md)** — measured on the old eager build (13.0 tok/s prose). Slower than MTP. Not the champion.


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

A request sent without `chat_template_kwargs` returned no reasoning block, and one with `enable_thinking: true` did.

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
