# Spark port notes

These are the fixes between Jarrelscy's `sm120` fork (`88c94233`) and the first successful token on GB10 / SM121. The fork is still the runtime. These notes are the delta.

## 1. Flash-attention package

The fork tree shadows the base vLLM install and ships a `vllm_flash_attn` stub that exports `compile_flash_attn_varlen_func_from_specs` but has no `layers/`, `cute/`, or `ops/`. Replacing the whole package with the base image's copy removes that symbol and model inspection fails. Leaving the stub makes `vllm.vllm_flash_attn.layers` fail after the weights load.

`serve/entrypoint.sh` keeps the fork's `vllm_flash_attn/__init__.py` and links only the missing subpackages from the base image. `cute/` is linked as a real directory of symlinks. A symlink of the `cute` directory itself makes the fork treat it as upstream flash-attn source and rewrite `sys.modules`.

## 2. MTP QKV scales

Pro QKV is stored grouped per KV head, packed as eight groups. The scale tensor is 216 rows. A plain TP4 split yields `[54, 48]`. The parameter is `[53, 48]`.

The target model already requantizes that layout in `_shard_fp8_qkv_proj`. The MTP loader has to call the same function. After that, every rank gets weight `[6784, 6144]` and scale `[53, 48]`.

The checkpoint contains MTP layers 0, 1, and 2. `MiMoV2MultiTokenPredictor` in the fork sets `num_mtp_layers = 1`, so extra speculative tokens replay layer 0. That is the configuration behind the MTP=2 champion number.

## 3. Fused SiLU op import

Loading the MTP draft imports the activation-quant fusion pass before `vllm._C` has registered `silu_and_mul_quant`. The op exists in this image. Import `vllm._C` at the top of `act_quant_fusion.py` before the fused-op table is built.

## 4. Memory

`--language-model-only` is required for the 1M text pool on this build. Without it, startup profiles one maximum-size video and the KV budget drops by several GiB per rank.

`--gpu-memory-utilization` stays at **0.85**. Do not raise it to buy context.

Expert parallelism is unsupported in this quant method. The serve is TP4 only. Expert tensors stay compressed at decode.

## 5. RoCE

`NCCL_NET=IB` on the 200G NIC. During one short generation each node's RoCE transmit counter moved by tens of mebibytes while the TCP counters on that NIC moved kilobytes. The GID index is per node: an index that is correct on three Sparks was wrong on the fourth.

## 6. All three MTP heads, non-chain, on the V2 runner (image v4, 2026-09-27)

The checkpoint ships three MTP heads (`model.mtp.layers.0-2`), one per draft step. Xiaomi's SGLang deploy runs them as multi-layer EAGLE in **non-chain** mode (`multi_layer_eagle_worker_v2`): every head reads the **target's** hidden state `h[i]` and the token `x[i+1+k]` at position `i`.

**What was wrong.** This vLLM fork runs **Model Runner V2**. Its `MTPSpeculator`, inherited from `AutoRegressiveSpeculator`, drafts step 0 over the verified span. It then drafts steps 1.. as one-token decodes that feed the previous step's output hidden state back in, and it never passes `spec_step_idx`, so **head 0 ran every draft step**. Earlier images also built only one head. Live per-position acceptance on prose was about 0.61 / 0.20 / 0.05: heads 1-2 had never run. (A V1-runner proposer patch in image v2/v3 was dead code on this runner.)

**Fix, in image v4 (`mimo_v2_mtp.py`).**
- **Build.** `MIMO_MTP_LAYERS=3` builds all three heads.
- **Routing.** On `MTPSpeculator` only, the draft prefill runs head 0 unchanged, then heads 1..K-1 over the same span. They use the saved target hidden states, the same positions, attention metadata and slots, input ids rotated left once per head, and head (k-1)'s draft in each request's last valid slot. `propose()` then skips the chained decode steps. All K drafts come out of the draft-prefill routine, so V2's FULL draft-prefill CUDA graph captures every head.
- **Count.** The number of heads comes from `speculative_config.draft_model_config.hf_config.n_predict`. Inside the drafter, `vllm_config.model_config` is the target's config.

**Verification.** On-policy data was captured, and the heads were re-implemented in plain PyTorch (`dflash-ft/train/mtp_ref.py`) and simulated step by step. On the same 12 held-out prompts, live now matches the simulation:

| Same prompts | pos 0 | pos 1 | pos 2 | tokens/pass (k=3) |
|---|---:|---:|---:|---:|
| prose, live v4 | 0.666 | 0.405 | 0.244 | 2.32 |
| prose, simulation | 0.680 | 0.420 | 0.262 | 2.36 |
| code, live v4 | 0.806 | 0.658 | 0.516 | 2.98 |
| code, simulation | 0.846 | 0.709 | 0.574 | 3.13 |

**Not implemented: the boundary stash.** Head k's last k slots keep draft-based inputs and are not rewritten. SGLang recomputes them from stashed target hidden states; simulation puts that at about 2% of tokens/pass.

**CUDA graphs.** Capturing heads 1-2 in the prefill graph was correct but did not lower pass time. Each extra head costs about 13 ms per pass, dominated by TP4 all-reduces over RoCE and the vocab-parallel logits gather, not by kernel launches. So **k=2 is the default**, best for prose and concurrency; **k=3 is best for code**.

## 7. Speed path (2026-09-26)

- **CUDA graphs.** torch.compile with `FULL_AND_PIECEWISE` works on this fork. The ablit capture hooks in `mimo_v2.py` used to do a file `open()` per layer per forward. That meant 140 syscalls per step and a graph break. They are now gated once at import (`MIMO_ABLIT_CAPTURE=1` re-enables them).
- **ARVQ prefill.** By default, prefill ran through the per-slot decode kernel, at 128 tok/s. `VLLM_ARVQ_GROUPED_PREFILL=1`, `_COMPACT_PREFILL=1` and `_SORT_NATIVE_PREFILL=1`, plus `--max-num-batched-tokens 5120`, give about 530 tok/s. 5120 is the largest chunk that stays under the grouped path's 1 GiB FP32 output bound. All four knobs are defaults in the image. The batched prefill in section 8 supersedes the grouped path.
- **Sliding window** is working on the 60 SWA layers (window 128), in both the KV cache (only the 10 full-attention layers hold per-token KV) and the Triton DiffKV kernel.
- **All-reduce** is PyNCCL over RoCE. It is about half of each decode pass. `NCCL_PROTO=Simple` changed nothing.

## 8. Expert-batched ARVQ prefill (2026-09-26)

Profile of an 11K-token prefill with the older grouped path (rank 0, 24.4 s of GPU time):
- **Native `hybrid_kernel` on hot routes: 7.1 s.** It runs 4 FP4 activation planes.
- **The grouped cold path: about 9.5 s.** It is a per-expert loop of about 43K dequant launches and 43K matmuls, and the CPU was launch-bound ("Command Buffer Full").
- **NCCL all-reduce: 4.5 s.**

v3 adds `nvfp4_arvq_batched_prefill.py` and `grouped.cu`:

- **Two kernels.** One gate/up and one down GEMM kernel covers all routes, cold (ARVQ 8+8 codebook, decoded in registers from a shared-memory LUT with the hardware FP4 convert) and hot (NVFP4).
- **Arithmetic.** FP16 tensor cores with FP32 accumulation. The down projection adds into an FP32 output with vector atomics. The grid is sliced into 512 columns for L2 reuse.
- **Gate.** `arvq_mlp` takes this path when `VLLM_ARVQ_BATCHED_PREFILL=1`, the chunk has at least `VLLM_ARVQ_BATCHED_MIN_TOKENS` tokens (default 32), and the stream is not capturing a CUDA graph. Decode and MTP steps keep the native kernels.
- **mcbook16 (v5) layers** are rejected by `supported()` and use the old paths. This checkpoint is v4 throughout.

Harness, per MoE layer on one TP4 rank:

| Layer | Tokens | Native | Grouped (old) | Batched (now) |
|---|---:|---:|---:|---:|
| 30 (all cold) | 5120 | 489 ms | 90 ms | 18.2 ms |
| 64 (291 cold, 93 hot) | 5120 | 623 ms | 147 ms | 18.9 ms |

Accuracy of the batched path:
- Cosine vs native is 0.9999996. Relative Frobenius error vs an FP32 reference is 1.709e-3, compared with 1.717e-3 for native, 1.735e-3 for grouped, and a 1.658e-3 bf16 floor.
- Greedy outputs on the cluster drift from the grouped build at tokens 31-53. That matches run-to-run drift within one boot (tokens 53-58), which comes from FP32 atomics and NCCL.

Cluster result: 9.5K-token prefill goes from 505 to 1,291 tok/s, and 38K-token from 527 to 1,038 tok/s. All-reduce is now the largest prefill cost.

## 9. Attention o_proj in FP8 (weights update, 2026-09-27)

**What the profile showed.** A v4 decode step takes about 93 ms, and the GPU is busy about 98% of it, so the step is bound by memory bandwidth plus the TP4 all-reduces. The single largest item was the target's attention `o_proj`: about 17 ms per step, reading roughly 3.5 GB per rank of **BF16** weights.

**Why it was BF16.** Xiaomi's `quantization_config.source_fp8.ignored_layers` lists every `model.layers.N.self_attn.o_proj`, so these layers stayed BF16 while `qkv_proj` is FP8.

**The change.** All 70 decoder `o_proj` weights are quantized to FP8 e4m3 with 128x128 block scales, the same format as `qkv_proj`. The worst relative Frobenius error is 2.7%. The decoder entries are removed from `ignored_layers`; the MTP decoder `o_proj` stays BF16. `backbone-001` shrinks from 30.2 GB to 23.2 GB. Converter: `oproj-fp8/convert_oproj_fp8.py` in the campaign tree.

**Quality gate.** Teacher-forced NLL over 60 held-out on-policy responses (53,575 tokens), on identical token sequences:

| | BF16 o_proj | FP8 o_proj |
|---|---:|---:|
| mean NLL | 0.4443 | 0.4455 (+0.28%, perplexity x1.0012) |
| per class | code 0.243, prose 0.652, reason 0.120 | code 0.244, prose 0.655, reason 0.116 |

All 14 benign requests in the answer check were answered.

**Speed (k=2):**

| | BF16 o_proj | FP8 o_proj |
|---|---:|---:|
| prose | 22.0 tok/s | 24.9 tok/s (+13%) |
| code | 30.4 tok/s | 34.1 tok/s (+12%) |

## 10. NCCL over both PCIe paths of the cabled CX-7 port (2026-09-27)

**Topology.** Each Spark has one CX-7 port cabled. Linux shows that single 200G port as **two RoCE devices**, `rocep1s0f1` and `roceP2p1s0f1`. They are the same port reached through two PCIe functions. The unplugged port's two devices show DOWN.

**The problem.** With `NCCL_IB_HCA=rocep1s0f1` alone, NCCL's tuner modelled about 8 GB/s for ring all-reduce, and prefill all-reduces measured about 8-12 GB/s. That is well under line rate, because one PCIe path is the limit.

**The fix.** `serve/launch-rank.sh` now defaults to `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`. NCCL then alternates channels over both devices (NET/IB/0 and NET/IB/1). No second cable is needed. Both devices had IPv4 RoCE v2 GIDs at the same index on each node, and each has its own subnet: 10.100.10.x and 10.100.11.x.

**Result.**
- Prefill all-reduce time dropped 23%.
- Prefill was 14-25% faster: 9.5K tokens went from 936 to 1,169 tok/s, and 38K from 1,014 to 1,152 tok/s.
- Decode did not change; its all-reduces are small and bound by latency.
- `NCCL_PROTO=Simple` alone changed prefill all-reduce time by only about 10%.

Note that the kernel name `ncclDevKernel_AllReduce_..._RING_LL` does not identify the protocol, because NCCL dispatches every protocol through a few generic kernels.
