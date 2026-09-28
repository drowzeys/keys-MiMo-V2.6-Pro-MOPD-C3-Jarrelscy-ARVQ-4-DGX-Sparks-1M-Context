# MOPD × Jarrelscy ARVQ integration on 4 DGX Sparks (2026-09-28)

Xiaomi released [MiMo-V2.6-Pro-MOPD](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-MOPD) to fix tool-call repetition in
the RL model: the same call emitted many times in one turn. We carried it into Jarrelscy's ARVQ / NVFP4 hybrid.

**Weights (same v4 launcher, GPU util 0.85):**

| | HF |
|---|---|
| **C3 stock** | [drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21](https://huggingface.co/drowzeys/keys-MiMo-V2.6-Pro-MOPD-Jarrelscy-ARVQ-hybrid21) |

`serve/launch-rank.sh` is the full-speed recipe. Candidate trees that symlink into stock 63430f7 also need `BASE_MOUNT`.

## What changed between RL and MOPD

- **Non-expert weights:** 369 of the 575 decoder non-expert tensors changed, each by only ~0.1-0.2% relative L2. The changes are in attention `qkv_proj` / `o_proj`, the norms, `embed_tokens`, `lm_head` and dense layer 0. Router gates and the MTP heads are identical.
- **Routed experts:** changed by ~1.4% relative L2. That is far below the resolution of the 2-bit ARVQ codebooks (~45-60% weight error), so a cold expert cannot carry the update without a new fit. NVFP4 experts carry it exactly.
- **The stock ARVQ tree:** its non-expert tensors are byte-identical to the RL source (575/575), so swapping in MOPD's versions is exact.

## Candidates

All candidates use FP8 attention `o_proj` (SPARK-PORT.md §9) and image v4 with MTP k=2.

| | Build |
|---|---|
| C0 | stock RL ARVQ (control) |
| C1 | C0 + MOPD non-expert weights ([`mopd/build_c1.py`](mopd/build_c1.py)) |
| C2 | C1 + MOPD in the published 5% NVFP4 hot set ([`mopd/reencode_layer.py`](mopd/reencode_layer.py) `--hot-only`) |
| **C3 (published)** | C1 + hot set widened to 21% (5,560 experts), all filled from MOPD ([`mopd/build_stage3.py`](mopd/build_stage3.py)) |

**C3 hot-set selection.** C3 uses Jarrelscy's own `prepare_hybrid.select_hot` (global routing-weighted benefit score, per-layer cap 192) with count 5,560 instead of 1,325.
- It is a strict superset of his set, and changes layers 27-68.
- Each changed layer gets a new roster (all hot experts requantized from MOPD with `hybrid.quantize`) and cold files holding only the rows of experts that stay cold.
- The loader reads the per-layer counts from `config.json`, so no code change is needed.

## Tool-call repetition

[`mopd/toolrep_bench.py`](mopd/toolrep_bench.py) uses Xiaomi's within-turn metric: duplicates after JSON canonicalisation of (name, arguments). It replays 87 real Hermes agent requests 5 times each (435 turns) with server defaults.

| | Turns with a repeated call | Flooding turns (32+ calls) | Calls per tool turn | Largest turn | Hit 8192-token cap |
|---|---:|---:|---:|---:|---:|
| C0 stock RL | 30% | 5.5% | 60 | 354 | 7% |
| C1 | 9.5% | 1.4% | 21 | 451 | 2% |
| C2 | 16% | 2.5% | 35 | 335 | 4% |
| **C3** | **7.4%** | **0.9%** | **7.6** | **152** | **1%** |

- **Abliteration made looping worse:** stock RL 30% vs abliterated 53%.
- **MOPD's non-expert weights do most of the fix (C1).**
- **C2 does not beat C1 reliably.** The difference is within noise at this sample size.
- **C3 is the best on every measure.**

## Quality

[`mopd/qeval.py`](mopd/qeval.py) runs greedy with thinking off, on fixed samples. HumanEval completions are executed against the official tests. The ± figures are 95% margins.

| | C1 | **C3** | C3 − C1 |
|---|---:|---:|---:|
| GSM8K (400) | 93.2% | **96.8%** | +3.5 ± 3.0 |
| HumanEval (164) | 92.1% | **93.9%** | +1.8 ± 5.5 |
| MMLU-Pro (500) | 72.2% | **77.4%** | +5.2 ± 5.4 |

C3's NLL on our RL on-policy held-out text rises from 0.475 to 0.541. That is expected: C3 moves toward MOPD's distribution. The benchmarks above show it is not a quality loss.

## Speed and memory (MTP k=2, 512 tokens, temperature 1.0)

| | Prose | Code | 4 requests | Prefill 9.5K / 38K |
|---|---:|---:|---:|---:|
| C1 | 23.5 | 33.9 | 48.2 | 1,074 / 1,075 |
| **C3 stock (2026-09-28 live)** | **24.2** | **32.2** | **46.3** | 883 / 1074 |

- **C3's cost:** its extra NVFP4 experts add ~11 GiB of weights per rank. Decode drops ~5-8% because a 4-bit expert is more bytes to read than a 2-bit one.
- **KV pool:** measured at 2,116,828 tokens on C0 and **1,282,005 tokens on C3** (weights 72.2 → 83.2 GiB per rank), which still fits the 1M context.

## Limits

- **79% of experts still hold RL weights (2-bit cold set).** A full MOPD cold re-fit needs Jarrelscy's activation calibration and PV pipeline. Our fixed-codebook re-encoder in delta mode transferred only 11-14% of the MOPD update.
- **Paths:** the scripts in `mopd/` are the ones we ran and keep our cluster paths (node IPs, `/home/keyspark/...`, the MOPD download location). Edit them before use.
- **This repo publishes stock C3 only.** Refusal 5/32 · cyber 6/22 thinking off. Grafting dealign `o_proj` onto these weights brought the tool-call loop back, so that tree is not published.
