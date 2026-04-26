<h1 align="center">vllm-awq4-qwen</h1>

<p align="center">
  <strong>Qwen 3.6-27B (AWQ-INT4) + DFlash speculative decoding on AMD Strix Halo (gfx1151).<br>
  Vision · tools · 256K context · OpenAI-compatible · Docker.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/Single--stream-24.8_t%2Fs_peak-red" alt="Speed" />
  <img src="https://img.shields.io/badge/vs_no--spec-+340%25-red" alt="Speedup" />
  <img src="https://img.shields.io/badge/Model-Qwen3.6--27B--AWQ4-0b7285" alt="Model" />
  <img src="https://img.shields.io/badge/Quant-AWQ_INT4_W4A16_g32-purple" alt="Quant" />
  <img src="https://img.shields.io/badge/Context-256K_native-orange" alt="Context" />
  <img src="https://img.shields.io/badge/Vision-yes-blue" alt="Vision" />
  <img src="https://img.shields.io/badge/Tools-yes-blue" alt="Tools" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Ubuntu-26.04-E95420?logo=ubuntu&logoColor=white" alt="Ubuntu" />
  <img src="https://img.shields.io/badge/ROCm-TheRock_7.13_nightly-ED1C24?logo=amd&logoColor=white" alt="ROCm" />
  <img src="https://img.shields.io/badge/GPU-gfx1151_(RDNA_3.5)-ED1C24?logo=amd&logoColor=white" alt="GPU" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/vLLM-0.20.0_(src+patched)-4B2E83" alt="vLLM" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
</p>

---

## ⚡ The numbers

| | Single-stream t/s | Hardware | Cost |
|---|---|---|---|
| 🟩 *DGX Spark FP8 baseline (no spec)* | 7.8 | NVIDIA GB10 Blackwell | ≈ $4 000 |
| 🟩 *Our BF16 sibling repo, no spec* | 4.3 | AMD Strix Halo | ≈ $1 500 |
| 🟩 *Our AWQ4, no spec (this repo, baseline)* | 5.6 | AMD Strix Halo | ≈ $1 500 |
| 🟧 *DGX Spark FP8 + DFlash + MTP (claimed)* | 20-25 | NVIDIA GB10 Blackwell | ≈ $4 000 |
| 🟥 **Our AWQ4 + DFlash N=8 (this repo)** | **24.8 peak / 18.5 mean** | **AMD Strix Halo** | **≈ $1 500** |
| ⚪ *DGX Spark NVFP4 + DFlash, Blackwell-locked* | 83.9 median | NVIDIA GB10 sm_121a only | ≈ $4 000 |

**On any vendor-portable / upstream-vLLM path, AMD Strix Halo iGPU at $1500 matches NVIDIA DGX Spark at $4000 on Qwen 3.6-27B.**

We do *not* beat **Blackwell-specific NVFP4**  -  that uses sm_121a-only kernels not portable to any other GPU (not even other NVIDIA cards). See [vs DGX Spark](#-vs-dgx-spark--honest-comparison) for sources.

> **+340% over no-spec baseline**  -  *5.6 → 24.8 t/s* on `/v1/responses`, single-stream, full 256K context, on a fanless integrated GPU.

---

## ✅ What works

| Feature | Status |
|---|---|
| 🟢 `/v1/chat/completions` | full chat with thinking |
| 🟢 `/v1/responses` | reasoning-parser separates `<think>` from output |
| 🟢 `/v1/completions` | raw text completion |
| 🟢 Tool calling on `/v1/chat/completions` | `qwen3_coder` parser, non-streaming |
| 🟢 Tool calling on `/v1/responses` | same  -  *known bug-prone path, works here* |
| 🟢 Vision (image input) | ViT on `TRITON_ATTN`, LM on `ROCM_ATTN` + DFlash |
| 🟢 256K context | full `--max-model-len 262144`, ~232K usable after vLLM padding |
| 🟢 DFlash speculative decoding | N=1, N=4, N=8 all tested |
| 🟢 SWA in DFlash drafter | PR #40898 cherry-pick handles interleaved sliding-window |
| 🟡 Streaming tool calls | upstream parser bug; use non-streaming |
| 🔴 HIP graphs | freezes on gfx1151, kept off via `--enforce-eager` |
| 🔴 `Qwen/Qwen3.6-27B-FP8` | Triton w8a8 autotune stall on hybrid model |

---

## 📊 Bench (DFlash N=8, 3 runs each, median)

| Endpoint | Test | Prompt tok | Output tok | t/s |
|---|---|---:|---:|---:|
| chat | factual ("speed of light") | 29 | 232 | 21.82 |
| chat | explainer ("entanglement") | 27 | 1 562 | 18.50 |
| **responses** | reasoning ("3 trains") | 57 | 1 216 | **24.80 ⚡** |
| chat + image (1280×720) | scene description | 910 | 1 014 | 13.84 |
| chat + image (1024×1024) | object list | 1 053 | 857 | 13.82 |
| chat + tools | get_weather (Tokyo) | 318 | 142 | 15.53 |
| **responses + tools** | get_weather (Paris) | 336 | 95 | **13.26 ✅** *bug-prone combo works* |
| chat | **Three.js codegen** | 60 | 3 237 | **18.42** *(saved as runnable HTML, [see demo](#-the-threejs-demo))* |
| completions | short factual ("capital of France") | 5 | 8 | 6.34 *(8 tokens, dominated by first-token overhead)* |
| chat | 25K-token long-context synthesis | ~22 000 | up to 2 048 | not captured cleanly  -  *see [Known issues](#%EF%B8%8F-limits--honest-caveats)* |

> Wall-clock client-side. `temperature=0`, max-num-seqs=1, single-stream. The vLLM engine logs report ~25-27 t/s internal (excludes round-trip + initial prefill). Run yourself: `python3 test/bench_full.py`. Raw results: [`test/bench_full_results.json`](test/bench_full_results.json).

### DFlash acceptance scaling (vLLM internal metrics)

The drafter is **genuinely guessing the right tokens**, not just running and getting rejected:

| Spec tokens N | Mean accepted / round | Avg acceptance | Per-stream t/s (steady) |
|---:|---:|---:|---:|
| 0 (no spec) | n/a | n/a | 5.64 |
| 1 | 1.52 | 52% | 8.95 |
| 4 | 3.20 | 64% | 17.92 |
| 8 | 5.64 to 6.35 | 51-67% (varies by content) | 19.80 (chat) / **24.80 (responses)** |

Per-position acceptance falls off as N grows (drafter is less confident predicting 8 tokens ahead than 4). Position-0 stays in the 83-95% range  -  i.e. the very next token is almost always right.

---

## 🥊 vs DGX Spark  -  honest comparison

| | Strix Halo + AWQ4 + DFlash (this repo) | DGX Spark FP8 + DFlash + MTP | DGX Spark NVFP4 + DFlash (Blackwell-only) |
|---|---|---|---|
| Single-stream median | **18.5 t/s** | 20-25 t/s | 83.9 t/s |
| Single-stream peak | **24.8 t/s** | 25 t/s | 127.5 t/s |
| Aggregate at concurrent streams | *not yet tested* | *not published* | 313.6 t/s @ 128 streams |
| Context | 256K | 256K | 256K |
| Vision support | ✅ | ✅ | ✅ |
| Hardware | AMD iGPU, ROCm 7.13 | NVIDIA GB10 Blackwell | NVIDIA GB10 Blackwell sm_121a |
| Stack | Upstream vLLM v0.20.0 + 16 patches | Upstream vLLM | Custom CUDA 13 + FlashInfer 0.6.8 + sm_121a-only |
| Cost | **≈ $1 500** | ≈ $4 000 | ≈ $4 000 |
| Open source toolchain | 100% | mostly | partly (NVFP4 kernels are NVIDIA's) |

**Sources cited above:**
- DGX Spark FP8 baseline + claimed DFlash+MTP: [vLLM issue #40632 (hongxiayang)](https://github.com/vllm-project/vllm/issues/40632), [DFlash blog](https://z-lab.ai/projects/dflash/)
- DGX Spark NVFP4: [AEON-7/Qwen3.6-NVFP4-DFlash README](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash)  -  single-stream median 83.9 t/s, p95 127.5, aggregate 313.6 @ 128 streams. Hardware-locked: *"Image is hardcoded for GB10 only; incompatible with Hopper, Ampere, B200, or desktop Blackwell variants without rebuild."*
- Our numbers: `python3 test/bench_full.py` against this repo's Docker image, deterministic temp=0, 3 runs per endpoint.

---

## 🚀 Quick start

<details>
<summary><b>Click to expand  -  full install in ≈ 30-40 min</b></summary>

### 1. Hardware required
- AMD Ryzen AI Max+ 395 "Strix Halo" or compatible gfx1151 iGPU
- 128 GB UMA (BIOS UMA frame buffer set to **2 GB minimum**, *not* maxed)
- Linux host with `/dev/kfd` + `/dev/dri` exposed to Docker
- ≥ 100 GB free disk for models + caches

### 2. Host setup (one-time)

#### BIOS
Set the dedicated GPU VRAM carve-out to its **minimum (2 GB / 2048 MB)**.
Menu varies: *UMA Frame Buffer Size*, *iGPU Memory*, *GPU Shared Memory*. You want GTT-on-demand, not fixed pre-alloc.

#### GRUB (raise TTM page limit so the GPU can map ~116 GiB of GTT)
```bash
sudo nano /etc/default/grub
# set:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash ttm.pages_limit=30408704 amdgpu.noretry=0 amdgpu.gpu_recovery=1"
sudo update-grub
sudo reboot

# verify after reboot:
cat /sys/class/drm/card1/device/mem_info_gtt_total
# expect ~124554670080 (≈ 116 GiB)
```

### 3. Accept HF gated-model terms (manual, blocking)
The DFlash drafter is gated on Hugging Face  -  you must click "Agree" once:
- ➡️ **<https://huggingface.co/z-lab/Qwen3.6-27B-DFlash>** ← log in, accept conditions

The target model `cyankiwi/Qwen3.6-27B-AWQ-INT4` is ungated.

### 4. Clone + configure
```bash
git clone <this-repo>
cd vllm-awq4-qwen
cp .env.template .env
nano .env
# set:
#   VLLM_HOST_MODELS_DIR=/path/to/your/hf/cache
#   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx     # from https://huggingface.co/settings/tokens
```

### 5. Pre-download both models (in parallel)
```bash
export $(grep -E '^(HF_TOKEN|VLLM_HOST_MODELS_DIR)=' .env | xargs)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download cyankiwi/Qwen3.6-27B-AWQ-INT4 --cache-dir "$VLLM_HOST_MODELS_DIR/hub" &
HF_HUB_ENABLE_HF_TRANSFER=1 hf download z-lab/Qwen3.6-27B-DFlash       --cache-dir "$VLLM_HOST_MODELS_DIR/hub" &
wait
# ≈ 14 GB target + ≈ 3.3 GB drafter
```

### 6. Build the Docker image (≈ 25-35 min, one-time)
```bash
docker compose build
```
Multi-stage: TheRock ROCm 7.13 nightly tarball → PyTorch from `rocm.nightlies.amd.com/v2-staging/gfx1151/` → vLLM v0.20.0 source → 16 idempotent string-replace patches → C/HIP extensions for gfx1151.

### 7. Boot the engine
```bash
docker compose up -d
docker logs -f vllm-awq4-qwen
# wait for: "Application startup complete"
# cold start ≈ 8-10 min (Triton JIT compiles ROCM_ATTN + DFlash kernels)
# warm restart < 30 s (kernel cache persisted to ./.triton-cache)
```

### 8. Talk to it
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.6-27B-AWQ4","messages":[{"role":"user","content":"hi"}]}'
```

### Tunable env vars (.env overrides)
| var | default | meaning |
|---|---|---|
| `VLLM_DFLASH_N` | `8` | speculative tokens; higher = more parallelism, lower acceptance past ~8 |
| `VLLM_GPU_MEMORY_UTIL` | `0.9` | KV cache budget |
| `VLLM_MAX_NUM_SEQS` | `1` | bump to 4-10 for aggregate throughput |
| `VLLM_MAX_MODEL_LEN` | `262144` | full Qwen 256K |
| `VLLM_MODEL_ID` | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | target model |

</details>

---

## 🎮 The Three.js demo

Single-shot a Three.js Minecraft-style voxel game, generated by the model:

> **Prompt:** *"Write a complete single-file HTML using Three.js (CDN) to render a Minecraft-style voxel world: 16×16 procedural terrain (sin/cos), 3 block colors by height, WASD + mouse-look (Pointer Lock), ambient + directional light, sky-blue background. Output ONLY the complete HTML."*
>
> **Output:** 7 591 chars, 3 237 tokens, **175 s @ 18.4 t/s sustained**, no thinking, no retries, no edits  -  first try.
>
> **Verification:** `<!DOCTYPE>` ✓ · `</html>` ✓ · 35 Three.js symbol references · resize handler · CDN load OK in browser. **It runs.**

Open it: `xdg-open test/bench_full_threejs.html` or copy to your desktop and double-click.

---

## 🥗 Cross-quant quality reference

We didn't run quality benchmarks ourselves  -  these are **published numbers** from each model's official source:

| Variant | Source | Headline benchmarks | Quality vs BF16 |
|---|---|---|---|
| Qwen3.6-27B BF16 (base) | [Qwen / HF](https://huggingface.co/Qwen/Qwen3.6-27B) | MMLU-Pro **86.2** • GPQA Diamond **87.8** • LiveCodeBench v6 **83.9** • SWE-bench Verified **77.2** • C-Eval **91.4** | reference |
| Qwen3.6-27B-FP8 (official) | [Qwen / HF](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | "fine-grained FP8, block-128" | *"performance metrics nearly identical to those of the original model"*  -  Qwen team |
| **Qwen3.6-27B-AWQ-INT4 (cyankiwi) ← used here** | [HF model card](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) | compressed-tensors W4A16 g32, MSE observer, asymmetric, **vision blocks excluded from quant** | typical AWQ INT4 g32: < 1% MMLU loss per AWQ paper |
| Qwen3.6-27B-Q8_0 (Unsloth GGUF) | [HF GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) | 28.6 GB | "essentially lossless" per [Unsloth Dynamic 2.0 docs](https://unsloth.ai/docs/basics/unsloth-dynamic-v2.0-gguf) |

**Why we picked AWQ-INT4 g32:**
- ~14 GiB on disk (smallest near-lossless quant for Qwen 3.6-27B)
- Vision blocks kept BF16 (cyankiwi's `quantization_config.ignore` excludes the visual tower) → multimodal stays full-precision
- Auto-detected by vLLM via `quant_method: compressed-tensors` (no manual `--quantization` flag needed)

---

## 🛠️ What we contributed

We don't fork vLLM. We cherry-pick two upstream PRs as idempotent string-replace patches in `scripts/patch_strix.py`:

<details>
<summary><b>Patch 13  -  PR #40176 cherry-pick (ROCm DFlash on gfx1151)</b></summary>

- **Upstream**: [vllm-project/vllm#40176](https://github.com/vllm-project/vllm/pull/40176) by AMD's [@micah-wil](https://github.com/micah-wil), merged 2026-04-22 to main (commit `6d09769700`)
- **Why we patch**: v0.20.0 was tagged 2026-04-23 from a release branch that *did not* back-port this merge.
- **Diff**: 41+ / 13- across 4 files
  - `vllm/v1/attention/backends/rocm_attn.py`  -  `causal: bool` field, `supports_non_causal()=True` (no arch gate)
  - `vllm/v1/attention/ops/prefix_prefill.py`  -  `CAUSAL: tl.constexpr` parameter to `_fwd_kernel`, branched K-range and mask logic
  - `vllm/v1/attention/ops/chunked_prefill_paged_decode.py`  -  `causal=` plumbing
  - `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`  -  explicit `supports_non_causal()=False`
- **Effect**: vLLM allows DFlash to run on `ROCM_ATTN` on gfx1151. Without this, DFlash refuses to load (`supports_non_causal=False`).
- Source diff archived: `.research/vllm-dflash-prs/raw/PR-40176.patch`

</details>

<details>
<summary><b>Patch 14  -  PR #40898 cherry-pick (SWA support in DFlash drafter)</b></summary>

- **Upstream**: [vllm-project/vllm#40898](https://github.com/vllm-project/vllm/pull/40898) by [@jianc99](https://github.com/jianc99) (DFlash paper author + drafter publisher)  -  **OPEN as of 2026-04-26**
- **Why we patch**: drafter author explicitly recommends installing this PR for vanilla vLLM compatibility with the `z-lab/Qwen3.6-27B-DFlash` drafter (4 sliding + 1 full attention layers).
- **Diff**: 156+ / 1- across 4 files
  - `vllm/model_executor/models/qwen3_dflash.py`  -  `_get_dflash_layer_types`, `sliding_window` threading, `sliding_attention_layer_names` set
  - `vllm/transformers_utils/configs/speculators/algos.py`  -  preserves `layer_types`, `sliding_window` etc.
  - `vllm/v1/spec_decode/dflash.py`  -  per-layer `causal=True` metadata for SWA layers
  - **`vllm/v1/worker/gpu_model_runner.py`  -  `target_layer_ids` `+1` shift fix** (this is a **correctness bug**, not just optimization  -  without it the drafter reads the wrong target hidden states and acceptance plummets)
- **Verified live**: engine logs show `Using auxiliary layers from speculative config: (2, 17, 32, 47, 62)`  -  the drafter's `target_layer_ids: [1, 16, 31, 46, 61]` correctly +1-shifted.
- Source diff archived: `.research/vllm-dflash-prs/raw/PR-40898.patch`

</details>

<details>
<summary><b>14 hardware-enablement patches from kyuz0 (verbatim)</b></summary>

`scripts/patch_strix.py` patches 1-12 are kept verbatim from [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) (Donato Capitella, the de-facto Strix Halo + vLLM stack maintainer). They handle:
- `amdsmi` disable + `MagicMock` (Strix Halo APUs don't expose amdsmi in containers)
- Force gfx1151 detection where vLLM falls back to gfx1100
- Disable AITER FP8 linear / fused MoE / RMSNorm on gfx1x (CDNA-only assembly)
- AITER chip_info JIT path fix
- `flash_attn_interface.py` soft-import (resilience to AITER JIT failures)
- Triton MoE kernel cap (allow gfx11xx)
- ROCM-21812 APU VRAM dynamic margin (50% VRAM clamp workaround)
- Misc: `hipCtx*` warnings, `qwen35` GGUF arch alias, Triton `AttrsDescriptor.__repr__`

These are gfx1151-driven, not quant-driven. Same patches the BF16 sibling repo uses.

</details>

---

## ⚠️ Limits & honest caveats

- **Single-stream only configured** (`--max-num-seqs 1`). Open it to 4-10 for aggregate throughput.
- **Drafter is gated** on HuggingFace  -  manual approval required, no ungated mirror.
- **Streaming tool calls have known upstream bugs** (vLLM PRs #40785, #40787 closed unmerged). Use non-streaming.
- **Triton JIT cold-start ≈ 8-10 min.** Persisted to `./.triton-cache`, warm restarts < 30 s.
- **DFlash acceptance drops past N≈8** (drafter less accurate further ahead). N=15 may give marginal further gain; we stopped at N=8 (best steady-state on chat is 19.8 t/s, on `/v1/responses` is 24.8).
- **Numbers in this README are wall-clock client-side.** vLLM internal generation throughput is ≈ 25-27 t/s on these workloads (engine excludes round-trip + initial prefill).
- **HIP graphs freeze on gfx1151**  -  `--enforce-eager` mandatory.
- **Official `Qwen/Qwen3.6-27B-FP8` doesn't init** on RDNA 3.5 (Triton w8a8 autotune stall on the hybrid model's DeltaNet partitions). AWQ-INT4 sidesteps this and is structurally a better fit since RDNA 3.5 has no native FP8 anyway.

### 🚨 vLLM v0.20.0 qwen3 reasoning parser DOES NOT STREAM REASONING on `/v1/chat/completions`

Confirmed bug  -  independent of DFlash, present even without spec decode. With `--reasoning-parser qwen3` enabled and `stream: true`:

- **`/v1/chat/completions`**: parser **buffers** all `<think>...</think>` content server-side, emits **zero** `delta.reasoning_content` chunks during the stream. Client sees a long "ttft" silence (~10-30 s of thinking) and then a burst of the post-thinking answer at the end.
- **`/v1/responses`**: streams correctly  -  `response.reasoning_text.delta` events arrive at t+0.33 s, then `response.output_text.delta` after `</think>`. Same engine, same DFlash, same numbers. Different code path that actually works.

Verified via direct curl probe in this repo (`stream:true`, simple math prompt):

| Endpoint | `reasoning` deltas during stream | First delta arrives |
|---|---|---|
| `/v1/chat/completions` | **0** | t+12.52 s (just the post-think answer) |
| `/v1/responses` | **62** | **t+0.33 s** ✅ |

**Workaround for end-user clients that need streaming reasoning visible:**
- **Use `/v1/responses`**  -  fully working today, recommended.
- Alternatively disable the parser (remove `--reasoning-parser qwen3` from `docker-compose.yml`) and let raw `<think>...</think>` text stream as part of `delta.content`. Tradeoff: `/v1/responses` no longer auto-separates reasoning into structured `output[].type == "reasoning"` items.
- Or accept it: send `stream: false` on chat completions when you need to display reasoning.

The included `.tools/qwen-cli.py` uses `/v1/responses` to dodge the bug.

This is worth filing upstream  -  the qwen3 parser's streaming path appears to never call `extract_reasoning_content_streaming` correctly on the chat-completions code path. Affects every model using `--reasoning-parser qwen3`, not just DFlash workloads.

> **Transparency note**: this is the issue we hit and verified. There are almost certainly **other** vLLM v0.20.0 quirks we did not exercise  -  long-context corner cases, edge sampler params, multimodal + tools combinations, prefix-cache edge cases, etc. We only documented what our bench surfaced. If you run a workload mode we didn't test (different drafter, different attention backend, hybrid grad cudagraph, etc.) and hit something weird, the right move is *not* "the repo is broken" but "vLLM v0.20.0 is early on this code path  -  check upstream issues, file if new". DFlash + reasoning-parser + ROCm-on-RDNA3.5 are all simultaneously in active flux upstream as of 2026-04-26.

### 🚨 DFlash speculative decoding is *early upstream  -  fragile path*

DFlash landed in vLLM main on **2026-03-30** ([PR #36847](https://github.com/vllm-project/vllm/pull/36847)). As of the time of this README, **5+ DFlash bug-fix PRs are still open** and several DFlash-class bug reports are unresolved across multiple GPU vendors (NVIDIA, AMD, both):

| Open issue / PR | Title | Vendor | Why it matters here |
|---|---|---|---|
| [#39928](https://github.com/vllm-project/vllm/issues/39928) | Qwen3.5 DFlash gives strange responses on SM90 | NVIDIA H100 | Confirms DFlash output-quality bugs are not unique to AMD |
| [#40624](https://github.com/vllm-project/vllm/issues/40624) | Gemma4 0% prefix cache hits with hybrid attention + DFlash | All | Hybrid (DeltaNet-style) attention + DFlash interaction is the same class as our Qwen 3.6-27B target |
| [#40382](https://github.com/vllm-project/vllm/issues/40382) | Gemma-4 + DFlash unservable on Ampere  -  non-causal + head_dim=256 has no compatible attention backend | NVIDIA A100 | DFlash needs `supports_non_causal=True` backends; not all backends qualify (this is exactly what our Patch 13 fixes for `ROCM_ATTN`) |
| [#40425](https://github.com/vllm-project/vllm/pull/40425) | Fix quantized DFlash Qwen3 draft support | All | Open PR fixing quantized drafter loading |
| [#40334](https://github.com/vllm-project/vllm/pull/40334) | fix(dflash): dtype mismatch in `combine_hidden_states` | All | Open PR fixing a dtype bug in the auxiliary-hidden-state path |
| [#40727](https://github.com/vllm-project/vllm/pull/40727) | Update dflash aux layer indexing | All | Related to the same `+1` shift bug our Patch 14d (PR #40898) addresses |
| [#40632](https://github.com/vllm-project/vllm/issues/40632) | Support DFlash for Kimi K2.5 and Qwen3.5-27B for AMD | AMD | The umbrella issue for AMD-side DFlash support  -  our work plugs into this |

**Bottom line: DFlash is not "stable upstream" yet on any vendor.** It works for our specific Qwen 3.6-27B + AWQ4 + N=8 path because we've patched around the issues we hit, but you may encounter unfixed-upstream behaviors not seen here, especially with different drafters, different N values, or different attention backends.

If output quality looks off (incoherent, repetitive, garbled), **first try `--speculative-config '{"method":"dflash", ..., "num_speculative_tokens":1}'`**  -  that's the safest setting; it isolates whether the issue is DFlash itself vs the multi-token spec-decode path.

<details>
<summary><b>Stuck DFlash worker on client disconnect  -  what we hit during the long-context test (recovery documented)</b></summary>

**Symptom (we hit it once during ~22K-token long-context decode):** if the client TCP-disconnects mid-decode while `--speculative-config method=dflash` is active, the EngineCore worker may not gracefully unwind. The API server (`/health`, `/v1/models`) stays responsive but new `/v1/chat/completions` requests time out forever. EngineCore burns 79-200% CPU in a tight loop. Kernel logs:

```
workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us 5 times,
consider switching to WQ_UNBOUND
```

**Likely cause:** a refcount / cancel race between the DFlash proposer thread and the KFD GPU buffer release path when a request is cancelled mid-decode. The DFlash worker keeps running while the client cancellation tries (and fails) to reclaim its buffers. Plausibly related to the open issues table above  -  DFlash's cancellation/cleanup path is not yet hardened upstream.

**Recovery:** `docker compose restart vllm-awq4-qwen` (≈ 9 min cold boot). Run `./scripts/dump_logs.sh stuck-state` *before* the restart to preserve diagnostics for an upstream report.

**Mitigation while waiting for upstream fix:**
- Prefer client-side timeouts larger than your expected decode time
- Avoid `SIGKILL`'ing the client mid-stream  -  use `SIGINT` or wait for graceful completion
- For very long-context requests (>10K prompt tokens), consider testing with `num_speculative_tokens=1` first to confirm baseline correctness, then ramp up

If you reproduce this, file under the [DFlash umbrella tracker (#40632)](https://github.com/vllm-project/vllm/issues/40632) with output of `./scripts/dump_logs.sh`.

</details>

---

## 🥨 Sibling repos (same hardware, different quants)

| | this repo | [`vllm-qwen`](https://github.com/hec-ovi/vllm-qwen) | [`llama-qwen`](https://github.com/hec-ovi/llama-qwen) |
|---|---|---|---|
| Format | AWQ-INT4 (compressed-tensors) | BF16 safetensors | Q8_0 GGUF |
| Memory at idle | ≈ 36 GiB | ≈ 105 GiB | ≈ 35 GiB |
| **Decode (no spec)** | **5.6 t/s** | 4.3 t/s | 7.5 t/s |
| **Decode (DFlash N=8)** | **19.8 t/s chat / 24.8 responses** | n/a | n/a |
| Vision | ✅ | ✅ | ❌ no `mmproj` in GGUF |
| `/v1/responses` reasoning | ✅ | ✅ | ❌ |
| Tool calls | ✅ non-stream | ⚠ parser bugs | ✅ via `--jinja` |
| 256K context | ✅ | ✅ | ✅ |

**Rule of thumb:** vision + reasoning + smallest near-lossless footprint → **this repo**. Pure text + speed without vision → `llama-qwen`. Official Qwen team weights → `vllm-qwen`.

---

## 📁 Repo layout

```
.
├── Dockerfile               # multi-stage: Ubuntu + TheRock + torch + vLLM v0.20.0 from source
├── docker-compose.yml       # one service, restart=no, --enforce-eager, DFlash N=8
├── .env.template            # the one config file you edit
├── scripts/
│   ├── install_rocm_sdk.sh  # TheRock S3 nightly tarball → /opt/rocm
│   ├── patch_strix.py       # 16 idempotent string-replace patches (1089 LOC)
│   └── dump_logs.sh         # snapshot engine + kernel logs before any down/restart
├── test/
│   ├── bench.py             # original 5-endpoint harness (peso prompt)
│   ├── bench_full.py        # generic 5-endpoint + tools (chat+responses) + 2 images + Three.js
│   └── bench_longctx.py     # 25K-token synthesis test using real .research data
├── LICENSE                  # Unlicense (public domain)
└── README.md
```

13 user-written files. No vendored binaries, no untracked tarballs.

---

## 🔬 Run the benches

```bash
# 5-endpoint sweep + 2 images + tools (chat + responses) + Three.js codegen
python3 test/bench_full.py
# -> test/bench_full_results.json  +  test/bench_full_threejs.html

# Long-context (~25K tokens of real .research/ data, hard synthesis question)
python3 -u test/bench_longctx.py    # use -u (unbuffered) to avoid pipe-buffering surprises
# -> test/bench_longctx_result.json

# Original 5-endpoint harness (peso prompt, kept for back-compat)
python3 test/bench.py
# -> test/bench_results.json
```

Each script is self-contained and prints to stdout. Engine must already be running.

---

## 🙏 Credits

- [@micah-wil](https://github.com/micah-wil) (AMD)  -  PR #40176, the actual fix that lights up DFlash on RDNA 3.5
- [@jianc99](https://github.com/jianc99) (z-lab)  -  DFlash paper, drafter model, PR #40898
- [@kyuz0](https://github.com/kyuz0) (Donato Capitella, Reversec)  -  the Strix Halo + vLLM patch bundle that hardware-enables gfx1151
- [@hongxiayang](https://github.com/hongxiayang) (AMD)  -  MI3xx DFlash verification, vLLM issue #40632 stewardship
- [@cyankiwi](https://huggingface.co/cyankiwi)  -  the AWQ-INT4 quant we serve as target

---

## 📜 License

[The Unlicense](LICENSE)  -  public domain. Use, modify, distribute, sell, fork. No attribution required, no warranty given.
