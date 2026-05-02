<h1 align="center">vllm-awq4-qwen</h1>

<p align="center">
  <strong>Qwen 3.6-27B (AWQ-INT4) + DFlash speculative decoding on AMD Strix Halo (gfx1151).<br>
  Vision · tools · 256K context · OpenAI-compatible · Docker.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Working-brightgreen" alt="Status" />
  <img src="https://img.shields.io/badge/Single--stream-24.8_t%2Fs_peak-red" alt="Speed" />
  <img src="https://img.shields.io/badge/3--stream_aggregate-41_t%2Fs_peak-red" alt="3-stream aggregate" />
  <img src="https://img.shields.io/badge/Prefill-105--134_t%2Fs_(0--32K,_HIP_kernel)-brightgreen" alt="Prefill" />
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

| | Single-stream t/s | Hardware |
|---|---|---|
| 🟩 *DGX Spark FP8 baseline (no spec)* | 7.8 | NVIDIA GB10 Blackwell |
| 🟩 *Our BF16 sibling repo, no spec* | 4.3 | AMD Strix Halo |
| 🟩 *Our AWQ4, no spec (this repo, baseline)* | 5.6 | AMD Strix Halo |
| 🟧 *DGX Spark FP8 + DFlash + MTP (claimed)* | 20-25 | NVIDIA GB10 Blackwell |
| 🟥 **Our AWQ4 + DFlash N=8 (this repo)** | **24.8 peak / 18.5 mean** | **AMD Strix Halo** |
| ⚪ *DGX Spark NVFP4 + DFlash, Blackwell-locked* | 83.9 median | NVIDIA GB10 sm_121a only |

> **+340% over no-spec baseline** - *5.6 → 24.8 t/s* on `/v1/responses`, single-stream, full 256K context, on a fanless integrated GPU.

### 📈 Prefill (input processing) speed

Prefill is now **flat at 105 to 134 t/s from 0 to 32K context** thanks to a custom HIP kernel that ships with this repo (`csrc/awq_mmq_gfx1151/`, registered into vLLM by [Patch 16](#-what-we-contributed)). Before the kernel, prefill was a steep cliff curve on TritonW4A16 (132 t/s @ 1k -> 77 t/s @ 2k -> 38 t/s @ 4k). The kernel is a **3.4x improvement at the 4k pain point** and roughly **2.8x at 32k**, while leaving decode bit-for-bit identical (the kernel only takes over for M >= 32; decode shapes still route through TritonW4A16 via the M-dispatch in `vllm_kernel.py`).

**Full bench matrix** (measured 2026-05-02, production config: `util=0.55, max_model_len=65536, max_num_seqs=1`; streaming `/v1/chat/completions`, `temperature=0`, `max_tokens=2048`, 2 runs per cell, mean t/s reported. Min/max within 1% of mean: extremely deterministic.):

| ctx     | case        | prefill t/s | decode t/s |
|---------|-------------|------------:|-----------:|
| 0       | normal      | 98.6        | 12.40      |
| 0       | thinking    | 118.5       | 23.80      |
| 0       | no_thinking | 118.7       | 22.25      |
| 2048    | normal      | 133.0       | 11.16      |
| 2048    | thinking    | 133.8       | 18.99      |
| 2048    | no_thinking | 134.5       | 20.30      |
| 4096    | normal      | 131.2       | 9.85       |
| 4096    | thinking    | 131.3       | 17.49      |
| 4096    | no_thinking | 131.4       | 18.45      |
| 8192    | normal      | 126.8       | 8.39       |
| 8192    | thinking    | 127.5       | 13.66      |
| 8192    | no_thinking | 127.1       | 13.58      |
| 16384   | normal      | 118.9       | 5.90       |
| 16384   | thinking    | 119.6       | 9.15       |
| 16384   | no_thinking | 120.4       | 9.49       |
| 32768   | normal      | 105.8       | 3.42       |
| 32768   | thinking    | 106.6       | 4.59       |
| 32768   | no_thinking | 106.3       | 4.61       |

(`two_tools` cases omitted: model exits ~77 tokens after issuing two tool calls, so decode rate is a small-N artifact, not signal. Prefill numbers for `two_tools` track the other cases within 2%.)

**What the kernel did fix.** Stock TritonW4A16 had two specific failure modes on gfx1151:
- Tile shapes hard-coded for MI300 (304 CUs, wave64). gfx1151 is 40 CUs / wave32: bad occupancy at large M.
- Dequant-to-fp16 strategy throws away INT8 WMMA throughput. Our kernel keeps int8 throughout and uses `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (AMD WMMA v1, INT8), which is 2x peak per the AMD ISA docs.

**What the kernel did NOT fix: long-context decode.** Past ~8k context, decode drops to single digits. That is **attention scaling on the KV cache**, not GEMM. The HIP kernel does not touch attention. Fixing it would mean a flash-attention-style KV-aware kernel, which is separate scope.

Background and the original cliff curve live in [`.research/mmq-q4-gfx1151-port/FINDINGS.md`](.research/mmq-q4-gfx1151-port/FINDINGS.md) (the kernel port itself), [`.research/prefill-perf-experiments/CHANGELOG.md`](.research/prefill-perf-experiments/CHANGELOG.md), and [`.research/int4-wmma-rdna-path/FINDINGS.md`](.research/int4-wmma-rdna-path/FINDINGS.md).

---

## ✅ What works

| Feature | Status |
|---|---|
| 🟢 `/v1/chat/completions` | full chat with thinking |
| 🟢 `/v1/responses` | reasoning-parser separates `<think>` from output |
| 🟢 `/v1/completions` | raw text completion |
| 🟢 Vision (image input) | ViT on `TRITON_ATTN`, LM on `ROCM_ATTN` + DFlash |
| 🟢 256K context | full `--max-model-len 262144`, ~232K usable after vLLM padding |
| 🟢 DFlash speculative decoding | N=1, N=4, N=8 all tested |
| 🟢 SWA in DFlash drafter | PR #40898 cherry-pick handles interleaved sliding-window |
| 🔴 HIP graphs | freezes on gfx1151, kept off via `--enforce-eager` |
| 🔴 `Qwen/Qwen3.6-27B-FP8` | Triton w8a8 autotune stall on hybrid model |

### 🛠️ Tool calling support matrix

The streaming-vs-non-streaming behavior is the part most people get wrong. To be unambiguous:

| Endpoint | `stream: false` (wait for full JSON) | `stream: true` (SSE deltas) |
|---|---|---|
| `/v1/chat/completions` + tools | 🟢 **works** - `qwen3_coder` parser produces clean `tool_calls` array. Verified 9 / 9 across single + parallel + multi-tool + round-trip. | 🟡 **buggy** - upstream parser bug (vLLM PRs #40783, #40785, #40787 unmerged). Avoid. |
| `/v1/responses` + tools | 🔴 **broken when combined with `enable_thinking=false`** - tool-call XML lands raw in the `reasoning` field, no `function_call` item parsed. Architectural fix not in this repo. | 🟢 **works** - `function_call` items emitted as proper `response.output_item.added/done` events. Verified across tiny + 2K context, with reasoning on or off. **This is the recommended path for agents.** |

**TL;DR for client builders**: agents that need tool calls should use either `/v1/chat/completions` with `stream: false`, OR `/v1/responses` with `stream: true`. Avoid the other two diagonals.

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
| **responses + tools** | get_weather (Paris) | 336 | 95 | **13.26 ✅** *non-stream, thinking on; streaming variant verified separately in T2/T4 of [`verify_responses_streaming.py`](test/verify_responses_streaming.py)* |
| chat | **Three.js codegen** | 60 | 3 237 | **18.42** *(saved as runnable HTML, [see demo](#-the-threejs-demo))* |
| completions | short factual ("capital of France") | 5 | 8 | 6.34 *(8 tokens, dominated by first-token overhead)* |
| chat | 25K-token long-context synthesis | ~22 000 | up to 2 048 | not captured cleanly  -  *see [Honest limitations](#%EF%B8%8F-honest-limitations)* |

> Wall-clock client-side. `temperature=0`, max-num-seqs=1, single-stream. The vLLM engine logs report ~25-27 t/s internal (excludes round-trip + initial prefill). Run yourself: `python3 test/bench_full.py`. Raw results: [`test/bench_full_results.json`](test/bench_full_results.json).

---

### ⏱️ Spin-up time is ~9 min on every restart (even with all caches warm)

This is the planning number for any `docker compose up`. We measured 8m27s, 8m44s, 8m37s across reboots in this session - the magic number is **~9 min**, regardless of whether the host page cache, Triton kernel cache, and Linux disk cache are warm. The breakdown matches every boot we ran:

```
~95 s   Model load                  target weights (14 GB AWQ4) + DFlash drafter (3.3 GB BF16)
                                    from disk; faster on warm page cache, slower on first
                                    boot after host reboot.
~6-7 m  profile_run + autotune      vLLM runs synthetic max-batch forward passes to size
                                    the KV cache pool, JITs Triton kernels, and wires up
                                    the DFlash speculative pipeline + SWA causal metadata.
                                    This is the dominant cost and it runs every time.
~5 s    Server startup              FastAPI + Uvicorn + /health green.
                                    ----------
~9 min total
```

**What is and isn't cached across restarts:**

| Component | Persisted to host? | Saves on restart? |
|---|---|---|
| Triton compiled kernels | ✅ via `./.triton-cache/` mount | partial (~30 s saved on second boot at same config) |
| Triton autotune choices | ⚠ same dir, recompute is fast | minor |
| MIOpen solver database | ❌ not host-mounted | nothing (re-searches every boot; `MIOPEN_FIND_MODE=FAST` mitigates) |
| DFlash drafter wiring | ❌ rebuilt every boot | nothing |
| `profile_run` forward passes | ❌ must run every boot | nothing |
| Model weights → page cache | ✅ Linux page cache | huge if you don't `drop_caches` between |

**Even a "warm" restart costs ~8.5 min** (we measured exactly this). The Triton cache helps modestly; the dominant ~7 min is the engine doing real GPU work to validate the configuration is OK to serve. There is no shortcut that doesn't carry OOM risk - `--num-gpu-blocks-override` would skip the memory probe (~1-2 min) but pins the KV block count to a stale config the moment any related env var changes, leading to silent OOM at runtime.

**Practical implications:**
- Restart is roughly the cost of a coffee. Plan it.
- **`.env` changes require `docker compose down && up -d`**, NOT `docker compose restart`. `restart` reuses the running container and never re-reads the env file. We hit this bumping `VLLM_MAX_NUM_SEQS` 1→3 and the engine kept reporting `max_num_seqs=1` until a full down + up cycle. Image rebuilds (`docker compose build`) are also `down + up`, not `restart`.
- Treat the engine as a long-lived daily-driver service. The "9-min restart" is paying for safety (profile_run validates the entire pipeline runs without crashing), not waste.

---

### 🧩 Recommended setup for multi-tenant / RAG / agent serving

**Daily-driver profile we ship as the .env defaults** (single-stream, HIP kernel on, comfortable memory budget):

| Setting | Value | Rationale |
|---|---|---|
| `VLLM_MAX_NUM_SEQS` | `1` | single-stream is fast enough now that the prefill kernel landed; flat 105 to 134 t/s prefill from 0 to 32K context |
| `VLLM_MAX_MODEL_LEN` | `65536` | 64K context. The HIP kernel's dual-storage cost (~22 GiB extra weight memory, see below) makes 128K tight at the 0.55 util cap; 64K is the comfortable production point |
| `VLLM_GPU_MEMORY_UTIL` | `0.55` | sets a ~70 GiB cap on what vLLM may claim. Tuned to actual need (model weights + dual-storage + 64K KV pool), not maxed out. Leaves ~60 GiB of UMA free for sibling services. No swap pressure. |

We **dropped `max_model_len` from `262144` to `65536` and dialed `gpu_memory_utilization` to `0.55`** specifically to fit the kernel's dual-storage cost without leaving the box swap-bound. Multi-stream still works (verified up to 3 concurrent in the [stress test below](#-multi-stream--tool-calling-stress-test-3-concurrent)), just bump `VLLM_MAX_NUM_SEQS` and either keep 64K or drop to 32K per slot.

**What this fits alongside vLLM on the same Strix Halo box** (verified running concurrently):
- this vLLM instance serving Qwen 3.6-27B
- a RAG api stack (FastAPI + pgvector + 2× HuggingFace text-embeddings-inference + memgraph) - **CPU-only**
- a Piper TTS-backed web UI ([gladosproject](https://github.com/hec-ovi/gladosproject)) - **CPU-only**

All three coexist without OOM or contention because only vLLM uses the iGPU; the embedding/reranker models and Piper TTS run on CPU and don't compete for the UMA pool.

**Important: only the streaming path on `/v1/responses` is patched.** The local Patch 15 fix in this repo wires `chat_template_kwargs.enable_thinking=false` through to the chat-template renderer on the streaming code path of `/v1/responses` only. The non-streaming path (`stream: false`) on `/v1/responses` still has a separate, deeper routing bug (output text and tool-call XML land in the `reasoning` field instead of `output_text` / `function_call` items). That bug requires a different, larger architectural patch which is **out of scope for this repo**. Build agents and clients against `/v1/responses` with `stream: true` and the engine behaves correctly; do not use `stream: false` on `/v1/responses` if you have set `enable_thinking=false`. See [Verified streaming behavior with Patch 15](#-verified-streaming-behavior-with-patch-15) for the test matrix.

---

### 🤝 Multi-stream + tool calling stress test (3 concurrent)

We tested with `max_num_seqs=3` to verify the engine handles concurrent multi-agent workloads, since this is the realistic scenario when serving a RAG api, a chat UI, and an automation client off the same vLLM box. Results from this session:

| Round | Test shape | Tool calls per response | Result |
|---|---|---|---|
| A | 3 parallel requests, each asks `get_weather` for 3 different cities | **3 parallel calls** in one response | 9 / 9 succeeded; cities + args correct |
| B | 3 parallel requests, each combines `calculate` + `search_web` in one ask | **2 different tools** in one response | 9 / 9 succeeded; both tools called with valid JSON args |
| C | 3 parallel requests, each combines complex multi-param `book_flight` (5 fields incl. enum + ISO date) + `get_weather` | **2 tools, 5+ args each** in one response | 9 / 9 succeeded; all required + optional params populated |
| D | Full round-trip: send tool calls back as `function_call_output` items, get final synthesized answer | n/a (assembling tool results) | clean natural-language answer using both tool results, even called out a count discrepancy ("only 2 results despite requesting 5") |

**Findings**:
- `parallel_tool_calls: true` works on `/v1/responses`. Same-tool fan-out (Round A) and different-tool combo (Round B/C) both verified.
- Three concurrent streams hit `Running: 3 reqs, Waiting: 0` in the scheduler. Per-stream decode held at ~13.5 t/s (vs ~18 t/s solo). Aggregate peaked at **41 t/s**.
- **Zero engine errors observed** across all 9 + 1 round-trip requests. `vllm:request_success_total{finished_reason="error"} = 0`.
- Tool call args validated: integer fields are integers, enums are valid values, ISO dates parsed correctly. The `qwen3_coder` tool-call parser handles non-streaming round-trips cleanly.
- `chat_template_kwargs.enable_thinking=false` was set on every request but the model still produced reasoning. **This was the gap that motivated local Patch 15** (see [What we contributed](#%EF%B8%8F-what-we-contributed)). After the patch, the kwarg routes through correctly and reasoning is genuinely skipped on `/v1/responses` streaming.

This is enough proof that the same vLLM instance can serve **a multi-agent / multi-client setup** (e.g. one process orchestrating an agentic graph of 3 simultaneous reasoning workers, each calling tools) without queueing or correctness issues. Use the **3-agent multi-stream profile** in [Tunable env vars](#tunable-env-vars-env-overrides) above.

### Reasoning intensity limitation (Qwen3.6 specific)

Qwen3.6 tends to **over-reason** even on trivial tool-routing prompts. We saw same-class prompts produce reasoning anywhere from ~400 chars (terse) to **2,075 chars** (verbose) - that variance is the dominant source of wall-time noise across our concurrent tests, not decode speed. TTFT histograms therefore look bimodal (some requests start streaming visible content in 5 s, others take 60+ s while the model thinks). If your downstream UI surfaces TTFT as a single number, expect it to be noisy on this model. **If you don't want reasoning at all**, send `chat_template_kwargs: {"enable_thinking": false}` on `/v1/responses` (streaming) or `/v1/chat/completions`  -  Patch 15 in this repo wires the kwarg through the responses path that vLLM upstream silently dropped. See [Verified streaming behavior with Patch 15](#-verified-streaming-behavior-with-patch-15).

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
| Aggregate at 3 concurrent streams | **27-41 t/s peak** (~13.5 t/s/stream) | *not published* | n/a |
| Context | 256K | 256K | 256K |
| Vision support | ✅ | ✅ | ✅ |
| Hardware | AMD iGPU, ROCm 7.13 | NVIDIA GB10 Blackwell | NVIDIA GB10 Blackwell sm_121a |
| Stack | Upstream vLLM v0.20.0 + 18 patches (incl. local HIP MMQ kernel) | Upstream vLLM | Custom CUDA 13 + FlashInfer 0.6.8 + sm_121a-only |
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
Multi-stage: TheRock ROCm 7.13 nightly tarball → PyTorch from `rocm.nightlies.amd.com/v2-staging/gfx1151/` → vLLM v0.20.0 source → 18 idempotent string-replace patches (incl. Patch 16 that registers the local AWQ-INT4 MMQ HIP kernel) → C/HIP extensions for gfx1151.

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
| var | shipping default | meaning |
|---|---|---|
| `VLLM_DFLASH_N` | `8` | speculative tokens; higher = more parallelism, lower acceptance past ~8 |
| `VLLM_GPU_MEMORY_UTIL` | `0.55` | hard cap on UMA the engine may claim. Sized to actual need, not maxed out. At 0.55 vLLM reserves ~64 GiB and leaves ~60 GiB system free, no swap pressure. **Right-size to actual concurrency x context.** At low `MAX_NUM_SEQS` and high util vLLM allocates an oversized KV pool that sits unused. See [Recommended profiles](#recommended-profiles). |
| `VLLM_MAX_NUM_SEQS` | `1` | concurrent decode slots. `1` is the daily-driver default since the HIP kernel landed (single-stream prefill is now fast enough that batching is no longer how you hide cost). Multi-stream still works (verified at 3, see profiles). |
| `VLLM_MAX_MODEL_LEN` | `65536` | 64K context. Chosen because the kernel's dual-storage cost (~22 GiB extra weight memory for the M-dispatch fallback path) makes 128K tight at the 0.55 util cap. 64K is a comfortable production point and covers almost any agent / RAG retrieval window. Raise to `131072` or `262144` if you accept the tighter memory budget. |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | scheduler token budget per iteration (vLLM v0.20.0 online-serving default). Bump to `16384` to let long prefills land in fewer chunks. Tradeoff: more KV pressure when batching many concurrent decodes. |
| `VLLM_MODEL_ID` | `cyankiwi/Qwen3.6-27B-AWQ-INT4` | target model |

### Recommended profiles

| Profile | `MAX_NUM_SEQS` | `MAX_MODEL_LEN` | `GPU_MEMORY_UTIL` | Budget cap (util x UMA) | Use case |
|---|---|---|---|---|---|
| **Single-stream + HIP kernel** ⭐ (shipping default) | `1` | `65536` | `0.55` | ~70 GiB cap (measured ~64 GiB, ~60 GiB free) | one client, full benefit of the prefill kernel, dual-storage weight cost fits comfortably; daily driver |
| Single-user, max context | `1` | `262144` | `0.9` | ~115 GiB cap | one chat at a time, full 256K, accepts the tighter memory budget |
| 3-agent multi-stream | `3` | `131072` | `0.7` | ~90 GiB cap | 3 simultaneous clients with KV headroom; HIP kernel still kicks in on prefill at M >= 32 |
| 3-up, share the box | `3` | `65536` | `0.55` | ~70 GiB cap | 3 clients, 64K each, leaves room on the box for a RAG api / embedding service / TTS / etc. |

> The "Budget cap" column is `gpu_memory_utilization x UMA pool size (128 GiB)`. It's a *ceiling*, not a target. Actual claimed memory is whatever vLLM's `profile_run` plus model weights plus KV cache pool actually need, which is usually well below the cap. **However**: at high util with low `MAX_NUM_SEQS`, vLLM still allocates a KV pool sized to fill that cap, and most of it goes unused (single-stream cannot consume an 80+ GiB KV pool). Right-size util to your concurrency x context, do not pick `0.9` reflexively. **The HIP MMQ kernel adds a one-time dual-storage cost**: ~22 GiB of extra weight memory holds the transposed TritonW4A16-format weights so decode (M < 32) can fall through to Triton without re-packing per call. This is why the shipping default drops `MAX_MODEL_LEN` from 256K to 64K: the dual-storage budget plus 64K KV per slot fits cleanly under a 0.55 util cap.

**Daily-driver profile, what we ship today:** vLLM at `max_num_seqs=1, max_model_len=65536, gpu_memory_utilization=0.55`. The kernel registration succeeds at startup (`Patch 16: RocmMmqQ4LinearKernel registered at _POSSIBLE_KERNELS[ROCM][0]` in the engine logs). Verified coexisting with three CPU-only sibling services: a RAG api stack (FastAPI + pgvector + 2x HuggingFace text-embeddings-inference + memgraph) and a Piper TTS-backed web UI ([gladosproject](https://github.com/hec-ovi/gladosproject)). All three coexist without OOM or contention because:

- Embedding/reranker models run **CPU-only** (HF TEI `cpu-1.9` image)
- Piper TTS runs **CPU-only** (`PiperVoice.load(use_cuda=False)` is the default; we don't override) - confirmed by inspecting `piper.voice` source
- Only vLLM uses the iGPU, so the CPU-bound services don't compete for the 128 GiB UMA pool

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

We don't fork vLLM. We cherry-pick two upstream PRs and add one local fix, all as idempotent string-replace patches in `scripts/patch_strix.py`:

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
<summary><b>Patch 15  -  local fix: thread <code>chat_template_kwargs</code> through <code>/v1/responses</code></b></summary>

- **Status**: not yet filed upstream  -  worth a vLLM PR; the gap looks accidental.
- **Why we patch**: vLLM's `ResponsesRequest` (in `vllm/entrypoints/openai/responses/protocol.py`) had no `chat_template_kwargs` field at all, and `to_chat_params()` hard-coded `merge_kwargs({}, dict(...))`, ignoring user input. Effect on Qwen3.6: clients that send `chat_template_kwargs: {"enable_thinking": false}` to `/v1/responses` got reasoning anyway, while the same kwarg worked on `/v1/chat/completions` (different code path).
- **Diff**: 6+ / 1- across 1 file - `vllm/entrypoints/openai/responses/protocol.py`. Two anchors:
  - **15a**: adds `chat_template_kwargs: dict[str, Any] | None = None` field to `ResponsesRequest`, sandwiched between `user` and `skip_special_tokens`.
  - **15b**: in `to_chat_params()`, replaces `merge_kwargs({}, dict(...))` with `merge_kwargs(self.chat_template_kwargs or {}, dict(...))`. User-supplied kwargs flow through; vLLM's hardcoded keys (`add_generation_prompt`, `continue_final_message`, `reasoning_effort`) keep precedence.
- **What it does NOT fix**: the routing of model output channels in *non-streaming* `/v1/responses` when `enable_thinking=false`. That requires porting `prompt_is_reasoning_end` from `chat_completion/serving.py` to `responses/serving.py` (separate, larger architectural patch). For this repo we only support and verify the **streaming** path  -  see [Verified streaming behavior](#-verified-streaming-behavior-with-patch-15) below.
- **Verified live**: `python3 test/verify_responses_streaming.py`. T1-T4 (think_off variants) all show 0 chars in the `reasoning_text` channel, content correctly routed to `output_text` or parsed `function_call` items. T5 control (think_on) confirms reasoning still routes to `reasoning_text` when expected.

</details>

<details>
<summary><b>Patch 16  -  local: register the AWQ-INT4 MMQ HIP custom op (Strix Halo prefill kernel)</b></summary>

- **Status**: ships in this repo. The kernel itself lives at [`csrc/awq_mmq_gfx1151/`](csrc/awq_mmq_gfx1151/) and the `.so` is built inside the container against TheRock 7.13.0a + PyTorch 2.10. The patch is a six-line registration block appended to vLLM's mixed-precision linear-kernel dispatcher.
- **Why we patch**: vLLM's stock `TritonW4A16LinearKernel` had two specific failure modes on gfx1151:
  - Tile shapes hard-coded for MI300 (304 CUs, wave64). gfx1151 is 40 CUs / wave32: bad occupancy at large M, which is the prefill regime.
  - Dequant-to-fp16 strategy throws away INT8 WMMA throughput. AMD WMMA v1's `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` is 2x peak vs the fp16 path on this silicon.
- **Diff**: 17-line registration block appended to `vllm/model_executor/kernels/linear/__init__.py`. On engine startup it adds `/root/csrc/awq_mmq_gfx1151` to `sys.path`, imports `RocmMmqQ4LinearKernel` from `awq_mmq_gfx1151.vllm_kernel`, and inserts it at position 0 of `_POSSIBLE_KERNELS[PlatformEnum.ROCM]` so `choose_mp_linear_kernel` picks it ahead of `TritonW4A16LinearKernel` for the W4A16 g32 path. If the import fails (e.g. `.so` not built yet), the kernel list is left untouched and TritonW4A16 keeps its slot.
- **What the kernel does**: ports llama.cpp's MMQ Q4 pattern verbatim. Weights stay int4 in registers, unpacked to int8 in LDS tiles, accumulated through chained WMMA iu8 calls into an i32 tile, with per-group (group_size=32) fp32 dequantization at K-group boundaries. Tile shapes `(mmq_x=48, mmq_y=64, nwarps=4)` are taken from the lhl/pedapudi gist for RDNA3_5 (master defaults of 128/128/8 spill VGPRs on gfx1151's 1536-VGPR-per-SIMD budget).
- **Decode path is preserved bit-for-bit**: `apply_weights` in `vllm_kernel.py` dispatches on M. For M >= 32 (prefill) we use the HIP kernel; for M < 32 (decode, including DFlash N=8 spec rounds at M=8 typical) we fall through to `triton_w4a16_gemm` from vLLM's existing TritonW4A16, with weights pre-transposed at `process_weights_after_loading` time so there is no per-call repack. **Cost**: ~22 GiB of dual-storage weight memory. **Benefit**: the daily driver's 12 to 24 t/s decode floor is untouched.
- **Verified live**: engine logs show `Patch 16: RocmMmqQ4LinearKernel registered at _POSSIBLE_KERNELS[ROCM][0]`. Bench matrix in [Prefill speed](#-prefill-input-processing-speed) confirms the flat 105 to 134 t/s curve; standalone correctness vs the scalar reference in [`csrc/awq_mmq_gfx1151/test_correctness.py`](csrc/awq_mmq_gfx1151/test_correctness.py).
- Background and full kernel structure: [`.research/mmq-q4-gfx1151-port/FINDINGS.md`](.research/mmq-q4-gfx1151-port/FINDINGS.md).

</details>

<details>
<summary><b>12 hardware-enablement patches from kyuz0 (verbatim) + 5 local additions</b></summary>

`scripts/patch_strix.py` patches 1 through 10 (numbered 1, 1.5, 2, 3, 3.5, 5, 6, 7 [twice], 8, 9, 10, totaling 12 sub-patches) are kept verbatim from [kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) (Donato Capitella, the de-facto Strix Halo + vLLM stack maintainer). Patches 11 and 12 are small local additions on top of that base (hipCtx deprecation silence and a `qwen35` GGUF-arch alias). Patches 13, 14, 15 are local cherry-picks of upstream PRs and a small local fix (PR #40176 non-causal attn for DFlash, PR #40898 SWA support, and a `chat_template_kwargs` plumbing fix on `/v1/responses`). Patch 16 is the local registration of the AWQ-INT4 MMQ HIP custom op into vLLM's mixed-precision dispatcher (see the dedicated panel above). They handle:
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


## 🥨 Sibling repos (same hardware, different quants)

| | this repo | [`vllm-qwen`](https://github.com/hec-ovi/vllm-qwen) | [`llama-qwen`](https://github.com/hec-ovi/llama-qwen) |
|---|---|---|---|
| Format | AWQ-INT4 (compressed-tensors) | BF16 safetensors | Q8_0 GGUF |
| Memory at idle | ≈ 36 GiB | ≈ 105 GiB | ≈ 35 GiB |
| **Decode (no spec)** | **5.6 t/s** | 4.3 t/s | 7.5 t/s |
| **Decode (DFlash N=8)** | **19.8 t/s chat / 24.8 responses** | n/a | n/a |
| Vision | ✅ | ✅ | ⚠ no `mmproj` in our GGUF (but a Reddit user reports vision is enabled with a different build  -  not verified by us) |
| `/v1/responses` reasoning | ✅ | ✅ | ❌ |
| Tool calls | ✅ non-stream | ⚠ parser bugs | ✅ via `--jinja` |
| 256K context | ✅ | ✅ | ✅ |

**Rule of thumb:** vision + reasoning + smallest near-lossless footprint → **this repo**. Pure text + speed without vision → `llama-qwen`. Official Qwen team weights → `vllm-qwen`.

---

## 📁 Repo layout

```
.
├── Dockerfile               # multi-stage: Ubuntu + TheRock + torch + vLLM v0.20.0 from source
├── docker-compose.yml       # one service, restart=no, --enforce-eager, DFlash N=8, host-mounts ./csrc
├── .env.template            # the one config file you edit
├── glados.py                # tiny REPL/one-shot CLI for fast testing (no deps, stdlib only)
├── csrc/
│   └── awq_mmq_gfx1151/                  # HIP custom op: AWQ-INT4 MMQ Q4 prefill kernel for gfx1151
│       ├── awq_mmq_gfx1151_kernel.hip    # v0 scalar reference + v1 WMMA iu8 (mmq_x=48, mmq_y=64, nwarps=4)
│       ├── bindings.cpp                  # torch.ops.awq_mmq_gfx1151.mmq_q4_gemm dispatcher
│       ├── setup.py                      # built inside the container, --offload-arch=gfx1151
│       ├── awq_mmq_gfx1151/vllm_kernel.py # MPLinearKernel adapter + M-dispatch (M>=32 -> HIP, else Triton)
│       └── test_correctness.py           # standalone correctness vs scalar reference
├── scripts/
│   ├── install_rocm_sdk.sh  # TheRock nightly tarball -> /opt/rocm (rocm.nightlies.amd.com mirror, ~50x faster than S3 origin)
│   ├── patch_strix.py       # 18 idempotent string-replace patches (1187 LOC) - 12 from kyuz0 (verbatim) + Patches 11/12 (local boilerplate) + Patches 13/14 (PR cherry-picks #40176, #40898) + Patch 15 (local /v1/responses chat_template_kwargs) + Patch 16 (register AWQ-INT4 MMQ HIP kernel into _POSSIBLE_KERNELS[ROCM][0])
│   └── dump_logs.sh         # snapshot engine + kernel logs before any down/restart
├── test/
│   ├── bench.py                       # original 5-endpoint harness (peso prompt)
│   ├── bench_full.py                  # generic 5-endpoint + tools (chat+responses) + 2 images + Three.js
│   ├── bench_longctx.py               # 25K-token synthesis test using real .research data
│   ├── bench_matrix.py                # streaming bench matrix (ctx 0/2k/4k/8k/16k/32k x 4 cases) - the kernel-validation harness
│   ├── bench_prefill_warmup.py        # cold-vs-warm prefill rate from /metrics deltas
│   └── verify_responses_streaming.py  # SSE-traced T1-T5 reasoning/tool-call verification (post Patch 15)
├── LICENSE                  # Unlicense (public domain)
└── README.md
```

The kernel `.so` is built inside the container at startup (or out-of-band via `python setup.py build_ext --inplace` in `/workspace/csrc/awq_mmq_gfx1151/`); no vendored binaries, no untracked tarballs.

---

## 🔬 Run the benches

```bash
# Streaming bench matrix: 6 contexts x 4 cases, the harness that produced the
# Prefill table above. Use this after any kernel change to confirm the curve.
python3 test/bench_matrix.py
# -> test/bench_matrix_results.json (per-cell prefill / decode t/s, mean/min/max)

# Cold-vs-warm 1k-token prefill from /metrics deltas (autotune amortization probe)
python3 test/bench_prefill_warmup.py
# -> stdout only

# Streaming /v1/responses reasoning + tool-call verification (T1-T5 matrix)
python3 test/verify_responses_streaming.py
# -> per-test SSE event-type breakdown + reasoning/output_text channel char counts
#    (use this to confirm Patch 15 is live after a rebuild)

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

- [@micah-wil](https://github.com/micah-wil) (AMD) - PR #40176, the actual fix that lights up DFlash on RDNA 3.5
- [@jianc99](https://github.com/jianc99) (z-lab) - DFlash paper, drafter model, PR #40898
- [@kyuz0](https://github.com/kyuz0) (Donato Capitella, Reversec) - the Strix Halo + vLLM patch bundle that hardware-enables gfx1151
- [@hongxiayang](https://github.com/hongxiayang) (AMD) - MI3xx DFlash verification, vLLM issue #40632 stewardship
- [@cyankiwi](https://huggingface.co/cyankiwi) - the AWQ-INT4 quant we serve as target

---

## ⚠️ Honest limitations

- **Shipping `.env` defaults to single-stream + 64K context** (`VLLM_MAX_NUM_SEQS=1`, `VLLM_MAX_MODEL_LEN=65536`, `VLLM_GPU_MEMORY_UTIL=0.55`). The HIP prefill kernel makes single-stream fast enough that batching is no longer how you hide cost. Multi-stream (up to **3 concurrent verified**, see [profiles](#recommended-profiles) and [stress test](#-multi-stream--tool-calling-stress-test-3-concurrent)) still works, just bump `VLLM_MAX_NUM_SEQS` and adjust `MAX_MODEL_LEN`/`GPU_MEMORY_UTIL`.
- **Drafter is gated** on HuggingFace - manual approval required, no ungated mirror.
- **Tool calling has two working modes and two broken modes.** Streaming tool calls on `/v1/chat/completions` have upstream parser bugs (vLLM PRs #40785, #40787 closed unmerged); non-streaming `/v1/responses` + tools when combined with `enable_thinking=false` is broken locally (channel routing, see Patch 15 scope below). Use `/v1/chat/completions` with `stream: false`, OR `/v1/responses` with `stream: true`. Full breakdown in [the tool calling support matrix](#%EF%B8%8F-tool-calling-support-matrix).
- **Cold start ≈ 9 min on every restart**, even with all caches warm. Full breakdown in the dedicated [Spin-up time section](#%EF%B8%8F-spin-up-time-is-9-min-on-every-restart-even-with-all-caches-warm) above.
- **DFlash acceptance drops past N≈8** (drafter is less accurate further ahead). N=15 may give marginal further gain; we stopped at N=8 (best steady-state on chat is 19.8 t/s, on `/v1/responses` is 24.8).
- **Numbers in this README are wall-clock client-side.** vLLM internal generation throughput is ≈ 25-27 t/s on these workloads (engine excludes round-trip + initial prefill).
- **HIP graphs freeze on gfx1151** - `--enforce-eager` mandatory.
- **Official `Qwen/Qwen3.6-27B-FP8` doesn't init** on RDNA 3.5 (Triton w8a8 autotune stall on the hybrid model's DeltaNet partitions). AWQ-INT4 sidesteps this and is structurally a better fit since RDNA 3.5 has no native FP8 anyway.
- **Non-streaming `/v1/responses` with `enable_thinking=false` is NOT patched in this repo.** Patch 15 only covers the streaming path. The non-streaming path still misroutes content into the `reasoning` field and leaves tool-call XML unparsed. Use `stream: true` on `/v1/responses` for any agent or RAG client.
- **Long-context decode is not what the HIP kernel fixed.** Past ~8K context, decode falls into single-digit t/s (5.9 t/s @ 16K, 3.4 t/s @ 32K on the `normal` case, see [Prefill table](#-prefill-input-processing-speed)). That is **attention scaling on the KV cache**, not GEMM, and it routes through a different code path the kernel does not touch. Fixing it would require a flash-attention-style KV-aware kernel (separate scope). Prefill itself stays flat at 105 to 134 t/s across the whole 0 to 32K range.
- **Kernel adds a one-time ~22 GiB dual-storage weight cost.** `apply_weights` keeps two copies of the W4A16 weights so decode (M < 32) can fall through to `triton_w4a16_gemm` without per-call repack. This is why the shipping default is `MAX_MODEL_LEN=65536` and `GPU_MEMORY_UTIL=0.55` instead of the older 128K @ 0.5 profile. Raising to 128K or 256K still works, just tighter.

### 🚨 vLLM v0.20.0 qwen3 reasoning parser DOES NOT STREAM REASONING on `/v1/chat/completions`

Confirmed bug - independent of DFlash, present even without spec decode. With `--reasoning-parser qwen3` enabled and `stream: true`:

- **`/v1/chat/completions`**: parser **buffers** all `<think>...</think>` content server-side, emits **zero** `delta.reasoning_content` chunks during the stream. Client sees a long "ttft" silence (~10-30 s of thinking) and then a burst of the post-thinking answer at the end.
- **`/v1/responses`**: streams correctly - `response.reasoning_text.delta` events arrive at t+0.33 s, then `response.output_text.delta` after `</think>`. Same engine, same DFlash, same numbers. Different code path that actually works.

Verified via direct curl probe in this repo (`stream:true`, simple math prompt):

| Endpoint | `reasoning` deltas during stream | First delta arrives |
|---|---|---|
| `/v1/chat/completions` | **0** | t+12.52 s (just the post-think answer) |
| `/v1/responses` | **62** | **t+0.33 s** ✅ |

**Workaround for end-user clients that need streaming reasoning visible:**
- **Use `/v1/responses`** - fully working today, recommended.
- Alternatively disable the parser (remove `--reasoning-parser qwen3` from `docker-compose.yml`) and let raw `<think>...</think>` text stream as part of `delta.content`. Tradeoff: `/v1/responses` no longer auto-separates reasoning into structured `output[].type == "reasoning"` items.
- Or accept it: send `stream: false` on chat completions when you need to display reasoning.

The included `glados.py` uses `/v1/responses` to dodge the bug.

This is worth filing upstream - the qwen3 parser's streaming path appears to never call `extract_reasoning_content_streaming` correctly on the chat-completions code path. Affects every model using `--reasoning-parser qwen3`, not just DFlash workloads.

> **Transparency note**: this is the issue we hit and verified. There are almost certainly **other** vLLM v0.20.0 quirks we did not exercise - long-context corner cases, edge sampler params, multimodal + tools combinations, prefix-cache edge cases, etc. We only documented what our bench surfaced. If you run a workload mode we didn't test (different drafter, different attention backend, hybrid grad cudagraph, etc.) and hit something weird, the right move is *not* "the repo is broken" but "vLLM v0.20.0 is early on this code path - check upstream issues, file if new". DFlash + reasoning-parser + ROCm-on-RDNA3.5 are all simultaneously in active flux upstream as of 2026-04-26.

### ✅ Verified streaming behavior with Patch 15

Originally `chat_template_kwargs.enable_thinking=false` was silently ignored on `/v1/responses` (vLLM's `ResponsesRequest` had no field for it; the renderer received an empty kwargs dict). **Local Patch 15** wires the field through and re-runs prove the kwarg now flows correctly.

The Qwen3.6 chat template (verbatim from `chat_template.jinja:148-152`):

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}        ← empty think block prefilled
    {%- else %}
        {{- '<think>\n' }}                       ← open think tag, model thinks
    {%- endif %}
{%- endif %}
```

When `enable_thinking=false` reaches the template, it injects a closed `<think></think>` at the start of the assistant turn. The model's first generated token is therefore *past* the closed think block: it physically cannot emit reasoning. Reasoning is **genuinely skipped, not hidden** - latency drops accordingly.

**Verification matrix** (`python3 test/verify_responses_streaming.py`, all five tests on the running engine):

| Test | Setup | Reasoning chars | Output text chars | Tool result | Wall | Verdict |
|---|---|---:|---:|---|---:|---|
| T1 | tiny prompt, no tools, `think_off` | 0 | 2 (`"42"`) | n/a | 0.47 s | ✅ |
| T2 | tiny prompt, with tools, `think_off` | 0 | 0 | `get_weather({"city":"Tokyo"})` parsed | 3.34 s | ✅ |
| T3 | ~2K context, no tools, `think_off` | 0 | 100 (correct synthesis answer) | n/a | 7.92 s | ✅ |
| T4 | ~2K context, with tools, `think_off` | 0 | 0 | `get_weather({"city":"Santa Clara"})` parsed | 12.52 s | ✅ |
| T5 | tiny, no tools, `think_ON` (control) | 354 | 4 (`"\n\n42"`) | n/a | 6.13 s | ✅ |

T5 proves the channel routing is preserved when reasoning IS expected: with thinking on, the `reasoning_text` channel populates and the `output_text` channel still gets the final answer. With thinking off (T1-T4), `reasoning_text` is empty across both no-tool and tool variants and across both tiny and 2K-context prompts.

**One known gap deliberately left out of scope**: non-streaming `/v1/responses` (i.e. `stream=false`) with `enable_thinking=false` still misroutes content into the `reasoning_text` field, and tool-call XML stays raw inside that field instead of being parsed into `function_call` items. The fix is a separate, larger patch (port `prompt_is_reasoning_end` from `chat_completion/serving.py`). For this repo we **only support the streaming path** on `/v1/responses`. Always send `stream: true` and the engine behaves correctly.

**Practical client guidance**:
- For fast non-thinking responses on `/v1/responses`: send `stream: true` + `chat_template_kwargs: {"enable_thinking": false}`. Patch 15 makes this work end-to-end.
- For streaming reasoning visible to a UI: send `stream: true` and omit the kwarg (default thinking on); reasoning arrives as `response.reasoning_text.delta` events.
- Patch 15 should land upstream as a vLLM PR  -  the gap looks accidental and the fix is six lines.

### 🚨 DFlash speculative decoding is *early upstream - fragile path*

DFlash landed in vLLM main on **2026-03-30** ([PR #36847](https://github.com/vllm-project/vllm/pull/36847)). As of the time of this README, **5+ DFlash bug-fix PRs are still open** and several DFlash-class bug reports are unresolved across multiple GPU vendors (NVIDIA, AMD, both):

| Open issue / PR | Title | Vendor | Why it matters here |
|---|---|---|---|
| [#39928](https://github.com/vllm-project/vllm/issues/39928) | Qwen3.5 DFlash gives strange responses on SM90 | NVIDIA H100 | Confirms DFlash output-quality bugs are not unique to AMD |
| [#40624](https://github.com/vllm-project/vllm/issues/40624) | Gemma4 0% prefix cache hits with hybrid attention + DFlash | All | Hybrid (DeltaNet-style) attention + DFlash interaction is the same class as our Qwen 3.6-27B target |
| [#40382](https://github.com/vllm-project/vllm/issues/40382) | Gemma-4 + DFlash unservable on Ampere - non-causal + head_dim=256 has no compatible attention backend | NVIDIA A100 | DFlash needs `supports_non_causal=True` backends; not all backends qualify (this is exactly what our Patch 13 fixes for `ROCM_ATTN`) |
| [#40425](https://github.com/vllm-project/vllm/pull/40425) | Fix quantized DFlash Qwen3 draft support | All | Open PR fixing quantized drafter loading |
| [#40334](https://github.com/vllm-project/vllm/pull/40334) | fix(dflash): dtype mismatch in `combine_hidden_states` | All | Open PR fixing a dtype bug in the auxiliary-hidden-state path |
| [#40727](https://github.com/vllm-project/vllm/pull/40727) | Update dflash aux layer indexing | All | Related to the same `+1` shift bug our Patch 14d (PR #40898) addresses |
| [#40632](https://github.com/vllm-project/vllm/issues/40632) | Support DFlash for Kimi K2.5 and Qwen3.5-27B for AMD | AMD | The umbrella issue for AMD-side DFlash support - our work plugs into this |

**Bottom line: DFlash is not "stable upstream" yet on any vendor.** It works for our specific Qwen 3.6-27B + AWQ4 + N=8 path because we've patched around the issues we hit, but you may encounter unfixed-upstream behaviors not seen here, especially with different drafters, different N values, or different attention backends.

If output quality looks off (incoherent, repetitive, garbled), **first try `--speculative-config '{"method":"dflash", ..., "num_speculative_tokens":1}'`** - that's the safest setting; it isolates whether the issue is DFlash itself vs the multi-token spec-decode path.

<details>
<summary><b>Stuck DFlash worker on client disconnect - what we hit during the long-context test (recovery documented)</b></summary>

**Symptom (we hit it once during ~22K-token long-context decode):** if the client TCP-disconnects mid-decode while `--speculative-config method=dflash` is active, the EngineCore worker may not gracefully unwind. The API server (`/health`, `/v1/models`) stays responsive but new `/v1/chat/completions` requests time out forever. EngineCore burns 79-200% CPU in a tight loop. Kernel logs:

```
workqueue: kfd_process_wq_release [amdgpu] hogged CPU for >10000us 5 times,
consider switching to WQ_UNBOUND
```

**Likely cause:** a refcount / cancel race between the DFlash proposer thread and the KFD GPU buffer release path when a request is cancelled mid-decode. The DFlash worker keeps running while the client cancellation tries (and fails) to reclaim its buffers. Plausibly related to the open issues table above - DFlash's cancellation/cleanup path is not yet hardened upstream.

**Recovery:** `docker compose restart vllm-awq4-qwen` (≈ 9 min cold boot). Run `./scripts/dump_logs.sh stuck-state` *before* the restart to preserve diagnostics for an upstream report.

**Mitigation while waiting for upstream fix:**
- Prefer client-side timeouts larger than your expected decode time
- Avoid `SIGKILL`'ing the client mid-stream - use `SIGINT` or wait for graceful completion
- For very long-context requests (>10K prompt tokens), consider testing with `num_speculative_tokens=1` first to confirm baseline correctness, then ramp up

If you reproduce this, file under the [DFlash umbrella tracker (#40632)](https://github.com/vllm-project/vllm/issues/40632) with output of `./scripts/dump_logs.sh`.

</details>

---

<details>
<summary><img src="https://img.shields.io/badge/%F0%9F%9F%A7_HIDDEN APERTURE_TRANSMISSION-CLICK_TO_DECRYPT-FF6B00?style=for-the-badge&labelColor=222222" alt="Hidden aperture transmission - click to decrypt" /></summary>

```
                  .,-:;//;:=,
              . :H@@@MM@M#H/.,+%;,
           ,/X+ +M@@M@MM%=,-%HMMM@X/,
         -+@MM; $M@@MH+-,;XMMMM@MMMM@+-
        ;@M@@M- XM@X;. -+XXXXXHHH@M@M#@/.
      ,%MM@@MH ,@%=             .---=-=:=,.
      =@#@@@MX.,                -%HX$$%%%:;
     =-./@M@M$                   .;@MMMM@MM:
     X@/ -$MM/                    . +MM@@@M$
    ,@M@H: :@:                    . =X#@@@@-
    ,@@@MMX, .                    /H- ;@M@M=
    .H@@@@M@+,                    %MM+..%#$.
     /MMMM@MMH/.                  XM@MH; =;
      /%+%$XHH@$=              , .H@@@@MX,
       .=--------.           -%H.,@@@@@MX,
       .%MM@@@HHHXX$$$%+- .:$MMX =M@@MM%.
         =XMMM@MM@MM#H;,-+HMM@M+ /MMMX=
           =%@M@M#@$-.=$@MM@@@M; %M%=
             ,:+$+-,/H#MMMMMMM@= =,
                  =++%%%%+/:-.
```
>
> *Welcome to GLaDOS, powered by Qwen 3.6-27B (AWQ-INT4) + DFlash on AMD Strix Halo.*

🟧 **You found the secret CLI.** Once your engine is running:

```bash
docker compose up -d
# wait for: "Application startup complete" in `docker logs -f vllm-awq4-qwen`

./glados.py                       # interactive REPL with Aperture banner + a random GLaDOS quote
./glados.py "explain mitosis"     # one-shot
```

In the REPL: type your prompt and hit Enter. Streams `<thinking>...</thinking>` live (via the `/v1/responses` path that actually works), then the answer, then a one-line stats summary: `prompt→output tokens · wall time · wall t/s · vLLM ground-truth t/s · DFlash acceptance %`.

To leave: `exit`, `quit`, `:q`, or `Ctrl-D`. To abort an in-flight generation without leaving: `Ctrl-C`.

The banner shows in **Aperture orange** in your terminal (markdown above can't render true color, but `glados.py` uses ANSI 208 in the actual TTY).

*Thank you for participating in this Aperture Science computer-aided enrichment activity.*

</details>

---

## 📜 License

[The Unlicense](LICENSE) - public domain. Use, modify, distribute, sell, fork. No attribution required, no warranty given.
