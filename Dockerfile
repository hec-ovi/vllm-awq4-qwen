# vllm-awq4-qwen source-build image  -  Ubuntu 26.04 + TheRock ROCm 7.13 nightly + vLLM from source.
#
# Builds vLLM 0.20.0+ against TheRock ROCm 7.13 nightly for AMD Strix Halo
# (gfx1151 / RDNA 3.5). Target model: cyankiwi/Qwen3.6-27B-AWQ-INT4
# (compressed-tensors W4A16, group_size 32, vision tower preserved BF16).
#
# Why this differs from vllm-qwen (BF16):
#   - vLLM v0.20.0 (released 2026-04-23) is the first stable cut adding
#     gfx1150/1151/1201 device IDs AND PR #36505 which routes AWQ
#     through AWQMarlinLinearMethod -> ConchLinearKernel on ROCm:
#     measured +57% prefill / +73% decode on gfx1151 vs the legacy
#     ops.awq_gemm path. We pin VLLM_COMMIT to the v0.20.0 release tag
#     by default; .env can override.
#   - Wheel index switched from rocm.prereleases (frozen at 7.12.0rc1)
#     to rocm.nightlies.amd.com/v2-staging/gfx1151/ which serves the
#     live 7.13.0a daily set (matched torch/torchvision/torchaudio/triton).
#
# What we DON'T build:
#   - AITER custom build  : SKIPPED. Disabled at runtime via
#                           VLLM_ROCM_USE_AITER=0; the patch bundle also
#                           fences AITER FP8 / MoE / RMSNorm off on gfx1x.
#   - Flash-Attention      : SKIPPED. ROCm/flash-attention main_perf HEAD
#                           is a 2.2-3.7x ViT regression on gfx1151
#                           (Dao-AILab/flash-attention#2392). vLLM's
#                           TRITON_ATTN + AOTriton experimental SDPA is
#                           the recommended path on RDNA 3.5.
#   - bitsandbytes ROCm    : SKIPPED. AWQ via compressed-tensors.
#   - Custom RCCL          : SKIPPED. Single-iGPU, no NCCL.

FROM ubuntu:26.04

# 1. Build + inference-runtime system deps (inference-only  -  no IB, no ffmpeg).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential cmake ninja-build \
      aria2 tar xz-utils \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. TheRock ROCm SDK -> /opt/rocm. The install script resolves the latest
# 7.13.0a nightly tarball at build time (today: 7.13.0a20260426).
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
    ROCM_MAJOR_VER=${ROCM_MAJOR_VER} GFX=${GFX} /tmp/install_rocm_sdk.sh && \
    rm /tmp/install_rocm_sdk.sh

# 3. Python 3.12 venv via uv (Ubuntu 26.04 ships 3.14, no python3.12 apt pkg).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install --upgrade pip wheel packaging "setuptools<80.0.0"

# 4. PyTorch + triton from the AMD v2-staging gfx1151 nightly index.
# This index serves complete matched daily sets; today resolves to:
#   torch        2.10.0+rocm7.13.0a20260426
#   torchvision  0.26.0+rocm7.13.0a20260426
#   torchaudio   2.9.0 +rocm7.13.0a20260426
#   triton       3.6.0 +rocm7.13.0a20260426
RUN uv pip install --pre torch torchvision torchaudio \
      --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ && \
    rm -rf /root/.cache/uv /root/.cache/pip

# 5. Build tool deps for the vLLM native build.
RUN uv pip install --upgrade cmake ninja packaging wheel numpy \
      "setuptools-scm>=8" "setuptools<80.0.0" scikit-build-core pybind11

# 6. Conch Triton kernels  -  required by vLLM's AWQMarlin-on-ROCm path
# selected via choose_mp_linear_kernel (PR #36505). Without this, AWQ
# falls back to the slow legacy ops.awq_gemm.
RUN uv pip install conch-triton-kernels

# 7. Clone vLLM and apply the Strix Halo patch bundle.
# Default pin: v0.20.0 release tag (2026-04-23) which includes:
#   - PR #36505: AWQMarlin on ROCm (+57% prefill / +73% decode for AWQ)
#   - RDNA 3.5/4 device-ID detection (gfx1150/1151/1201)
#   - Initial GDN attention for Qwen3-Next / Qwen3.5
# Override via build-arg if you need a different sha or tracking HEAD.
ARG VLLM_COMMIT=v0.20.0
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm
RUN if [ -n "$VLLM_COMMIT" ]; then \
      echo "Pinning vLLM to ${VLLM_COMMIT}"; \
      git checkout "${VLLM_COMMIT}"; \
    else \
      echo "Tracking vLLM HEAD: $(git rev-parse --short HEAD)"; \
    fi

COPY scripts/patch_strix.py /opt/vllm/patch_strix.py
RUN python /opt/vllm/patch_strix.py

# 7b. pkg-config  -  required by ROCm 7.13's rocm_smi-config.cmake which
# vLLM's find_package(Torch) → Caffe2 → LoadHIP chain pulls in. Kept as
# a separate RUN here (rather than in step 1) so adding it doesn't
# invalidate the ROCm tarball / torch wheel cache layers above.
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && \
    rm -rf /var/lib/apt/lists/*

# 8. Build vLLM against TheRock ROCm.
# CC/CXX forced to ROCm clang so the compiled extensions have an ABI
# matching the torch wheels.
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    CMAKE_PREFIX_PATH=/opt/rocm \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1151 \
    HIP_ARCHITECTURES=gfx1151 \
    GPU_TARGETS=gfx1151 \
    AMDGPU_TARGETS=gfx1151 \
    MAX_JOBS=4 \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
    echo "Building with bitcode: $HIP_DEVICE_LIB_PATH" && \
    export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151" && \
    uv pip install --no-build-isolation --no-deps -v . && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/*

# 8b. vLLM runtime dependencies.
# vLLM declares deps as `dynamic` in pyproject.toml and loads them from
# requirements/common.txt + requirements/rocm.txt at build time. Built
# with --no-deps above; install the same lists here. The constraint pins
# torch/triton to our v2-staging wheels so uv doesn't replace them with
# vanilla CUDA torch from PyPI when resolving.
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    TRITON_VER=$(python -c "import triton; print(triton.__version__)") && \
    printf "torch==%s\ntriton==%s\n" "$TORCH_VER" "$TRITON_VER" > /tmp/constraints.txt && \
    echo "Pinning runtime deps to torch==$TORCH_VER, triton==$TRITON_VER" && \
    uv pip install --no-build-isolation \
      -r /opt/vllm/requirements/common.txt \
      -r /opt/vllm/requirements/rocm.txt \
      --constraint /tmp/constraints.txt && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/constraints.txt

# 9. Runtime env. Mirrors /etc/profile.d/rocm-sdk.sh for non-interactive use.
# These are the env vars our research found load-bearing for AWQ vision
# on gfx1151  -  see .research/flashattention-strix-halo and
# .research/vllm-strix-halo-issues for the per-flag justification.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    ROCBLAS_USE_HIPBLASLT=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    MIOPEN_FIND_MODE=FAST \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    VLLM_ROCM_USE_AITER=0 \
    VLLM_USE_TRITON_AWQ=1 \
    VLLM_DISABLE_COMPILE_CACHE=1

WORKDIR /opt
CMD ["/bin/bash"]
