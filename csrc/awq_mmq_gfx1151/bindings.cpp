// pybind11 / TORCH_LIBRARY bindings for the AWQ-INT4 MMQ HIP custom op.
//
// Exposes torch.ops.awq_mmq_gfx1151.mmq_q4_gemm(x, w_packed, scales, version) -> out.
//
// version: 0 = v0 scalar reference (always works), 1 = v1 WMMA + LDS staging.
//
// Tensor contracts (verified against vLLM v0.20.0 compressed_tensors_wNa16.py):
//   x         : (M, K)        fp16,  CUDA, contiguous
//   w_packed  : (N, K / 8)    int32, CUDA, contiguous
//                 8 uint4 values per int32, low-nibble first (shifts 0,4,...,28).
//                 weight_type = uint4b8: stored values [0,15] decode to signed [-8,7]
//                 via subtraction by 8 (matches llama.cpp's __vsubss4 recenter).
//   scales    : (N, K / 32)   fp16,  CUDA, contiguous (group_size=32 for cyankiwi AWQ4)
//   out       : (M, N)        fp16,  CUDA, contiguous, allocated here

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_mmq_q4_gemm_gfx1151(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    at::Tensor& out,
    int64_t version);

namespace {

constexpr int64_t kPackFactor = 8;
constexpr int64_t kGroupSize = 32;

at::Tensor mmq_q4_gemm_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    int64_t version) {

    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(w_packed.is_cuda(), "w_packed must be CUDA");
    TORCH_CHECK(scales.is_cuda(), "scales must be CUDA");

    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");

    TORCH_CHECK(x.dim() == 2 && w_packed.dim() == 2 && scales.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() && scales.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(0);

    TORCH_CHECK(w_packed.size(1) * kPackFactor == K,
                "w_packed last dim mismatch: expected K/8 = ", K / kPackFactor,
                " got ", w_packed.size(1));
    TORCH_CHECK(scales.size(0) == N && scales.size(1) * kGroupSize == K,
                "scales shape mismatch: expected (", N, ", ", K / kGroupSize,
                ") got (", scales.size(0), ", ", scales.size(1), ")");
    TORCH_CHECK(K % kGroupSize == 0, "K must be divisible by group_size (32)");
    TORCH_CHECK(version == 0 || version == 1, "version must be 0 (scalar) or 1 (WMMA)");

    if (w_zeros.defined() && w_zeros.numel() > 0) {
        // Packed [N/8, K/32] int32, 8 uint4 zeros per int32 (TritonW4A16 layout).
        TORCH_CHECK(w_zeros.is_cuda() && w_zeros.scalar_type() == at::kInt,
                    "w_zeros must be CUDA int32 (packed)");
        TORCH_CHECK(w_zeros.dim() == 2 && w_zeros.size(0) * 8 == N && w_zeros.size(1) * kGroupSize == K,
                    "w_zeros shape mismatch: expected (", N / 8, ", ", K / kGroupSize, ")");
        TORCH_CHECK(w_zeros.is_contiguous(), "w_zeros must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_q4_gemm_gfx1151(x, w_packed, scales, w_zeros, out, version);
    return out;
}

}  // namespace

TORCH_LIBRARY(awq_mmq_gfx1151, m) {
    m.def("mmq_q4_gemm(Tensor x, Tensor w_packed, Tensor scales, Tensor w_zeros, int version) -> Tensor");
}

TORCH_LIBRARY_IMPL(awq_mmq_gfx1151, CUDA, m) {
    m.impl("mmq_q4_gemm", &mmq_q4_gemm_forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "AWQ-INT4 MMQ kernel for gfx1151. torch.ops.awq_mmq_gfx1151.mmq_q4_gemm(x, w_packed, scales, version)";
}
