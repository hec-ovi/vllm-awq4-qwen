// AWQ-INT4 MMQ kernel for prefill on AMD Strix Halo (gfx1151, RDNA 3.5).
//
// Two kernels coexist:
//   v0: scalar reference (one thread per output element). Slow but
//       mathematically correct. Validates data path / harness / build.
//   v1: WMMA + LDS staging with the (mmq_x=48, mmq_y=64, nwarps=4) tile
//       shape from llama.cpp's pedapudi gist for RDNA3_5. Uses
//       __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32 chained twice per
//       K-group of 32 (WMMA K=16, group_size=32).
//
// Tensor convention (matches vLLM's compressed_tensors_wNa16):
//   x         (M, K)        fp16    activations
//   w_packed  (N, K/8)      int32   weights, 8 uint4b8 per int32
//   scales    (N, K/32)     fp16    per-group weight scales
//   out       (M, N)        fp16    output
//
// MMQ tile semantics (from llama.cpp): MMQ_X is the N tile (weight rows /
// output cols), MMQ_Y is the M tile (activation rows / output rows).
//
// Reference: .research/mmq-q4-gfx1151-port/FINDINGS.md

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace awq_mmq_gfx1151 {

constexpr int GROUP_SIZE = 32;
constexpr int PACK_FACTOR = 8;     // 8 uint4 per int32

// ============================================================
// v0: scalar reference (one thread per output element)
// ============================================================
constexpr int V0_BLOCK_M = 16;
constexpr int V0_BLOCK_N = 16;

__global__ void mmq_q4_gemm_kernel_v0(
    const __half*   __restrict__ x,
    const int*      __restrict__ w_packed,
    const __half*   __restrict__ scales,
    __half*         __restrict__ out,
    int M, int N, int K) {

    const int row = blockIdx.y * V0_BLOCK_M + threadIdx.y;
    const int col = blockIdx.x * V0_BLOCK_N + threadIdx.x;
    if (row >= M || col >= N) return;

    const int num_k_groups = K / GROUP_SIZE;
    const int packed_per_row = K / PACK_FACTOR;

    float acc = 0.0f;
    for (int g = 0; g < num_k_groups; ++g) {
        const float group_scale = __half2float(scales[col * num_k_groups + g]);
        float group_acc = 0.0f;
        const int k_base = g * GROUP_SIZE;
        for (int kk = 0; kk < GROUP_SIZE; ++kk) {
            const int k = k_base + kk;
            const int word = w_packed[col * packed_per_row + (k / PACK_FACTOR)];
            const int int8_val = ((word >> ((k % PACK_FACTOR) * 4)) & 0xF) - 8;
            group_acc += static_cast<float>(int8_val) * __half2float(x[row * K + k]);
        }
        acc += group_acc * group_scale;
    }
    out[row * N + col] = __float2half(acc);
}

// ============================================================
// v1: WMMA + LDS staging
// ============================================================
// MMQ_X chosen 64 instead of pedapudi's 48 because it divides Qwen 3.6 27B's
// hidden (5120) and intermediate (27648) without remainder. Bench (TODO) will
// confirm whether the wider tile costs prefill perf vs the published 48.
constexpr int MMQ_X = 64;            // N tile = weight rows = output cols
constexpr int MMQ_Y = 64;            // M tile = activation rows = output rows
constexpr int NWARPS = 4;
constexpr int WAVE_SIZE = 32;
constexpr int K_PER_INT = 4;         // 4 int8 per int32 in LDS

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int SUBTILES_M = MMQ_Y / WMMA_M;  // 4 -> warp_id maps 1:1 to M-row-sub-tile
constexpr int SUBTILES_N = MMQ_X / WMMA_N;  // 3 -> each warp iterates these

constexpr int K_INTS_PER_GROUP = GROUP_SIZE / K_PER_INT;  // 8 ints per row per K-group

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int8_t quant_fp32_to_int8(float v) {
    if (v >  127.0f) return  127;
    if (v < -128.0f) return -128;
    return static_cast<int8_t>(__float2int_rn(v));
}

// =================== Activation quantization pre-kernel ===================
// One block per activation row. Computes per-row max(|x|), writes the scale
// (max_abs / 127) and the int8 quantized row. Output:
//   x_int8     : (M, K) int8
//   act_scales : (M,)   fp32
//
// One block = 256 threads (8 warps wave32). Each thread strides through K
// computing local max, then warp-reduces, then block-reduces via LDS.
constexpr int ACTQUANT_BLOCK = 256;

__global__ void compute_act_int8_and_scales_kernel(
    const __half* __restrict__ x,
    int8_t*       __restrict__ x_int8,
    float*        __restrict__ act_scales,
    int M, int K) {

    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const __half* x_row = x + row * K;
    int8_t* out_row = x_int8 + row * K;

    // Pass 1: per-thread local max(|x|).
    float local_max = 0.0f;
    for (int k = tid; k < K; k += ACTQUANT_BLOCK) {
        float v = fabsf(__half2float(x_row[k]));
        if (v > local_max) local_max = v;
    }

    // Warp reduce via shuffle.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor(local_max, offset);
        if (other > local_max) local_max = other;
    }

    __shared__ float warp_maxes[ACTQUANT_BLOCK / 32];  // 8 warps
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) warp_maxes[warp_id] = local_max;
    __syncthreads();

    // First warp final-reduces 8 warp maxes.
    if (warp_id == 0) {
        float v = (lane < ACTQUANT_BLOCK / 32) ? warp_maxes[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            float other = __shfl_xor(v, offset);
            if (other > v) v = other;
        }
        if (lane == 0) {
            // act_scale = max_abs / 127 (saturate at tiny epsilon to avoid div-by-zero)
            float scale = v / 127.0f;
            if (scale < 1e-8f) scale = 1e-8f;
            warp_maxes[0] = scale;
            act_scales[row] = scale;
        }
    }
    __syncthreads();
    const float scale = warp_maxes[0];
    const float inv_scale = 1.0f / scale;

    // Pass 2: quantize each x[row,k] -> int8 using the row scale.
    for (int k = tid; k < K; k += ACTQUANT_BLOCK) {
        out_row[k] = quant_fp32_to_int8(__half2float(x_row[k]) * inv_scale);
    }
}

__global__ void mmq_q4_gemm_kernel_v1(
    const int8_t*   __restrict__ x_int8,        // (M, K) int8, pre-quantized by act-quant pre-kernel
    const float*    __restrict__ act_scales,    // (M,) fp32, per-row activation scale = max_abs / 127
    const int*      __restrict__ w_packed,      // (N, K/8) int32, 8 uint4 per int32 (low nibble first)
    const __half*   __restrict__ w_scales,      // (N, K/32) fp16, per-group weight scale
    const int*      __restrict__ w_zeros_packed,// (N/8, K/32) int32, 8 uint4 per int32 (TritonW4A16 layout)
    __half*         __restrict__ out,           // (M, N) fp16
    int M, int N, int K) {

    const int block_m = blockIdx.y * MMQ_Y;
    const int block_n = blockIdx.x * MMQ_X;
    const int tid = threadIdx.x;
    const int warp_id = tid / WAVE_SIZE;       // [0, 4)
    const int lane = tid % WAVE_SIZE;          // [0, 32)
    const int lane_lo = lane & 0x0F;           // [0, 16)  WMMA C row in 16x16 sub-tile
    const int lane_hi = lane >> 4;             // {0, 1}   WMMA C col-half offset

    // ---- LDS tiles (K-contiguous int8 packed 4-per-int32) ----
    // Single-buffered. We tried double-buffering with software pipelining;
    // microbench showed the same throughput but end-to-end vllm regressed
    // ~5%. The compiler-driven instruction reordering already overlaps load
    // with WMMA enough; the doubled LDS footprint and lambda-call indirection
    // cost more than the explicit prefetch saved.
    __shared__ int   weight_tile[MMQ_X][K_INTS_PER_GROUP];   // 64 * 8 * 4 = 2 KB
    __shared__ int   act_tile   [MMQ_Y][K_INTS_PER_GROUP];   // 64 * 8 * 4 = 2 KB
    __shared__ float weight_scales_lds[MMQ_X];               // 64 * 4 = 256 B
    __shared__ float act_scales_lds[MMQ_Y];                  // 64 * 4 = 256 B
    __shared__ int   weight_zeros_lds[MMQ_X];                // per-K-group zero per N row

    // Per-warp accumulator: SUBTILES_N col-sub-tiles, each int32x8 (8 i32 per lane).
    int32x8_t acc_i32[SUBTILES_N];
    float acc_fp32[SUBTILES_N][8];
    #pragma unroll
    for (int s = 0; s < SUBTILES_N; ++s) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) acc_fp32[s][e] = 0.0f;
    }

    const int num_k_groups = K / GROUP_SIZE;
    const int packed_per_row_global = K / PACK_FACTOR;
    const int packed_per_row_in_group = GROUP_SIZE / PACK_FACTOR;  // 4

    // act_scales is constant across K-groups; load once.
    if (tid < MMQ_Y) {
        const int abs_m = block_m + tid;
        act_scales_lds[tid] = (abs_m < M) ? act_scales[abs_m] : 0.0f;
    }

    for (int kg = 0; kg < num_k_groups; ++kg) {
        const int k_base = kg * GROUP_SIZE;

        // STAGE 1: per-K-group metadata (scales, zeros) into LDS.
        // Zeros stored in TritonW4A16's packed [N/8, K/G] int32 layout; we unpack inline.
        if (tid < MMQ_X) {
            const int abs_n = block_n + tid;
            float s = 0.0f;
            int z = 8;  // symmetric uint4b8 baseline
            if (abs_n < N) {
                s = __half2float(w_scales[abs_n * num_k_groups + kg]);
                if (w_zeros_packed != nullptr) {
                    const int packed = w_zeros_packed[(abs_n / 8) * num_k_groups + kg];
                    z = (packed >> ((abs_n % 8) * 4)) & 0xF;
                }
            }
            weight_scales_lds[tid] = s;
            weight_zeros_lds[tid] = z;
        }
        __syncthreads();  // weight unpack reads weight_zeros_lds, must be populated first

        // STAGE 2: Weights to LDS (zp from LDS for tighter coalesced gmem-load behavior).
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int packed_idx = tid + i * 128;
            if (packed_idx >= MMQ_X * packed_per_row_in_group) break;
            const int row = packed_idx / packed_per_row_in_group;
            const int pkg = packed_idx % packed_per_row_in_group;
            const int abs_n = block_n + row;
            int p = 0;
            if (abs_n < N) {
                p = w_packed[abs_n * packed_per_row_global + (k_base / PACK_FACTOR) + pkg];
            }
            const int zp = weight_zeros_lds[row];
            int int_lo = 0, int_hi = 0;
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                const int v_lo = ((p >> (n * 4))       & 0xF) - zp;
                const int v_hi = ((p >> ((n + 4) * 4)) & 0xF) - zp;
                int_lo |= (v_lo & 0xFF) << (n * 8);
                int_hi |= (v_hi & 0xFF) << (n * 8);
            }
            weight_tile[row][pkg * 2 + 0] = int_lo;
            weight_tile[row][pkg * 2 + 1] = int_hi;
        }

        // Activations to LDS.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int int_idx = tid + i * 128;
            if (int_idx >= MMQ_Y * K_INTS_PER_GROUP) break;
            const int row = int_idx / K_INTS_PER_GROUP;
            const int int_in_K = int_idx % K_INTS_PER_GROUP;
            const int abs_m = block_m + row;
            const int k_int = k_base + int_in_K * 4;
            int packed = 0;
            if (abs_m < M && k_int + 3 < K) {
                packed = *reinterpret_cast<const int*>(&x_int8[abs_m * K + k_int]);
            } else {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const int k = k_int + n;
                    int8_t q = 0;
                    if (abs_m < M && k < K) q = x_int8[abs_m * K + k];
                    packed |= static_cast<int>(static_cast<unsigned char>(q)) << (n * 8);
                }
            }
            act_tile[row][int_in_K] = packed;
        }

        __syncthreads();  // ensure tiles + scales are visible to all warps before WMMA

        // Reset i32 accumulator for this group.
        #pragma unroll
        for (int s = 0; s < SUBTILES_N; ++s) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) reinterpret_cast<int*>(&acc_i32[s])[e] = 0;
        }

        // WMMA inner loop: each warp does SUBTILES_N col-sub-tiles, two chained K-halves.
        const int a_row = warp_id * WMMA_M + lane_lo;
        const int* a_ptr = &act_tile[a_row][0];

        int32x4_t a_vec0, a_vec1;
        reinterpret_cast<int*>(&a_vec0)[0] = a_ptr[0];
        reinterpret_cast<int*>(&a_vec0)[1] = a_ptr[1];
        reinterpret_cast<int*>(&a_vec0)[2] = a_ptr[2];
        reinterpret_cast<int*>(&a_vec0)[3] = a_ptr[3];
        reinterpret_cast<int*>(&a_vec1)[0] = a_ptr[4];
        reinterpret_cast<int*>(&a_vec1)[1] = a_ptr[5];
        reinterpret_cast<int*>(&a_vec1)[2] = a_ptr[6];
        reinterpret_cast<int*>(&a_vec1)[3] = a_ptr[7];

        #pragma unroll
        for (int col_t = 0; col_t < SUBTILES_N; ++col_t) {
            const int b_row = col_t * WMMA_N + lane_lo;
            const int* b_ptr = &weight_tile[b_row][0];

            int32x4_t b_vec0, b_vec1;
            reinterpret_cast<int*>(&b_vec0)[0] = b_ptr[0];
            reinterpret_cast<int*>(&b_vec0)[1] = b_ptr[1];
            reinterpret_cast<int*>(&b_vec0)[2] = b_ptr[2];
            reinterpret_cast<int*>(&b_vec0)[3] = b_ptr[3];
            reinterpret_cast<int*>(&b_vec1)[0] = b_ptr[4];
            reinterpret_cast<int*>(&b_vec1)[1] = b_ptr[5];
            reinterpret_cast<int*>(&b_vec1)[2] = b_ptr[6];
            reinterpret_cast<int*>(&b_vec1)[3] = b_ptr[7];

            // Two chained WMMA calls. Convention: A=weights, B=acts (mirrors
            // llama.cpp's mma(C, A=weights, B=activations); verified by probe).
            acc_i32[col_t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, b_vec0, true, a_vec0, acc_i32[col_t], true);
            acc_i32[col_t] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, b_vec1, true, a_vec1, acc_i32[col_t], true);
        }

        // Apply per-K-group scale: i32 accumulator -> fp32 running sum.
        const int row_in_block = warp_id * WMMA_M + lane_lo;
        const float a_scale = act_scales_lds[row_in_block];
        #pragma unroll
        for (int col_t = 0; col_t < SUBTILES_N; ++col_t) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int col_in_subtile = 2 * e + lane_hi;
                const int col_in_block = col_t * WMMA_N + col_in_subtile;
                const float w_scale = weight_scales_lds[col_in_block];
                const int v_i32 = reinterpret_cast<int*>(&acc_i32[col_t])[e];
                acc_fp32[col_t][e] += static_cast<float>(v_i32) * w_scale * a_scale;
            }
        }

        __syncthreads();
    }

    // ====== Epilogue: cast fp32 -> fp16, write to out ======
    #pragma unroll
    for (int col_t = 0; col_t < SUBTILES_N; ++col_t) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row_in_block = warp_id * WMMA_M + lane_lo;
            const int col_in_block = col_t * WMMA_N + 2 * e + lane_hi;
            const int abs_row = block_m + row_in_block;
            const int abs_col = block_n + col_in_block;
            if (abs_row < M && abs_col < N) {
                out[abs_row * N + abs_col] = __float2half(acc_fp32[col_t][e]);
            }
        }
    }
}

}  // namespace awq_mmq_gfx1151

// =================== Host launcher ===================
// w_zeros is optional (pass an empty tensor for symmetric uint4b8).
// w_zeros expected layout: (N/8, K/32) int32, packed 8 uint4 per int32.
void launch_mmq_q4_gemm_gfx1151(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    at::Tensor& out,
    int64_t version) {

    using namespace awq_mmq_gfx1151;

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = w_packed.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (version == 1) {
        // v1.1: act-quant pre-kernel + WMMA + LDS staging
        TORCH_CHECK(K % GROUP_SIZE == 0, "v1 requires K % ", GROUP_SIZE, " == 0");
        // M and N tails are handled by in-kernel bounds checks; no shape mod required.

        // Allocate intermediate buffers for the pre-quantized activations.
        auto x_int8 = at::empty({M, K}, x.options().dtype(at::kChar));
        auto act_scales = at::empty({M}, x.options().dtype(at::kFloat));

        // Pre-kernel: per-row max-abs reduction + int8 quantization.
        dim3 quant_block(ACTQUANT_BLOCK);
        dim3 quant_grid(M);
        compute_act_int8_and_scales_kernel<<<quant_grid, quant_block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            x_int8.data_ptr<int8_t>(),
            act_scales.data_ptr<float>(),
            M, K);

        // Main GEMM kernel; ceil division for tail blocks.
        dim3 block(NWARPS * WAVE_SIZE);
        dim3 grid((N + MMQ_X - 1) / MMQ_X, (M + MMQ_Y - 1) / MMQ_Y);
        const int* w_zeros_ptr = (w_zeros.defined() && w_zeros.numel() > 0)
                                 ? w_zeros.data_ptr<int>() : nullptr;
        mmq_q4_gemm_kernel_v1<<<grid, block, 0, stream>>>(
            x_int8.data_ptr<int8_t>(),
            act_scales.data_ptr<float>(),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            w_zeros_ptr,
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K);
    } else {
        // v0: scalar reference (default)
        dim3 block(V0_BLOCK_N, V0_BLOCK_M);
        dim3 grid(
            (N + V0_BLOCK_N - 1) / V0_BLOCK_N,
            (M + V0_BLOCK_M - 1) / V0_BLOCK_M);
        mmq_q4_gemm_kernel_v0<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
            w_packed.data_ptr<int>(),
            reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            M, N, K);
    }
}
