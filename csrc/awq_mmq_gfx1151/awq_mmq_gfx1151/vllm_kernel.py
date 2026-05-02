"""
vLLM MPLinearKernel adapter for the AWQ-INT4 MMQ HIP custom op (gfx1151).

This module subclasses vllm's MPLinearKernel and routes apply_weights through
torch.ops.awq_mmq_gfx1151.mmq_q4_gemm. Registration into the dispatcher
(_POSSIBLE_KERNELS[ROCM]) is done by the patch_strix.py Patch 16, NOT here.

Tensor contract verified against vllm v0.20.0 compressed_tensors_wNa16:
  weight_packed:     [N, K//8]  int32  (8 uint4b8 per int32, low nibble first)
  weight_scale:      [N, K//G]  fp16
  zero_points:       absent for symmetric uint4b8
  g_idx:             absent (no act reordering supported)

Our kernel uses these tensors AS-IS — no repack at process_weights_after_loading
(unlike TritonW4A16 which transposes both). This is a load-time win.
"""
import torch

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
    MPLinearKernel,
    MPLinearLayerConfig,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

# Tile constants from the HIP kernel — kept in sync manually with awq_mmq_gfx1151_kernel.hip.
MMQ_X = 64   # N tile, must divide MMQ_X for full-block efficiency
MMQ_Y = 64   # M tile (handled with bounds checks for tails)
GROUP_SIZE = 32
SUPPORTED_QUANT_TYPES = [
    scalar_types.uint4b8,  # symmetric: zero point is implicit (=8)
    scalar_types.uint4,    # asymmetric: explicit per-group zero points
]


class RocmMmqQ4LinearKernel(MPLinearKernel):
    """WMMA i32 16x16x16 iu8 kernel for AWQ-INT4 W4A16 g32 sym on gfx1151."""

    SUPPORTED_QUANT_TYPES = SUPPORTED_QUANT_TYPES

    @classmethod
    def get_min_capability(cls) -> int:
        return 0  # gfx1151 capability check happens via on_gfx1x() in can_implement

    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        # Verbose debug: log every can_implement call with the full config.
        import logging
        _log = logging.getLogger(__name__)
        _log.warning(
            "RocmMmqQ4.can_implement called: full=%s partition=%s wt=%s act=%s g=%d zp=%s gidx=%s",
            c.full_weight_shape, c.partition_weight_shape, c.weight_type,
            c.act_type, c.group_size, c.zero_points, c.has_g_idx,
        )
        result = cls._can_implement_inner(c)
        _log.warning("RocmMmqQ4.can_implement -> %s", result)
        return result

    @classmethod
    def _can_implement_inner(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "RocmMmqQ4 targets ROCm only"

        try:
            from vllm.platforms.rocm import on_gfx1x
        except ImportError:
            return False, "vllm.platforms.rocm.on_gfx1x not available"
        if not on_gfx1x():
            return False, "RocmMmqQ4 targets gfx1151 (gfx1x) only"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"weight_type {c.weight_type} not supported; "
                f"only uint4b8 (symmetric AWQ-INT4)",
            )

        # bf16 accepted via inline cast in apply_weights. Native bf16 in the
        # kernel is a v1.2 task (act_quant kernel currently fp16-only).
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, f"only fp16/bf16 activations supported (got {c.act_type})"

        if c.group_size != GROUP_SIZE:
            return (
                False,
                f"group_size={c.group_size} not supported (only {GROUP_SIZE})",
            )

        # Asymmetric quant (zero_points=True) supported via per-group zp tensor.
        if c.has_g_idx:
            return False, "activation reordering (g_idx) not supported"

        K = c.partition_weight_shape[0]
        N = c.partition_weight_shape[1]
        if K % GROUP_SIZE != 0:
            return False, f"K={K} not divisible by group_size={GROUP_SIZE}"
        if N < MMQ_X:
            return False, f"N={N} smaller than MMQ_X tile ({MMQ_X})"
        # M is variable per call; tail handling in the kernel covers any M >= 1.

        # Verify the .so is importable (not just present on disk).
        try:
            import awq_mmq_gfx1151  # noqa: F401
        except ImportError as e:
            return False, f"awq_mmq_gfx1151 module not importable: {e}"

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """
        Dual-storage layout (decode/prefill dispatch):
        - Our kernel reads w_q AS-IS [N, K//8] int32, w_s [N, K//G] fp16,
          and w_zp PACKED [N//8, K//G] int32 (kernel does inline unpack).
        - TritonW4A16 fallback (small-M decode) needs transposed format:
          w_q_triton [K, N//8] int32, w_s_triton [K//G, N] fp16, w_zp_triton [K//G, N//8].
          Stored under `_awq_mmq_triton_*` attrs.
        Memory cost: ~1x extra weight (transposed copy of w_q). Original w_q
        retained for our kernel since the layouts differ.
        """
        from vllm.model_executor.layers.quantization.utils import replace_parameter

        w_q, w_s, w_zp, _ = self._get_weight_params(layer)

        if not w_q.is_contiguous():
            replace_parameter(layer, self.w_q_name,
                              torch.nn.Parameter(w_q.contiguous(), requires_grad=False))
        # Keep w_s in its native dtype (bf16 for Qwen 3.6) so the Triton fallback
        # path doesn't dtype-mismatch at compile time. Cast to fp16 inline in
        # apply_weights only when routing to our kernel.
        if not w_s.is_contiguous():
            replace_parameter(layer, self.w_s_name,
                              torch.nn.Parameter(w_s.contiguous(), requires_grad=False))
        # w_zp stays in PACKED [N//8, K//G] int32 format (no unpack — done in kernel).
        # Pre-compute fp16 cast of scales for our kernel path. Stored as tensor (not
        # nn.Parameter) so it doesn't pollute the layer's state_dict.
        layer._awq_mmq_w_s_fp16 = (
            getattr(layer, self.w_s_name).data.to(torch.float16).contiguous()
            if w_s.dtype != torch.float16 else getattr(layer, self.w_s_name).data
        )

        # ---- TritonW4A16 fallback format ----
        w_q_now = getattr(layer, self.w_q_name).data
        N_dim, K8 = w_q_now.shape
        K_dim = K8 * 8
        shifts = torch.arange(8, device=w_q_now.device, dtype=torch.int32) * 4
        w_unpacked = ((w_q_now.unsqueeze(-1) >> shifts) & 0xF).reshape(N_dim, K_dim)
        w_KN = w_unpacked.t().contiguous()
        N8 = N_dim // 8
        w_repacked = torch.sum((w_KN.view(K_dim, N8, 8) & 0xF) << shifts, dim=2, dtype=torch.int32)
        layer._awq_mmq_triton_w_q = w_repacked.contiguous()
        del w_unpacked, w_KN, w_repacked  # free intermediate buffers

        layer._awq_mmq_triton_w_s = getattr(layer, self.w_s_name).data.t().contiguous()
        layer._awq_mmq_triton_w_zp = w_zp.t().contiguous() if w_zp is not None else None

    # Below this M threshold, route to TritonW4A16 fallback (decode-shape).
    # Tuned to match the ~17 t/s decode floor: DFlash with N=8 spec tokens
    # gives M=8 typical, plus warmup/probe at M=1.
    SMALL_M_THRESHOLD = 32

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        M = x_2d.size(0)

        out_shape = x.shape[:-1] + (c.partition_weight_shape[1],)

        if M < self.SMALL_M_THRESHOLD:
            # Decode shape: route through TritonW4A16's fused dequant+matmul.
            from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
                triton_w4a16_gemm,
            )
            zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
            out = triton_w4a16_gemm(
                a=x_2d,
                b_q=layer._awq_mmq_triton_w_q,
                scales=layer._awq_mmq_triton_w_s,
                qzeros=layer._awq_mmq_triton_w_zp,
                group_size=c.group_size if c.group_size != -1 else c.partition_weight_shape[0],
                zp_bias=zp_bias,
            )
        else:
            # Prefill shape: route through our HIP MMQ Q4 kernel.
            w_q, _w_s_native, w_zp, _ = self._get_weight_params(layer)
            w_s_fp16 = layer._awq_mmq_w_s_fp16  # pre-cast at process_weights time
            orig_dtype = x_2d.dtype
            if x_2d.dtype != torch.float16:
                x_2d = x_2d.to(torch.float16)
            if w_zp is None:
                zp_in = torch.empty(0, dtype=torch.int32, device=x.device)
            else:
                # w_zp is PACKED [N//8, K//G] int32; kernel unpacks inline.
                zp_in = w_zp
            out = torch.ops.awq_mmq_gfx1151.mmq_q4_gemm(x_2d, w_q, w_s_fp16, zp_in, 1)
            if orig_dtype != torch.float16:
                out = out.to(orig_dtype)

        if bias is not None:
            out = out + bias

        return out.reshape(out_shape)
