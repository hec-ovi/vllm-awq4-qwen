import torch  # MUST come before `from . import _C` so libc10.so / libtorch.so are dlopen'd first

from . import _C  # noqa: F401


def mmq_q4_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    version: int = 0,
    w_zeros: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    AWQ-INT4 MMQ kernel for gfx1151.

    version: 0 = v0 scalar reference (always-correct, slow)
             1 = v1 WMMA + LDS staging
    w_zeros: optional (N, K/32) int8 per-group zero points (asymmetric quant);
             pass None for symmetric uint4b8 (kernel uses zero=8 baseline).
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.awq_mmq_gfx1151.mmq_q4_gemm(x, w_packed, scales, w_zeros, version)
