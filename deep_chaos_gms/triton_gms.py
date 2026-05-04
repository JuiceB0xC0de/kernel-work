"""Phase 2 Triton dense-matmul kernel.

The wrapper in `gms.py` does the gather (`index_select`) and scatter
(`Tensor.scatter`) in pure PyTorch so autograd handles their backward
correctly.  This file only provides the inner small-dense-matmul op.

A `torch.autograd.Function` wraps two Triton kernels (forward GEMM, backward
GEMMs for d_x and d_W).  Bias gradient is computed in plain PyTorch (small
reduction, not worth a kernel).

If Triton isn't available or the device isn't CUDA/ROCm, the public
`triton_dense_linear(x, W, b)` callable raises `ImportError` / `RuntimeError`
and the wrapper in gms.py falls back to `F.linear`.

This file is a Phase 2 placeholder: the kernel implementations are stubbed
out and the public callable currently always raises.  Phase 1 lands first
and gets verified end-to-end before we wire the real Triton kernel in here.
"""

from __future__ import annotations

import torch


_PHASE_2_NOT_IMPLEMENTED = (
    "deep_chaos_gms.triton_gms: Triton kernel is a Phase 2 deliverable and "
    "is not implemented yet. The wrapper in gms.py will fall back to "
    "F.linear for the inner matmul. Set backend='torch' to silence this."
)


def triton_dense_linear(
    x: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor | None,
) -> torch.Tensor:
    """Small dense matmul `y = x @ W.T + b` via a Triton kernel.

    Phase 2: not yet implemented; raises so the caller can fall back.
    """
    raise NotImplementedError(_PHASE_2_NOT_IMPLEMENTED)
