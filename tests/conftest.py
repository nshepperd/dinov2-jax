from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
# On Ampere+ GPUs JAX's default fp32 matmul precision is TF32, which costs
# ~1e-2 of drift over a full forward pass. Parity is judged at true fp32.
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
# The torch reference runs on CPU: deterministic fp32 with no fused-kernel
# variance, so any disagreement is ours.
os.environ.setdefault("CUDA_VISIBLE_DEVICES_TORCH", "")
