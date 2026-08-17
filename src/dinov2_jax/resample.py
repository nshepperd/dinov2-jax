"""Bicubic resampling that reproduces `torch.nn.functional.interpolate`.

DINOv2 interpolates its learned position embeddings with torch's bicubic
resize, in two historical flavours:

  - plain checkpoints: `antialias=False` with a `scale_factor` carrying a
    +0.1 offset kludge inherited from DINO,
  - register checkpoints: `antialias=True` with an exact output `size`.

jax.image.resize matches neither (different cubic kernel and edge handling),
and the feature values shift visibly if the position embeddings are off, so we
reproduce torch's coefficient computation here. Sizes are static under jit, so
the (out x in) weight matrices are precomputed in numpy and the resize itself
is two einsums.

Coefficient conventions, from aten/src/ATen/native/UpSample*:

  - antialias=False (`upsample_bicubic2d`): source coordinate
    `src = (dst + 0.5) * scale - 0.5` with `scale = 1/scale_factor` when a
    scale factor is given, else `in/out`. Four taps of the cubic convolution
    kernel with A = -0.75, indices clamped to the border (replicate padding).
  - antialias=True (`_upsample_bicubic2d_aa`): PIL-compatible resampling.
    Cubic kernel with A = -0.5, support widened by the scale when
    downsampling, weights normalized to sum to 1, taps clipped to the image.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def _cubic(x: np.ndarray, A: float) -> np.ndarray:
    """Cubic convolution kernel (Keys); torch uses A=-0.75, PIL A=-0.5."""
    x = np.abs(x)
    return np.where(
        x <= 1.0,
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0,
        np.where(x < 2.0, (((x - 5.0) * x + 8.0) * x - 4.0) * A, 0.0),
    )


def bicubic_weights(
    in_size: int,
    out_size: int,
    *,
    scale: float | None = None,
    antialias: bool = False,
) -> np.ndarray:
    """The (out_size, in_size) matrix W such that `out = W @ in` reproduces
    torch bicubic interpolation along one axis.

    `scale` is the *reciprocal* of torch's scale_factor argument, i.e.
    input units per output unit; None means `in_size / out_size` (torch's
    behaviour when only `size` is given).
    """
    if scale is None:
        scale = in_size / out_size
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    dst = np.arange(out_size, dtype=np.float64)

    if antialias and scale > 1.0:
        # Downsampling with antialiasing: stretch the kernel by the scale and
        # renormalize. Matches PIL and torch's `_upsample_bicubic2d_aa`.
        support = 2.0 * scale
        center = scale * (dst + 0.5)
        xmin = np.maximum(np.floor(center - support + 0.5).astype(np.int64), 0)
        xmax = np.minimum(np.floor(center + support + 0.5).astype(np.int64), in_size)
        for i in range(out_size):
            idx = np.arange(xmin[i], xmax[i])
            w = _cubic((idx + 0.5 - center[i]) / scale, A=-0.5)
            weights[i, idx] = w / w.sum()
    else:
        if antialias:
            # torch still uses the PIL kernel (A=-0.5, normalized) when
            # antialias=True even if it has no widening effect.
            src = scale * (dst + 0.5)
            xmin = np.maximum(np.floor(src - 2.0 + 0.5).astype(np.int64), 0)
            xmax = np.minimum(np.floor(src + 2.0 + 0.5).astype(np.int64), in_size)
            for i in range(out_size):
                idx = np.arange(xmin[i], xmax[i])
                w = _cubic(idx + 0.5 - src[i], A=-0.5)
                weights[i, idx] = w / w.sum()
        else:
            # Border-replicating 4-tap cubic, A=-0.75, unnormalized.
            src = (dst + 0.5) * scale - 0.5
            base = np.floor(src).astype(np.int64)
            for k in range(-1, 3):
                idx = np.clip(base + k, 0, in_size - 1)
                w = _cubic(src - (base + k), A=-0.75)
                np.add.at(weights, (np.arange(out_size), idx), w)
    return weights


def resize_bicubic_2d(
    x: Float[Array, "h w c"],
    out_hw: tuple[int, int],
    *,
    scales: tuple[float, float] | None = None,
    antialias: bool = False,
) -> Float[Array, "h2 w2 c"]:
    """Resize the two leading spatial axes of `x`, torch-bicubic-exactly.

    `scales` are input-units-per-output-unit for (h, w); None derives them
    from the sizes (torch's `size=` mode).
    """
    h, w, _ = x.shape
    (h2, w2) = out_hw
    wh = bicubic_weights(h, h2, scale=None if scales is None else scales[0], antialias=antialias)
    ww = bicubic_weights(w, w2, scale=None if scales is None else scales[1], antialias=antialias)
    wh = jnp.asarray(wh, dtype=jnp.float32)
    ww = jnp.asarray(ww, dtype=jnp.float32)
    return jnp.einsum("am,bn,mnc->abc", wh, ww, x.astype(jnp.float32))


__all__ = ["bicubic_weights", "resize_bicubic_2d"]
