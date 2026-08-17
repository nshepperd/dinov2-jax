"""Linear and normalization layers."""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from . import equinox_utils as eu
from .param import Param


class Linear(eqx.Module):
    weight: Param
    bias: Param | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    def __init__(self, in_features: int, out_features: int, *, use_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Param(out_features, in_features)
        self.bias = Param(out_features) if use_bias else None

    @property
    def use_bias(self) -> bool:
        return self.bias is not None

    def init_weights(self, key: PRNGKeyArray):
        weight = self.weight.set(
            jax.random.truncated_normal(key, -2, 2, self.weight.shape) * 0.02
        )
        bias = self.bias.set(jnp.zeros(self.bias.shape)) if self.bias is not None else None
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(self, x: Array):
        y = x @ self.weight().T.astype(x.dtype)
        if self.bias is not None:
            y += self.bias().astype(x.dtype)
        return y


class LayerNorm(eqx.Module):
    weight: Param
    bias: Param
    eps: float = eqx.field(static=True)

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = Param(hidden_size, array=jnp.ones((hidden_size,)))
        self.bias = Param(hidden_size, array=jnp.zeros((hidden_size,)))
        self.eps = eps

    def __call__(self, hidden_states: Array):
        dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        mean = hidden_states.mean(-1, keepdims=True)
        variance = jnp.var(hidden_states, axis=-1, keepdims=True)
        hidden_states = (hidden_states - mean) * jax.lax.rsqrt(variance + self.eps)
        return (self.weight() * hidden_states + self.bias()).astype(dtype)


__all__ = ["LayerNorm", "Linear"]
