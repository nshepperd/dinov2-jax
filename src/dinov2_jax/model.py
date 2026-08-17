"""DINOv2 vision transformer in JAX/Equinox.

A faithful port of `dinov2.models.vision_transformer.DinoVisionTransformer`
(inference paths only -- no drop path / dropout). Module and field names track
the torch implementation so that a Param's tree path is exactly its key in the
official state dicts.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from . import equinox_utils as eu
from .config import DinoV2Config
from .linear import LayerNorm, Linear
from .param import Param
from .resample import resize_bicubic_2d


class PatchEmbed(eqx.Module):
    """2D image to patch embedding: (B, C, H, W) -> (B, N, D).

    Torch uses a Conv2d whose stride equals its kernel, which is just a linear
    map over non-overlapping patches; the weight keeps Conv2d's layout so it
    loads verbatim from the checkpoint.
    """

    proj: PatchEmbedProj
    patch_size: int = eqx.field(static=True)

    def __init__(self, config: DinoV2Config):
        self.patch_size = config.patch_size
        self.proj = PatchEmbedProj(config.in_chans, config.embed_dim, config.patch_size)

    def __call__(
        self, x: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch patches embed_dim"]:
        _, _, H, W = x.shape
        p = self.patch_size
        assert H % p == 0, f"Input image height {H} is not a multiple of patch size {p}"
        assert W % p == 0, f"Input image width {W} is not a multiple of patch size {p}"
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) c p1 p2", p1=p, p2=p)
        return self.proj(x)


class PatchEmbedProj(eqx.Module):
    """The Conv2d inside PatchEmbed, applied to pre-extracted patches."""

    weight: Param
    bias: Param
    in_chans: int = eqx.field(static=True)
    embed_dim: int = eqx.field(static=True)
    patch_size: int = eqx.field(static=True)

    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.weight = Param(embed_dim, in_chans, patch_size, patch_size)
        self.bias = Param(embed_dim)

    def init_weights(self, key: PRNGKeyArray):
        weight = self.weight.set(
            jax.random.truncated_normal(key, -2, 2, self.weight.shape) * 0.02
        )
        bias = self.bias.set(jnp.zeros(self.bias.shape))
        return eu.replace(self, weight=weight, bias=bias)

    def __call__(
        self, x: Float[Array, "batch patches channels p1 p2"]
    ) -> Float[Array, "batch patches embed_dim"]:
        y = jnp.einsum("bncpq,dcpq->bnd", x, self.weight().astype(x.dtype))
        return y + self.bias().astype(x.dtype)


class Attention(eqx.Module):
    """Standard multi-head self-attention with a fused qkv projection."""

    qkv: Linear
    proj: Linear
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, dim: int, num_heads: int, *, qkv_bias: bool = True, proj_bias: bool = True):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = Linear(dim, dim * 3, use_bias=qkv_bias)
        self.proj = Linear(dim, dim, use_bias=proj_bias)

    def init_weights(self, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        return eu.replace(self, qkv=self.qkv.init_weights(k1), proj=self.proj.init_weights(k2))

    def __call__(
        self, x: Float[Array, "batch seq dim"]
    ) -> Float[Array, "batch seq dim"]:
        q, k, v = rearrange(
            self.qkv(x), "b n (p h d) -> p b n h d", p=3, h=self.num_heads, d=self.head_dim
        )
        if x.dtype in (jnp.float16, jnp.bfloat16) and jax.default_backend() == "gpu":
            # cuDNN flash attention: fused and O(seq) memory, but half
            # precision only. Mathematically the same softmax(qk^T)v with
            # scale 1/sqrt(head_dim).
            out = jax.nn.dot_product_attention(q, k, v, implementation="cudnn")
        else:
            attn = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.head_dim**-0.5
            attn = jax.nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bkhd->bqhd", attn, v)
        return self.proj(rearrange(out, "b n h d -> b n (h d)"))


class LayerScale(eqx.Module):
    gamma: Param

    def __init__(self, dim: int, init_values: float = 1e-5):
        self.gamma = Param(dim, array=jnp.full((dim,), init_values))

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        return x * self.gamma().astype(x.dtype)


class Mlp(eqx.Module):
    """Two-layer MLP with exact (erf) GELU, as in torch's nn.GELU default."""

    fc1: Linear
    fc2: Linear

    def __init__(self, in_features: int, hidden_features: int, *, use_bias: bool = True):
        self.fc1 = Linear(in_features, hidden_features, use_bias=use_bias)
        self.fc2 = Linear(hidden_features, in_features, use_bias=use_bias)

    def init_weights(self, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        return eu.replace(self, fc1=self.fc1.init_weights(k1), fc2=self.fc2.init_weights(k2))

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class SwiGLUFFNFused(eqx.Module):
    """SwiGLU FFN with fused input projection (used by the giant models)."""

    w12: Linear
    w3: Linear

    def __init__(self, in_features: int, hidden_features: int, *, use_bias: bool = True):
        self.w12 = Linear(in_features, 2 * hidden_features, use_bias=use_bias)
        self.w3 = Linear(hidden_features, in_features, use_bias=use_bias)

    def init_weights(self, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        return eu.replace(self, w12=self.w12.init_weights(k1), w3=self.w3.init_weights(k2))

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        x1, x2 = jnp.split(self.w12(x), 2, axis=-1)
        return self.w3(jax.nn.silu(x1) * x2)


class Block(eqx.Module):
    """Pre-norm transformer block with optional layerscale."""

    norm1: LayerNorm
    attn: Attention
    ls1: LayerScale | None
    norm2: LayerNorm
    mlp: Mlp | SwiGLUFFNFused
    ls2: LayerScale | None

    def __init__(self, config: DinoV2Config):
        dim = config.embed_dim
        self.norm1 = LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, config.num_heads)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        if config.ffn_layer == "mlp":
            self.mlp = Mlp(dim, config.ffn_hidden_dim)
        else:
            self.mlp = SwiGLUFFNFused(dim, config.ffn_hidden_dim)
        if config.init_values:
            self.ls1 = LayerScale(dim, config.init_values)
            self.ls2 = LayerScale(dim, config.init_values)
        else:
            self.ls1 = None
            self.ls2 = None

    def init_weights(self, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        return eu.replace(
            self, attn=self.attn.init_weights(k1), mlp=self.mlp.init_weights(k2)
        )

    def __call__(self, x: Float[Array, "batch seq dim"]) -> Float[Array, "batch seq dim"]:
        attn_out = self.attn(self.norm1(x))
        x = x + (self.ls1(attn_out) if self.ls1 is not None else attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + (self.ls2(mlp_out) if self.ls2 is not None else mlp_out)
        return x


class DinoVisionTransformer(eqx.Module):
    """The DINOv2 backbone.

    Calling the model returns the final-norm CLS token features (what the
    torch model's `forward` returns with the identity head, and what KDD /
    dgm-eval consume). `forward_features` returns the full dict of outputs;
    `get_intermediate_layers` mirrors the torch API.
    """

    config: DinoV2Config = eqx.field(static=True)

    cls_token: Param
    pos_embed: Param
    mask_token: Param
    register_tokens: Param | None
    patch_embed: PatchEmbed
    blocks: tuple[Block, ...]
    norm: LayerNorm

    def __init__(self, config: DinoV2Config):
        self.config = config
        D = config.embed_dim
        self.cls_token = Param(1, 1, D)
        self.pos_embed = Param(1, config.num_patches + 1, D)
        self.mask_token = Param(1, D)
        self.register_tokens = (
            Param(1, config.num_register_tokens, D) if config.num_register_tokens else None
        )
        self.patch_embed = PatchEmbed(config)
        self.blocks = tuple(Block(config) for _ in range(config.depth))
        self.norm = LayerNorm(D, eps=1e-6)

    @property
    def embed_dim(self) -> int:
        return self.config.embed_dim

    @property
    def patch_size(self) -> int:
        return self.config.patch_size

    @property
    def num_register_tokens(self) -> int:
        return self.config.num_register_tokens

    def init_weights(self, key: PRNGKeyArray):
        keys = jax.random.split(key, len(self.blocks) + 4)
        changes = dict(
            cls_token=self.cls_token.set(
                jax.random.normal(keys[0], self.cls_token.shape) * 1e-6
            ),
            pos_embed=self.pos_embed.set(
                jax.random.truncated_normal(keys[1], -2, 2, self.pos_embed.shape) * 0.02
            ),
            mask_token=self.mask_token.set(jnp.zeros(self.mask_token.shape)),
            patch_embed=eu.replace(
                self.patch_embed, proj=self.patch_embed.proj.init_weights(keys[2])
            ),
            blocks=tuple(
                block.init_weights(k) for block, k in zip(self.blocks, keys[3:-1])
            ),
        )
        if self.register_tokens is not None:
            changes["register_tokens"] = self.register_tokens.set(
                jax.random.normal(keys[-1], self.register_tokens.shape) * 1e-6
            )
        return eu.replace(self, **changes)

    def interpolate_pos_encoding(
        self, npatch: int, h: int, w: int, dtype
    ) -> Float[Array, "1 npatch_plus_1 dim"]:
        """Resize the learned position embedding to an (h, w)-pixel input.

        Reproduces the torch implementation exactly, including the historical
        +`interpolate_offset` scale-factor kludge (plain checkpoints) and
        antialiasing (register checkpoints). `h`/`w` are static under jit, so
        this compiles to a couple of einsums against precomputed weights.
        """
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed().astype(dtype)
        pos_embed = self.pos_embed().astype(jnp.float32)
        class_pos_embed = pos_embed[:, :1]
        patch_pos_embed = pos_embed[:, 1:]
        dim = pos_embed.shape[-1]
        h0 = h // self.config.patch_size
        w0 = w // self.config.patch_size
        M = math.isqrt(N)
        assert N == M * M
        assert npatch == h0 * w0
        if self.config.interpolate_offset:
            # torch is handed scale_factor=(h0+offset)/M and uses its
            # reciprocal as the source-units-per-output-pixel scale.
            scales = (M / (h0 + self.config.interpolate_offset), M / (w0 + self.config.interpolate_offset))
        else:
            scales = None
        patch_pos_embed = resize_bicubic_2d(
            patch_pos_embed.reshape(M, M, dim),
            (h0, w0),
            scales=scales,
            antialias=self.config.interpolate_antialias,
        ).reshape(1, h0 * w0, dim)
        return jnp.concatenate([class_pos_embed, patch_pos_embed], axis=1).astype(dtype)

    def prepare_tokens_with_masks(
        self,
        x: Float[Array, "batch channels height width"],
        masks: Bool[Array, "batch patches"] | None = None,
    ) -> Float[Array, "batch tokens dim"]:
        B, _, h, w = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = jnp.where(masks[..., None], self.mask_token().astype(x.dtype), x)

        cls_token = jnp.broadcast_to(self.cls_token().astype(x.dtype), (B, 1, x.shape[-1]))
        x = jnp.concatenate([cls_token, x], axis=1)
        x = x + self.interpolate_pos_encoding(x.shape[1] - 1, h, w, x.dtype)

        if self.register_tokens is not None:
            registers = jnp.broadcast_to(
                self.register_tokens().astype(x.dtype),
                (B, self.config.num_register_tokens, x.shape[-1]),
            )
            x = jnp.concatenate([x[:, :1], registers, x[:, 1:]], axis=1)

        return x

    def forward_features(
        self,
        x: Float[Array, "batch channels height width"],
        masks: Bool[Array, "batch patches"] | None = None,
    ) -> dict:
        x = self.prepare_tokens_with_masks(x, masks)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        R = self.config.num_register_tokens
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : R + 1],
            "x_norm_patchtokens": x_norm[:, R + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def get_intermediate_layers(
        self,
        x: Float[Array, "batch channels height width"],
        n: int | Sequence[int] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ) -> tuple:
        """Hidden states after the last `n` blocks (or the listed block
        indices), as tuples of patch tokens, optionally paired with the class
        token, matching the torch API."""
        _, _, h, w = x.shape
        blocks_to_take = (
            range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
        )
        outputs = []
        x = self.prepare_tokens_with_masks(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in blocks_to_take:
                outputs.append(x)
        assert len(outputs) == len(blocks_to_take), (
            f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        )
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.config.num_register_tokens :] for out in outputs]
        if reshape:
            p = self.config.patch_size
            outputs = [
                rearrange(out, "b (h w) d -> b d h w", h=h // p, w=w // p)
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def __call__(
        self,
        x: Float[Array, "batch channels height width"],
        masks: Bool[Array, "batch patches"] | None = None,
    ) -> Float[Array, "batch dim"]:
        return self.forward_features(x, masks)["x_norm_clstoken"]


__all__ = [
    "Attention",
    "Block",
    "DinoVisionTransformer",
    "LayerScale",
    "Mlp",
    "PatchEmbed",
    "SwiGLUFFNFused",
]
