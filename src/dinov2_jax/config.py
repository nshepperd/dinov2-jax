"""Pydantic configuration for DINOv2 vision transformers."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict


class DinoV2Config(BaseModel):
    """Configuration for a DINOv2 DinoVisionTransformer.

    Defaults follow the released LVD-142M checkpoints (patch 14, trained at
    518x518, layerscale enabled). `img_size` only fixes the size of the learned
    position embedding grid; inference at other resolutions interpolates it.
    """

    model_config = ConfigDict(frozen=True)

    img_size: int = 518
    patch_size: int = 14
    in_chans: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    ffn_layer: Literal["mlp", "swiglufused"] = "mlp"
    # layerscale init; None disables layerscale. All released DINOv2
    # checkpoints use 1.0.
    init_values: float | None = 1.0
    num_register_tokens: int = 0
    # Position embedding interpolation details. The plain checkpoints keep the
    # historical +0.1 scale-factor kludge from DINO; the register variants
    # dropped it and enabled antialiasing instead. Both must be reproduced
    # exactly for output parity.
    interpolate_antialias: bool = False
    interpolate_offset: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2

    @property
    def pos_embed_grid_size(self) -> int:
        return math.isqrt(self.num_patches)

    @property
    def ffn_hidden_dim(self) -> int:
        hidden = int(self.embed_dim * self.mlp_ratio)
        if self.ffn_layer == "swiglufused":
            # SwiGLUFFNFused shrinks the hidden dim by 2/3 (two input
            # projections for the price of one) and rounds up to a multiple
            # of 8.
            hidden = (int(hidden * 2 / 3) + 7) // 8 * 8
        return hidden


def _plain(**kwargs) -> DinoV2Config:
    return DinoV2Config(**kwargs)


def _reg(**kwargs) -> DinoV2Config:
    return DinoV2Config(
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


_VITS = dict(embed_dim=384, depth=12, num_heads=6)
_VITB = dict(embed_dim=768, depth=12, num_heads=12)
_VITL = dict(embed_dim=1024, depth=24, num_heads=16)
_VITG = dict(embed_dim=1536, depth=40, num_heads=24, ffn_layer="swiglufused")

# One config per torch.hub entry point in facebookresearch/dinov2.
DINOV2_CONFIGS: dict[str, DinoV2Config] = {
    "dinov2_vits14": _plain(**_VITS),
    "dinov2_vitb14": _plain(**_VITB),
    "dinov2_vitl14": _plain(**_VITL),
    "dinov2_vitg14": _plain(**_VITG),
    "dinov2_vits14_reg": _reg(**_VITS),
    "dinov2_vitb14_reg": _reg(**_VITB),
    "dinov2_vitl14_reg": _reg(**_VITL),
    "dinov2_vitg14_reg": _reg(**_VITG),
}


__all__ = ["DINOV2_CONFIGS", "DinoV2Config"]
