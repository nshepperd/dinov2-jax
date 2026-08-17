"""DINOv2 vision transformers in JAX/Equinox, weight-compatible with the
official Meta checkpoints."""
from __future__ import annotations

from .config import DINOV2_CONFIGS, DinoV2Config
from .equinox_utils import cast_floating, mapmod, mapmod_with_path, replace
from .linear import LayerNorm, Linear
from .loading import load_dinov2, load_state_dict
from .model import (
    Attention,
    Block,
    DinoVisionTransformer,
    LayerScale,
    Mlp,
    PatchEmbed,
    SwiGLUFFNFused,
)
from .param import AbstractParam, Param, params_with_paths, path_to_key
from .transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    classification_transform,
    dgm_eval_clean_transform,
    dgm_eval_transform,
)

__all__ = [
    "AbstractParam",
    "Attention",
    "Block",
    "DINOV2_CONFIGS",
    "DinoV2Config",
    "DinoVisionTransformer",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LayerNorm",
    "LayerScale",
    "Linear",
    "Mlp",
    "Param",
    "PatchEmbed",
    "SwiGLUFFNFused",
    "cast_floating",
    "classification_transform",
    "dgm_eval_clean_transform",
    "dgm_eval_transform",
    "load_dinov2",
    "load_state_dict",
    "mapmod",
    "mapmod_with_path",
    "params_with_paths",
    "path_to_key",
    "replace",
]
