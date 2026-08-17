"""Checkpoint download, conversion and loading.

The official checkpoints are torch-pickled `.pth` files. On first use we
download one, unpickle it with torch (the only step that needs torch -- install
the `convert` extra), and cache it as fp32 safetensors next to it; every later
load reads the safetensors directly.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import safetensors.numpy as st
from tqdm import tqdm

from . import equinox_utils as eu
from .config import DINOV2_CONFIGS, DinoV2Config
from .model import DinoVisionTransformer
from .param import AbstractParam

T = TypeVar("T", bound=eqx.Module)

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def checkpoint_url(name: str) -> str:
    """URL of the official pretrained backbone for a torch.hub model name,
    e.g. `dinov2_vitl14` or `dinov2_vits14_reg`."""
    config = DINOV2_CONFIGS[name]
    base_name = name.removesuffix("_reg")
    full_name = base_name + (f"_reg{config.num_register_tokens}" if name.endswith("_reg") else "")
    return f"{_DINOV2_BASE_URL}/{base_name}/{full_name}_pretrain.pth"


def default_cache_dir() -> Path:
    if "DINOV2_JAX_CACHE" in os.environ:
        return Path(os.environ["DINOV2_JAX_CACHE"])
    return Path("~/.cache/dinov2-jax").expanduser()


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length", 0))
        with (
            tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as bar,
            open(tmp, "wb") as f,
        ):
            while chunk := response.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def convert_pth_to_safetensors(pth_path: Path, out_path: Path):
    """Unpickle a torch checkpoint and rewrite it as fp32 safetensors."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"Converting {pth_path} requires torch to unpickle it. Install the "
            "convert extra (`uv pip install 'dinov2-jax[convert]'`) or place a "
            "pre-converted .safetensors file at the same path."
        ) from None
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    arrays = {k: v.float().numpy() for k, v in state_dict.items()}
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    st.save_file(arrays, tmp)
    tmp.rename(out_path)


def load_state_dict(module: T, state_dict: dict[str, jax.Array]) -> T:
    """Fill every Param in `module` from `state_dict`.

    A Param's path in the tree is its key in the state dict, so no module needs
    a load method of its own -- containers are traversed, Params are loaded.
    Keys are consumed as they are used, so whatever remains in `state_dict`
    afterwards is exactly what went unclaimed.
    """

    def visit(path: jax.tree_util.KeyPath, module: eqx.Module) -> eqx.Module:
        if isinstance(module, AbstractParam):
            return module.load_state_dict(state_dict, path)
        return module

    return eu.mapmod_with_path(visit, module)


def load_dinov2(
    name_or_path: str | Path,
    *,
    config: DinoV2Config | None = None,
    dtype: jnp.dtype | None = None,
    cache_dir: str | Path | None = None,
    error_on_unused: bool = True,
) -> DinoVisionTransformer:
    """Load a pretrained DINOv2 backbone.

    `name_or_path` is either a torch.hub model name (`dinov2_vits14`,
    `dinov2_vitb14`, `dinov2_vitl14`, `dinov2_vitg14`, or their `_reg`
    variants), downloaded and cached on first use, or a local `.pth` /
    `.safetensors` checkpoint (which requires `config`).

    `dtype` casts the loaded parameters (default: keep fp32).
    """
    name_or_path = str(name_or_path)
    if name_or_path in DINOV2_CONFIGS:
        name = name_or_path
        config = config or DINOV2_CONFIGS[name]
        cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        pth_path = cache / f"{name}_pretrain.pth"
        safetensors_path = pth_path.with_suffix(".safetensors")
        if not safetensors_path.exists():
            if not pth_path.exists():
                _download(checkpoint_url(name), pth_path)
            convert_pth_to_safetensors(pth_path, safetensors_path)
    else:
        if config is None:
            raise ValueError(
                "Loading from a local checkpoint path requires passing `config` "
                "(e.g. dinov2_jax.DINOV2_CONFIGS['dinov2_vitl14'])."
            )
        path = Path(name_or_path)
        if path.suffix == ".safetensors":
            safetensors_path = path
        else:
            safetensors_path = path.with_suffix(".safetensors")
            if not safetensors_path.exists():
                convert_pth_to_safetensors(path, safetensors_path)

    state_dict = {}
    for key, value in st.load_file(safetensors_path).items():
        array = jnp.asarray(value)
        if dtype is not None and np.issubdtype(value.dtype, np.floating):
            array = array.astype(dtype)
        state_dict[key] = array

    model = DinoVisionTransformer(config)
    model = load_state_dict(model, state_dict)
    if error_on_unused:
        assert len(state_dict) == 0, f"Unused keys in state_dict: {list(state_dict.keys())}"
    return model


__all__ = [
    "checkpoint_url",
    "convert_pth_to_safetensors",
    "default_cache_dir",
    "load_dinov2",
    "load_state_dict",
]
