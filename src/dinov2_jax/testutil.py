"""Helpers for parity-testing against the torch reference implementation.

Imports torch, so this module is only usable with the test dependency group
installed; the package itself never imports it.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import jax
import numpy as np
import torch


def to_jax(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return jax.numpy.asarray(x, dtype=dtype)


def extract_pytorch_outputs_and_inputs(
    model: torch.nn.Module, *args, **kwargs
) -> tuple[Any, OrderedDict[str, np.ndarray], OrderedDict[str, np.ndarray]]:
    """Run the model, capturing every submodule's (first) input and output."""
    outputs = OrderedDict()
    inputs = OrderedDict()
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach().cpu().numpy()
            if isinstance(input, tuple) and len(input) >= 1 and isinstance(input[0], torch.Tensor):
                inputs[name] = input[0].detach().cpu().numpy()
        return hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        out = model(*args, **kwargs)

    for hook in hooks:
        hook.remove()

    return out, outputs, inputs


__all__ = ["extract_pytorch_outputs_and_inputs", "to_jax"]
