from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import TypeVar

import equinox as eqx
import jax

T = TypeVar("T", bound=eqx.Module)


def replace(module: T, **changes) -> T:
    """Basically just dataclasses.replace but bypasses __init__."""
    T = type(module)
    ret = object.__new__(T)
    for field in dataclasses.fields(T):
        if field.name in changes:
            object.__setattr__(ret, field.name, changes[field.name])
        else:
            object.__setattr__(ret, field.name, getattr(module, field.name))
    field_names = {field.name for field in dataclasses.fields(T)}
    if any(k not in field_names for k in changes):
        raise ValueError(f"Invalid field names in changes: {set(changes.keys()) - field_names}")
    return ret


def mapmod(
    fn: Callable[[eqx.Module], eqx.Module],
    module: T
) -> T:
    """Applies `fn` to every submodule in `module` and then to `module` itself.

    Bottom-up recursion, like a catamorphism.
    """
    def proc(module):
        if not isinstance(module, eqx.Module):
            return module
        module = jax.tree_util.tree_map(proc, module, is_leaf=lambda m: isinstance(m, eqx.Module) and m is not module)
        return fn(module)
    return proc(module)


def mapmod_with_path(
    fn: Callable[[jax.tree_util.KeyPath, eqx.Module], eqx.Module], module: T
) -> T:
    """Applies `fn` to every submodule in `module` and then to `module` itself.

    Bottom-up recursion, like a catamorphism.
    """

    def proc(prefix, path, module):
        if not isinstance(module, eqx.Module):
            return module
        path = prefix + path
        module = jax.tree_util.tree_map_with_path(
            partial(proc, path),
            module,
            is_leaf=lambda m: isinstance(m, eqx.Module) and m is not module,
        )
        return fn(path, module)

    return proc((), (), module)


__all__ = ["mapmod", "mapmod_with_path", "replace"]
