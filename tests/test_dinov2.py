"""Parity tests against the official facebookresearch/dinov2 implementation.

Every test compares this package's outputs to the torch reference running the
released checkpoints in fp32 on CPU. Which checkpoints get tested is
controlled by DINOV2_JAX_TEST_MODELS (comma-separated torch.hub names);
the default covers the plain and register variants plus the KDD-standard
ViT-L/14.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from PIL import Image

from dinov2_jax import load_dinov2
from dinov2_jax.equinox_utils import mapmod_with_path
from dinov2_jax.model import DinoVisionTransformer
from dinov2_jax.param import params_with_paths, path_to_key
from dinov2_jax.resample import bicubic_weights
from dinov2_jax.testutil import extract_pytorch_outputs_and_inputs, to_jax
from dinov2_jax.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    classification_transform,
    dgm_eval_clean_transform,
    dgm_eval_transform,
)

TEST_MODELS = os.environ.get(
    "DINOV2_JAX_TEST_MODELS", "dinov2_vits14,dinov2_vits14_reg,dinov2_vitl14"
).split(",")

# fp32 end-to-end tolerance. Features are O(1)-O(10); 24 transformer blocks of
# reordered fp32 reductions accumulate ~1e-4 of absolute drift.
RTOL = 1e-3
ATOL = 2e-3


@pytest.fixture(scope="module", params=TEST_MODELS)
def model_name(request) -> str:
    return request.param


@pytest.fixture(scope="module")
def torch_model(model_name) -> torch.nn.Module:
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    return model.eval().float()


@pytest.fixture(scope="module")
def jax_model(model_name) -> DinoVisionTransformer:
    return load_dinov2(model_name)


@jax.jit
def jax_forward_features(model, x, masks=None):
    return model.forward_features(x, masks)


def random_images(batch: int, h: int, w: int, seed: int = 0) -> np.ndarray:
    """Normalized image batch, (B, 3, h, w) float32, deterministic."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 256, size=(batch, 3, h, w)).astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def assert_features_close(jax_val, torch_val, err_msg=""):
    jax_val = np.asarray(jax_val)
    torch_val = torch_val.detach().cpu().numpy()
    assert jax_val.shape == torch_val.shape, (
        f"{err_msg}: shape {jax_val.shape} vs {torch_val.shape}"
    )
    np.testing.assert_allclose(jax_val, torch_val, rtol=RTOL, atol=ATOL, err_msg=err_msg)


# =============================================================================
# Position embedding resampling (unit tests, torch is the oracle)
# =============================================================================


@pytest.mark.parametrize("antialias", [False, True])
@pytest.mark.parametrize(
    "in_size,out_size,offset",
    [
        (37, 16, 0.1),   # ViT-L/14 grid at 224x224, plain kludge
        (37, 16, 0.0),   # register-variant size mode
        (37, 40, 0.1),   # upsampling
        (37, 40, 0.0),
        (37, 12, 0.1),
        (37, 37, 0.0),   # identity size
        (16, 27, 0.1),   # non-square path uses different scales per axis
    ],
)
def test_bicubic_matches_torch(in_size, out_size, offset, antialias):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1, 5, in_size, in_size)).astype(np.float32)

    kwargs = {}
    if offset:
        scale_factor = (out_size + offset) / in_size
        kwargs["scale_factor"] = (scale_factor, scale_factor)
        scale = in_size / (out_size + offset)
    else:
        kwargs["size"] = (out_size, out_size)
        scale = None

    ref = torch.nn.functional.interpolate(
        torch.from_numpy(x), mode="bicubic", antialias=antialias, **kwargs
    ).numpy()
    assert ref.shape[-2:] == (out_size, out_size)

    w = bicubic_weights(in_size, out_size, scale=scale, antialias=antialias)
    out = np.einsum("am,bn,fcmn->fcab", w, w, x.astype(np.float64))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)


# =============================================================================
# Weight loading
# =============================================================================


def test_loaded_params(torch_model, jax_model):
    torch_params = torch_model.state_dict()

    # Every Param is named by its tree path, which is its state dict key. Check
    # the correspondence in both directions: comparing only where the names
    # happen to line up would pass just as well if nothing lined up at all.
    checked = set()
    for path, param in params_with_paths(jax_model):
        key = path_to_key(path)
        assert key in torch_params, f"Loaded param has no counterpart in torch: {key}"
        np.testing.assert_array_equal(
            np.asarray(param()), torch_params[key].numpy(), err_msg=key
        )
        checked.add(key)

    unchecked = set(torch_params) - checked
    assert not unchecked, f"torch weights not present in the JAX model: {sorted(unchecked)}"


# =============================================================================
# End-to-end forward parity
# =============================================================================


@pytest.mark.parametrize(
    "h,w",
    [
        (224, 224),  # the KDD / dgm-eval resolution; exercises interpolation
        (518, 518),  # native training resolution; no interpolation
        (168, 294),  # non-square, distinct scales per axis
    ],
)
def test_forward_features(torch_model, jax_model, h, w):
    x = random_images(2, h, w)
    with torch.no_grad():
        ref = torch_model.forward_features(torch.from_numpy(x))
    out = jax_forward_features(jax_model, jnp.asarray(x))
    for key in ["x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm"]:
        assert_features_close(out[key], ref[key], err_msg=key)


def test_forward_cls(torch_model, jax_model):
    """model(x) is the CLS feature vector used for KDD."""
    x = random_images(2, 224, 224, seed=1)
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(x))
    out = jax.jit(type(jax_model).__call__)(jax_model, jnp.asarray(x))
    assert out.shape == (2, jax_model.embed_dim)
    assert_features_close(out, ref, err_msg="cls features")


def test_forward_masks(torch_model, jax_model):
    x = random_images(2, 224, 224, seed=2)
    n_patches = (224 // 14) ** 2
    rng = np.random.default_rng(3)
    masks = rng.random((2, n_patches)) < 0.3
    with torch.no_grad():
        ref = torch_model.forward_features(
            torch.from_numpy(x), masks=torch.from_numpy(masks)
        )
    out = jax_forward_features(jax_model, jnp.asarray(x), masks=jnp.asarray(masks))
    for key in ["x_norm_clstoken", "x_norm_patchtokens"]:
        assert_features_close(out[key], ref[key], err_msg=key)


@pytest.mark.parametrize("n", [4, (2, 5)])
def test_get_intermediate_layers(torch_model, jax_model, n):
    x = random_images(2, 224, 224, seed=4)
    with torch.no_grad():
        ref = torch_model.get_intermediate_layers(
            torch.from_numpy(x), n=n, reshape=True, return_class_token=True, norm=True
        )
    out = jax_model.get_intermediate_layers(
        jnp.asarray(x), n=n, reshape=True, return_class_token=True, norm=True
    )
    assert len(out) == len(ref)
    for i, ((jax_patch, jax_cls), (ref_patch, ref_cls)) in enumerate(zip(out, ref)):
        assert_features_close(jax_patch, ref_patch, err_msg=f"layer {i} patch tokens")
        assert_features_close(jax_cls, ref_cls, err_msg=f"layer {i} cls token")


def test_get_intermediate_layers_unnormed(torch_model, jax_model):
    x = random_images(1, 224, 224, seed=5)
    with torch.no_grad():
        (ref,) = torch_model.get_intermediate_layers(torch.from_numpy(x), n=1, norm=False)
    (out,) = jax_model.get_intermediate_layers(jnp.asarray(x), n=1, norm=False)
    assert_features_close(out, ref, err_msg="unnormed last layer")


# =============================================================================
# Layerwise parity: feed each torch layer's recorded input into the
# corresponding JAX module and compare outputs. Catches per-layer bugs even
# where accumulated drift would stay within end-to-end tolerance.
# =============================================================================


def test_layerwise(torch_model, jax_model):
    x = random_images(2, 224, 224, seed=6)
    _, layer_outputs, layer_inputs = extract_pytorch_outputs_and_inputs(
        torch_model, torch.from_numpy(x)
    )

    jax_layers = {}

    def visit(path, module):
        jax_layers[path_to_key(path)] = module
        return module

    mapmod_with_path(visit, jax_model)

    compared = 0
    mismatched = []
    for name, torch_in in layer_inputs.items():
        if name not in layer_outputs or name not in jax_layers:
            continue
        jax_module = jax_layers[name]
        if not callable(jax_module):
            continue
        try:
            jax_out = np.asarray(jax_module(to_jax(torch_in)))
        except Exception:
            # Some modules take different input layouts (e.g. patch_embed.proj
            # sees pre-extracted patches here, a raw conv input in torch).
            continue
        torch_out = layer_outputs[name]
        if jax_out.shape != torch_out.shape:
            continue
        compared += 1
        close = np.isclose(jax_out, torch_out, rtol=1e-3, atol=1e-4)
        mismatch_pct = 100 * (1 - close.mean())
        max_diff = float(np.max(np.abs(jax_out - torch_out)))
        print(f"<{name}> max_diff={max_diff:.2e} mismatch={mismatch_pct:.2f}%")
        if mismatch_pct > 1.0:
            mismatched.append(name)

    # blocks, norms, attn, mlp, ls at minimum
    assert compared >= 6 * len(jax_model.blocks)
    assert not mismatched, f"Mismatched layers: {mismatched}"


# =============================================================================
# Preprocessing parity and the full KDD feature pipeline
# =============================================================================


def _test_image(seed: int = 7, size=(300, 200)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8))


def test_dgm_eval_transform_matches_torchvision():
    import torchvision.transforms as TF

    image = _test_image()
    ref = TF.Compose(
        [
            TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
            TF.ToTensor(),
            TF.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(image).numpy()
    ours = dgm_eval_transform(image)
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)


def test_dgm_eval_clean_transform_matches_reference():
    from torchvision.transforms.functional import to_tensor

    image = _test_image(seed=8)

    # dgm_eval.resizer.pil_resize, verbatim.
    def pil_resize(x, output_size):
        s1, s2 = output_size
        def resize_single_channel(x):
            img = Image.fromarray(x, mode="F")
            img = img.resize(output_size, resample=Image.BICUBIC)
            return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)
        x = np.array(x.convert("RGB")).astype(np.float32)
        x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
        x = np.concatenate(x, axis=2).astype(np.float32)
        return to_tensor(x) / 255

    ref = pil_resize(image, (224, 224)).numpy()
    ref = (ref - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    ours = dgm_eval_clean_transform(image)
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-6)


def test_classification_transform_matches_torchvision():
    import torchvision.transforms as TF

    image = _test_image(seed=9, size=(423, 517))
    ref = TF.Compose(
        [
            TF.Resize(256, TF.InterpolationMode.BICUBIC),
            TF.CenterCrop(224),
            TF.ToTensor(),
            TF.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(image).numpy()
    ours = classification_transform(image)
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-6)


def test_kdd_feature_pipeline(torch_model, jax_model, model_name):
    """The exact dgm-eval / KDD setup: PIL image -> bicubic 224 resize ->
    normalize -> CLS features, compared end to end."""
    images = [_test_image(seed=10 + i, size=(640, 480)) for i in range(4)]
    batch = np.stack([dgm_eval_transform(im) for im in images])
    with torch.no_grad():
        ref = torch_model(torch.from_numpy(batch))
    out = np.asarray(jax.jit(type(jax_model).__call__)(jax_model, jnp.asarray(batch)))
    ref = ref.numpy()

    assert_features_close(out, torch.from_numpy(ref), err_msg="KDD cls features")
    # Cosine similarity is what kernel distance ultimately sees.
    cos = np.sum(out * ref, -1) / (np.linalg.norm(out, axis=-1) * np.linalg.norm(ref, axis=-1))
    assert np.all(cos > 1 - 1e-6), f"cosine similarity too low: {cos}"


def test_config_matches_torch(torch_model, jax_model, model_name):
    """The static architecture facts torch exposes should agree."""
    assert torch_model.embed_dim == jax_model.embed_dim
    assert torch_model.patch_size == jax_model.patch_size
    assert torch_model.num_register_tokens == jax_model.num_register_tokens
    assert len(torch_model.blocks) == len(jax_model.blocks)
    assert torch_model.interpolate_offset == jax_model.config.interpolate_offset
    assert torch_model.interpolate_antialias == jax_model.config.interpolate_antialias
