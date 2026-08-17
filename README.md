# dinov2-jax

[DINOv2](https://github.com/facebookresearch/dinov2) vision transformers in
JAX/Equinox, weight-compatible with the official Meta checkpoints and parity
tested against the reference PyTorch implementation.

The motivating use case is computing KDD (kernel distance with DINOv2
features, as in [Stein et al.](https://github.com/layer6ai-labs/dgm-eval)):
`dinov2_vitl14` CLS-token features over dgm-eval's preprocessing, reproduced
here exactly. But the full backbone API is ported, so it works as a general
DINOv2 feature extractor for any of the released checkpoints.

## Usage

```python
import jax, jax.numpy as jnp
from PIL import Image
from dinov2_jax import load_dinov2, dgm_eval_transform

# Any of dinov2_vits14 / vitb14 / vitl14 / vitg14, plus the _reg variants.
# Downloads + converts the official checkpoint on first use
# (cached in ~/.cache/dinov2-jax, override with DINOV2_JAX_CACHE).
model = load_dinov2("dinov2_vitl14")

# KDD-standard features: dgm-eval preprocessing, CLS token (1024-d for ViT-L).
image = Image.open("image.png")
batch = jnp.asarray(dgm_eval_transform(image))[None]  # (1, 3, 224, 224)
features = jax.jit(type(model).__call__)(model, batch)  # (1, 1024)
```

The rest of the official backbone API is available with the same semantics:

```python
out = model.forward_features(batch)
# {"x_norm_clstoken", "x_norm_regtokens", "x_norm_patchtokens", "x_prenorm", "masks"}

layers = model.get_intermediate_layers(batch, n=4, reshape=True, return_class_token=True)
```

Inputs are `(batch, 3, H, W)`, ImageNet-normalized, with H and W multiples of
the patch size (14). Non-224 resolutions work exactly as in torch: the learned
position embeddings are resampled with a bit-exact port of torch's bicubic
interpolation, including the historical `+0.1` scale-factor kludge (plain
checkpoints) and antialiasing (`_reg` checkpoints) — this is load-bearing for
output parity, don't simplify it.

Preprocessing helpers in `dinov2_jax.transforms` (all pure PIL/numpy, no torch
dependency): `dgm_eval_transform` (dgm-eval default: bicubic resize to
224x224 + ImageNet normalize), `dgm_eval_clean_transform` (dgm-eval's
`clean_resize`), `classification_transform` (official dinov2 eval transform:
resize 256, center crop 224).

### KDD notes

For numbers comparable to published KDD tables: use `dinov2_vitl14` (no
registers — the `_reg` variants shift the feature distribution and change the
metric scale), `dgm_eval_transform`, CLS features, and dgm-eval's kernel
defaults for the MMD estimator itself.

On Ampere+ GPUs, JAX quietly runs fp32 matmuls in TF32 by default, which
drifts features by ~1e-2 relative to torch (torch defaults TF32 *off* for
matmuls, so dgm-eval's reference features are true fp32). For metric-grade
features set `JAX_DEFAULT_MATMUL_PRECISION=highest` (or
`jax.config.update("jax_default_matmul_precision", "highest")`).

### Checkpoint conversion

The official checkpoints are torch pickles, so the *first* load of a given
model needs torch importable to convert it to safetensors (cached; torch is
never needed again). Either install the `convert` extra or convert on a
machine that has torch and copy the `.safetensors` file into the cache.

## Development

```bash
uv sync --group test
uv run pytest            # parity tests; downloads checkpoints on first run
```

The test suite compares against `torch.hub.load("facebookresearch/dinov2", ...)`
running in fp32 on CPU: bidirectional state-dict correspondence, bicubic
resampling versus `torch.nn.functional.interpolate` as oracle, per-layer
input/output parity, end-to-end `forward_features` /
`get_intermediate_layers` / masking parity at several resolutions, exact
preprocessing equivalence against torchvision, and the full dgm-eval KDD
feature pipeline. `DINOV2_JAX_TEST_MODELS` selects which checkpoints to test
(default `dinov2_vits14,dinov2_vits14_reg,dinov2_vitl14`); set it to include
`dinov2_vitg14` for the 4.4GB giant.

## Structure

- `model.py` — the `DinoVisionTransformer` and its layers; module/field names
  track the torch implementation so a `Param`'s tree path is exactly its
  state-dict key.
- `resample.py` — torch-exact bicubic resampling for position embeddings.
- `loading.py` — checkpoint download / conversion / loading.
- `transforms.py` — dgm-eval and official preprocessing pipelines.
- `param.py`, `equinox_utils.py`, `linear.py` — parameter-slot machinery in
  the style of [qwenjax](https://github.com/nshepperd/qwenjax).
