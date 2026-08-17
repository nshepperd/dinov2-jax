"""Image preprocessing, matching the conventions DINOv2 features are used with.

Everything here is PIL + numpy, replicating the torchvision pipelines
bit-for-bit (torchvision's Resize/CenterCrop on PIL images just call into PIL,
and ToTensor is a divide by 255), so preprocessing parity does not depend on
torch being installed.

For KDD / dgm-eval numbers use `dgm_eval_transform` -- a straight bicubic
resize to 224x224 plus ImageNet normalization, dgm-eval's default. Its
`clean_resize` variant resizes float channels instead of uint8.
`classification_transform` is the official dinov2 eval transform
(resize-256 shorter side, center-crop 224).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize(
    x: np.ndarray, mean: np.ndarray = IMAGENET_MEAN, std: np.ndarray = IMAGENET_STD
) -> np.ndarray:
    """Channel-normalize a (3, H, W) float image in [0, 1]."""
    return (x - mean[:, None, None]) / std[:, None, None]


def _to_chw(image: Image.Image) -> np.ndarray:
    """torchvision ToTensor: HWC uint8 -> CHW float32 in [0, 1]."""
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


def dgm_eval_transform(image: Image.Image, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """dgm-eval's default DINOv2 transform: bicubic resize to 224x224,
    ToTensor, ImageNet normalize. Returns (3, 224, 224) float32."""
    image = image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BICUBIC)
    return normalize(_to_chw(image))


def dgm_eval_clean_transform(image: Image.Image, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """dgm-eval's `clean_resize` transform: resize each channel as a float
    ('F'-mode) PIL image, clip to [0, 255], then normalize."""
    x = np.asarray(image.convert("RGB"), dtype=np.float32)
    channels = []
    for c in range(3):
        img = Image.fromarray(x[:, :, c], mode="F")
        img = img.resize((size[1], size[0]), resample=Image.Resampling.BICUBIC)
        channels.append(np.asarray(img).clip(0, 255))
    resized = np.stack(channels, axis=0).astype(np.float32) / 255.0
    return normalize(resized)


def classification_transform(
    image: Image.Image, resize_size: int = 256, crop_size: int = 224
) -> np.ndarray:
    """The official dinov2 eval transform: bicubic resize of the shorter side
    to `resize_size`, center crop to `crop_size`, ImageNet normalize."""
    image = image.convert("RGB")
    w, h = image.size
    # torchvision Resize(int): shorter side to resize_size, keep aspect
    # ratio, truncating (not rounding) the long side.
    if w <= h:
        new_w, new_h = resize_size, int(resize_size * h / w)
    else:
        new_w, new_h = int(resize_size * w / h), resize_size
    image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
    # torchvision CenterCrop.
    left = int(round((new_w - crop_size) / 2.0))
    top = int(round((new_h - crop_size) / 2.0))
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return normalize(_to_chw(image))


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "classification_transform",
    "dgm_eval_clean_transform",
    "dgm_eval_transform",
    "normalize",
]
