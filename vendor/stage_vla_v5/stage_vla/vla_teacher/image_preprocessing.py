"""OpenVLA evaluation image preprocessing matched to official LIBERO code."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

OPENVLA_IMAGE_SIZE = 224
LIBERO_CENTER_CROP_AREA = 0.9
ImagePreprocessMode = Literal["none", "libero_official"]


def _rgb_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"image must have shape HxWx3 or HxWx4, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.isfinite(array).all():
            raise ValueError("image contains NaN/Inf")
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return Image.fromarray(array[..., :3], mode="RGB")


def preprocess_openvla_image(
    image: Image.Image | np.ndarray,
    *,
    mode: ImagePreprocessMode = "none",
) -> Image.Image:
    """Prepare an RGB image for OpenVLA using an explicit evaluation mode.

    ``libero_official`` reproduces OpenVLA's LIBERO evaluation path: JPEG
    encode/decode, Lanczos3 resize to 224, then a centered crop with area 0.9
    resized back to 224 with TensorFlow's crop-and-resize implementation.
    """
    pil_image = _rgb_pil(image)
    if mode == "none":
        return pil_image
    if mode != "libero_official":
        raise ValueError(f"unsupported OpenVLA image preprocessing mode: {mode}")

    # Lazy import keeps non-LIBERO utilities lightweight while matching the
    # official OpenVLA evaluation implementation exactly when requested.
    import tensorflow as tf

    tensor = tf.convert_to_tensor(np.asarray(pil_image, dtype=np.uint8))

    # Match experiments/robot/libero/libero_utils.py::resize_image.
    tensor = tf.image.encode_jpeg(tensor)
    tensor = tf.io.decode_image(tensor, expand_animations=False, dtype=tf.uint8)
    tensor = tf.image.resize(
        tensor,
        (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE),
        method="lanczos3",
        antialias=True,
    )
    tensor = tf.cast(tf.clip_by_value(tf.round(tensor), 0, 255), tf.uint8)

    # Match experiments/robot/openvla_utils.py::crop_and_resize.
    original_dtype = tensor.dtype
    tensor = tf.image.convert_image_dtype(tensor, tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    side_fraction = tf.sqrt(tf.constant(LIBERO_CENTER_CROP_AREA, dtype=tf.float32))
    offset = (1.0 - side_fraction) / 2.0
    boxes = tf.reshape(
        tf.stack((offset, offset, offset + side_fraction, offset + side_fraction)),
        (1, 4),
    )
    tensor = tf.image.crop_and_resize(
        tensor,
        boxes,
        tf.range(1),
        (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE),
    )[0]
    tensor = tf.clip_by_value(tensor, 0.0, 1.0)
    tensor = tf.image.convert_image_dtype(tensor, original_dtype, saturate=True)
    return Image.fromarray(tensor.numpy(), mode="RGB")
