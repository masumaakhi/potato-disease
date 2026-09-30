import io
import pytest
import torch
from PIL import Image

from app.services.preprocessing import (
    load_preprocessing_config,
    get_preprocessing_pipeline_info,
    get_inference_transforms,
    validate_and_load_image,
    preprocess_image,
    EmptyImageError,
    CorruptImageError,
    UnsupportedImageFormatError,
    ImagePreprocessingError,
)


def create_test_image(mode: str = "RGB", size: tuple = (300, 300), color=(100, 150, 200)) -> bytes:
    """Helper to create valid in-memory image bytes."""
    buf = io.BytesIO()
    if mode == "RGBA":
        color = (100, 150, 200, 255)
    elif mode == "L":
        color = 128
    elif mode == "CMYK":
        color = (50, 100, 150, 0)
    img = Image.new(mode, size, color=color)
    img.save(buf, format="PNG" if mode == "RGBA" else "JPEG")
    return buf.getvalue()


def test_preprocessing_config_loaded_correctly():
    """Verify parameters match research dataloader_preprocessing_config.json."""
    info = get_preprocessing_pipeline_info()
    assert info["resize"] == (256, 256)
    assert info["center_crop"] == 224
    assert info["image_size"] == 224
    assert info["normalization_mean"] == [0.485, 0.456, 0.406]
    assert info["normalization_std"] == [0.229, 0.224, 0.225]


def test_preprocess_rgb_image():
    """Verify RGB image produces correct tensor shape and dtype."""
    img_bytes = create_test_image("RGB", (320, 240))
    tensor = preprocess_image(img_bytes)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_different_modes():
    """Verify RGBA, Grayscale, and CMYK images are correctly converted to 3-channel RGB."""
    # RGBA
    rgba_bytes = create_test_image("RGBA", (256, 256))
    t_rgba = preprocess_image(rgba_bytes)
    assert t_rgba.shape == (1, 3, 224, 224)

    # Grayscale
    gray_bytes = create_test_image("L", (256, 256))
    t_gray = preprocess_image(gray_bytes)
    assert t_gray.shape == (1, 3, 224, 224)

    # CMYK
    cmyk_bytes = create_test_image("CMYK", (256, 256))
    t_cmyk = preprocess_image(cmyk_bytes)
    assert t_cmyk.shape == (1, 3, 224, 224)


def test_preprocess_from_pil_image():
    """Verify preprocess_image directly accepts PIL Image objects."""
    pil_img = Image.new("RGB", (400, 400), color=(50, 120, 60))
    tensor = preprocess_image(pil_img)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_normalization_applied():
    """Verify normalization formula (x - mean) / std is applied."""
    # Create pure black image (pixels = 0)
    black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    tensor = preprocess_image(black_img)

    # Expected channel 0 value for 0 input: (0 - 0.485) / 0.229 = -2.1179
    expected_c0 = (0.0 - 0.485) / 0.229
    actual_c0 = tensor[0, 0, 0, 0].item()
    assert abs(actual_c0 - expected_c0) < 0.05


def test_empty_image_error():
    """Verify empty image input raises EmptyImageError."""
    with pytest.raises(EmptyImageError):
        preprocess_image(b"")


def test_corrupt_image_error():
    """Verify corrupted image bytes raise an appropriate image exception."""
    corrupt_bytes = b"GIF89a\x00\x00\x00\x00" + b"\xff" * 20  # Incomplete header
    with pytest.raises((CorruptImageError, UnsupportedImageFormatError)):
        preprocess_image(corrupt_bytes)


def test_non_image_file_error():
    """Verify random text bytes raise UnsupportedImageFormatError."""
    text_bytes = b"This is not a potato leaf image, just plain text."
    with pytest.raises(UnsupportedImageFormatError):
        preprocess_image(text_bytes)
