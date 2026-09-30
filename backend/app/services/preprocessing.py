import io
import json
from pathlib import Path
from typing import Union, BinaryIO, Dict, Any, Tuple, Optional
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms

from ..core.config import settings

CONFIG_FILE = settings.CONFIGS_DIR / "dataloader_preprocessing_config.json"
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP", "TIFF", "MPO"}


class ImagePreprocessingError(ValueError):
    """Base exception for image preprocessing errors."""
    pass


class EmptyImageError(ImagePreprocessingError):
    """Raised when the image file is empty."""
    pass


class CorruptImageError(ImagePreprocessingError):
    """Raised when the image file is corrupt or truncated."""
    pass


class UnsupportedImageFormatError(ImagePreprocessingError):
    """Raised when the image file format is not supported."""
    pass


def load_preprocessing_config() -> Dict[str, Any]:
    """
    Loads preprocessing parameters from configs/dataloader_preprocessing_config.json.
    Raises FileNotFoundError if the research configuration file is missing.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Required research config file not found at: {CONFIG_FILE}"
        )

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def get_preprocessing_pipeline_info() -> Dict[str, Any]:
    """Returns the parsed configuration parameters for the inference pipeline."""
    cfg = load_preprocessing_config()
    eval_cfg = cfg.get("evaluation_preprocessing", {})
    resize = eval_cfg.get("resize", [cfg.get("resize_size", 256), cfg.get("resize_size", 256)])
    center_crop = eval_cfg.get("center_crop", cfg.get("image_size", 224))
    mean = cfg.get("normalization_mean", [0.485, 0.456, 0.406])
    std = cfg.get("normalization_std", [0.229, 0.224, 0.225])

    return {
        "resize": tuple(resize),
        "center_crop": center_crop,
        "image_size": cfg.get("image_size", 224),
        "normalization_mean": mean,
        "normalization_std": std,
        "supported_formats": sorted(list(SUPPORTED_FORMATS)),
    }


def validate_and_load_image(source: Union[bytes, BinaryIO, Image.Image, str, Path]) -> Image.Image:
    """
    Validates and loads an image into a verified PIL RGB Image.
    Catches corrupt images, empty files, and unsupported formats.
    """
    if isinstance(source, Image.Image):
        if source.size[0] <= 0 or source.size[1] <= 0:
            raise CorruptImageError("Image has invalid non-positive dimensions.")
        return source.convert("RGB") if source.mode != "RGB" else source

    raw_bytes: bytes

    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        raw_bytes = p.read_bytes()
    elif isinstance(source, bytes):
        raw_bytes = source
    elif hasattr(source, "read"):
        raw_bytes = source.read()
    else:
        raise ImagePreprocessingError(f"Unsupported source type: {type(source)}")

    if not raw_bytes or len(raw_bytes) == 0:
        raise EmptyImageError("Uploaded image file is empty (0 bytes).")

    # 1. First pass: open and verify integrity
    try:
        buffer = io.BytesIO(raw_bytes)
        img = Image.open(buffer)
        detected_format = (img.format or "").upper()

        if detected_format and detected_format not in SUPPORTED_FORMATS:
            raise UnsupportedImageFormatError(
                f"Unsupported image format: '{detected_format}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        img.verify()
    except UnidentifiedImageError as err:
        raise UnsupportedImageFormatError(
            f"Cannot identify image file. Ensure file is a valid image: {str(err)}"
        ) from err
    except (UnsupportedImageFormatError, EmptyImageError):
        raise
    except Exception as err:
        raise CorruptImageError(f"Corrupt or truncated image data: {str(err)}") from err

    # 2. Second pass: reopen for actual pixel manipulation (verify() invalidates stream)
    try:
        reopened_buffer = io.BytesIO(raw_bytes)
        img = Image.open(reopened_buffer)
        img.load()  # Force loading of pixel data to catch truncated streams

        if img.size[0] <= 0 or img.size[1] <= 0:
            raise CorruptImageError("Image has invalid non-positive dimensions.")

        if img.mode != "RGB":
            img = img.convert("RGB")

        return img
    except (ImagePreprocessingError, UnsupportedImageFormatError, EmptyImageError):
        raise
    except Exception as err:
        raise CorruptImageError(f"Failed to decode image pixels: {str(err)}") from err


def get_inference_transforms() -> transforms.Compose:
    """
    Builds the torchvision transform pipeline strictly adhering to the research config:
    Resize(256, 256) -> CenterCrop(224) -> ToTensor() -> Normalize(mean, std)
    """
    info = get_preprocessing_pipeline_info()
    resize = info["resize"]
    center_crop = info["center_crop"]
    mean = info["normalization_mean"]
    std = info["normalization_std"]

    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def preprocess_image(
    source: Union[bytes, BinaryIO, Image.Image, str, Path],
    device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    """
    Full inference preprocessing pipeline:
    1. Validates and loads image (catches corrupt/unsupported images)
    2. Applies research transforms (Resize 256x256 -> CenterCrop 224x224 -> ToTensor -> Normalize)
    3. Adds batch dimension -> returns (1, 3, 224, 224) Tensor on specified device.
    """
    image = validate_and_load_image(source)
    transform = get_inference_transforms()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)

    if device:
        tensor = tensor.to(device)

    return tensor
