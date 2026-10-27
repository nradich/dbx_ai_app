"""Shared, deterministic preprocessing for training evaluation and MLflow inference."""
import base64
import binascii
import io

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


def evaluation_transform():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def training_transform(crop_scale=(0.85, 1.0), crop_ratio=(0.9, 1.1), rotation_degrees=8,
                       color_jitter=(0.15, 0.15, 0.1, 0.0), random_erasing_p=0.0):
    """Defaults reproduce the v1 recipe exactly; the improved recipe passes stronger values."""
    brightness, contrast, saturation, hue = color_jitter
    steps = [
        transforms.Resize((240, 240), antialias=True),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=tuple(crop_scale), ratio=tuple(crop_ratio)),
        transforms.RandomRotation(rotation_degrees),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    if random_erasing_p > 0:  # Only appended when enabled so v1 random-number consumption is unchanged.
        steps.append(transforms.RandomErasing(p=random_erasing_p))
    return transforms.Compose(steps)


def open_rgb(path_or_file):
    with Image.open(path_or_file) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def decode_image(value):
    if not isinstance(value, str) or not value:
        raise ValueError("image_base64 must be a nonempty base64-encoded tail image")
    try:
        raw = base64.b64decode(value, validate=True)
        return open_rgb(io.BytesIO(raw))
    except (ValueError, binascii.Error, UnidentifiedImageError, OSError) as error:
        raise ValueError("image_base64 must contain a valid base64-encoded image") from error


def build_model(class_count, pretrained=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, class_count)
    return model
