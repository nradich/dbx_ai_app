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


def training_transform():
    return transforms.Compose([
        transforms.Resize((240, 240), antialias=True),
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


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
