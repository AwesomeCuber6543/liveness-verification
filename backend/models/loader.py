import os
import torch
from torchvision import transforms, models
from collections import OrderedDict

from ml.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from ml.utility import get_kernel, parse_model_name
from ml.generate_patches import CropImage
from config import (
    ANTISPOOF_MODEL_DIR, OCCLUSION_WEIGHT,
    OCCLUSION_MEAN, OCCLUSION_STD, OCCLUSION_SIZE,
)
from models.registry import ModelRegistry, AntiSpoofModel

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


def _load_antispoof_model(model_path: str, device: torch.device):
    model_name = os.path.basename(model_path)
    h_input, w_input, model_type, scale = parse_model_name(model_name)
    kernel_size = get_kernel(h_input, w_input)
    model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(device)

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    first_key = next(iter(state_dict))
    if first_key.startswith("module."):
        state_dict = OrderedDict(
            (k[7:], v) for k, v in state_dict.items()
        )
    model.load_state_dict(state_dict)
    model.eval()
    return AntiSpoofModel(
        name=model_name.split(".pth")[0].split("_")[-1],
        model=model,
        h_input=h_input,
        w_input=w_input,
        scale=scale,
    )


def _load_occlusion_model(weight_path: str, device: torch.device):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_features, 2)

    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    state_dict = {
        k.replace("model.", "", 1): v
        for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_all_models() -> ModelRegistry:
    device = torch.device("cpu")
    registry = ModelRegistry(device=device)

    # Anti-spoof models
    for fname in os.listdir(ANTISPOOF_MODEL_DIR):
        if not fname.endswith(".pth") or "convnext" in fname:
            continue
        path = os.path.join(ANTISPOOF_MODEL_DIR, fname)
        as_model = _load_antispoof_model(path, device)
        registry.antispoof_models.append(as_model)
        print(f"Loaded anti-spoof: {fname} (scale={as_model.scale}, input={as_model.h_input}x{as_model.w_input})")

    # Occlusion model
    print(f"Loading occlusion model: {OCCLUSION_WEIGHT}")
    registry.occlusion_model = _load_occlusion_model(str(OCCLUSION_WEIGHT), device)
    registry.occlusion_transform = transforms.Compose([
        transforms.Resize(OCCLUSION_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(OCCLUSION_MEAN, OCCLUSION_STD),
    ])
    print("Occlusion model loaded (ConvNeXt-Tiny)")

    # Cropper
    registry.cropper = CropImage()

    return registry
