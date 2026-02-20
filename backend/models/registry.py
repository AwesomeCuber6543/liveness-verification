from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class AntiSpoofModel:
    name: str
    model: torch.nn.Module
    h_input: int
    w_input: int
    scale: float | None


@dataclass
class ModelRegistry:
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    antispoof_models: list[AntiSpoofModel] = field(default_factory=list)
    occlusion_model: torch.nn.Module | None = None
    occlusion_transform: Any = None
    cropper: Any = None
