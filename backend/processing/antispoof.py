import numpy as np
import torch
import torch.nn.functional as F

from ml import transform as trans
from models.registry import ModelRegistry


def predict_single(img, model, device):
    """Run a single anti-spoof model on a cropped patch."""
    test_transform = trans.Compose([trans.ToTensor()])
    img = test_transform(img)
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        result = model.forward(img)
        result = F.softmax(result, dim=1).cpu().numpy()
    return result


def predict_ensemble(frame, bbox, registry: ModelRegistry):
    """Run all anti-spoof models on different scale crops and average predictions.
    Returns (label, confidence) where label=1 means Real."""
    bbox_list = [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
    prediction = np.zeros((1, 3))

    for as_model in registry.antispoof_models:
        crop_scale = as_model.scale if as_model.scale is not None else 1.0
        patch = registry.cropper.crop(
            org_img=frame,
            bbox=bbox_list,
            scale=crop_scale,
            out_w=as_model.w_input,
            out_h=as_model.h_input,
            crop=as_model.scale is not None,
        )
        result = predict_single(patch, as_model.model, registry.device)
        prediction += result

    prediction /= len(registry.antispoof_models)
    label = int(np.argmax(prediction))
    confidence = float(prediction[0][label])
    return label, confidence
