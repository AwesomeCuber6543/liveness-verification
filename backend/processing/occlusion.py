import cv2
import torch
from PIL import Image


def predict_occlusion(face_crop_bgr, occlusion_model, occlusion_transform, device):
    """Run occlusion classification on a cropped face (BGR numpy array).
    Returns (prediction, confidence) where prediction is 0=Clear, 1=Occluded."""
    face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    img_tensor = occlusion_transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = occlusion_model(img_tensor)
        probs = torch.softmax(output, 1)
        confidence, prediction = torch.max(probs, 1)
    return prediction.item(), confidence.item()
