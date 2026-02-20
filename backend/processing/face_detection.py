import numpy as np
import mediapipe as mp
from config import LANDMARKER_PATH


def create_landmarker():
    """Create a new MediaPipe FaceLandmarker in IMAGE mode (thread-safe, per-session)."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_face(landmarker, frame_rgb: np.ndarray):
    """Run face detection on an RGB frame. Returns (landmarks, bbox) or (None, None)."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None, None

    landmarks = result.face_landmarks[0]
    img_h, img_w = frame_rgb.shape[:2]

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = max(0, int(min(xs) * img_w))
    y_min = max(0, int(min(ys) * img_h))
    x_max = min(img_w, int(max(xs) * img_w))
    y_max = min(img_h, int(max(ys) * img_h))

    bbox = {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}
    return landmarks, bbox
