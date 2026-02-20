import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

BASE_DIR = Path(__file__).resolve().parent

# Weights / model paths
WEIGHTS_DIR = BASE_DIR / os.getenv("WEIGHTS_DIR", "weights")
ANTISPOOF_MODEL_DIR = BASE_DIR / os.getenv("ANTISPOOF_MODEL_DIR", "weights")
OCCLUSION_WEIGHT = BASE_DIR / os.getenv("OCCLUSION_WEIGHT", "weights/best_convnext_tiny.pth")
LANDMARKER_PATH = BASE_DIR / os.getenv("LANDMARKER_PATH", "weights/face_landmarker.task")

# Occlusion classifier
OCCLUSION_MEAN = [0.485, 0.456, 0.406]
OCCLUSION_STD = [0.229, 0.224, 0.225]
OCCLUSION_SIZE = [224, 224]

# Blink detection (MediaPipe landmark indices)
LEFT_EYE = [362, 380, 374, 263, 386, 385]
RIGHT_EYE = [33, 159, 158, 133, 153, 145]
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.30"))
EAR_CONSEC_FRAMES = 2

# Mouth detection
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MAR_THRESHOLD = float(os.getenv("MAR_THRESHOLD", "0.5"))

# Verification
CLEAR_FRAMES_REQUIRED = int(os.getenv("CLEAR_FRAMES_REQUIRED", "2"))

# Server
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
