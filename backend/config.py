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

# Flash Liveness
FLASH_NUM_COLORS = 18
FLASH_COLOR_DURATION_MS = 150
FLASH_GRID_ROWS = 8
FLASH_GRID_COLS = 8
FLASH_TEMPORAL_CORR_WEIGHT = 0.40
FLASH_LAG_WEIGHT = 0.15
FLASH_SPATIAL_VAR_WEIGHT = 0.25
FLASH_RATIO_STABILITY_WEIGHT = 0.20
FLASH_PASS_THRESHOLD = 0.50
FLASH_TEMPORAL_CORR_THRESHOLD = 0.3
FLASH_LAG_TOLERANCE_FRAMES = 2
FLASH_SPATIAL_VAR_MIN = 0.14
FLASH_SPATIAL_VAR_MAX = 0.30
FLASH_RATIO_STABILITY_THRESHOLD = 0.85
FLASH_EYE_SPECULAR_RATIO_MIN = 2.0

# Server
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
