from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
STATIC_FOLDER = BASE_DIR / "static"
PREVIEW_FOLDER = STATIC_FOLDER / "previews"
UPLOAD_FOLDER = BASE_DIR / "uploads"

PHASE1_MODEL_PATH = MODELS_DIR / "phase1_two_branch_residual_fft_best.pth"
PHASE2_XCEPTION_MODEL_PATH = MODELS_DIR / "phase2_xception_best_model.pth"
PHASE2_EFFICIENTNET_B2_MODEL_PATH = MODELS_DIR / "phase2_efficientnet_b2_best_model.pth"

STATIC_FOLDER.mkdir(exist_ok=True)
PREVIEW_FOLDER.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
MAX_CONTENT_LENGTH = 300 * 1024 * 1024

NUM_PHASE1_FRAMES = 15
PHASE1_IMG_SIZE = 224
PHASE1_CLASS_NAMES = ["ai", "camera"]

NUM_PHASE2_FACE_CROPS = 15
PHASE2_CANDIDATE_MULTIPLIER = 12
PHASE2_XCEPTION_IMAGE_SIZE = 299
PHASE2_EFFICIENTNET_B2_IMAGE_SIZE = 260
PHASE2_ENSEMBLE_WEIGHTS = {"xception": 0.45, "efficientnet_b2": 0.55}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
