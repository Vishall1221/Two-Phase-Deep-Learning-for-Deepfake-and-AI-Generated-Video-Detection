from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
STATIC_FOLDER = BASE_DIR / "static"
PREVIEW_FOLDER = STATIC_FOLDER / "previews"
MODEL_PATH = BASE_DIR / "two_branch_residual_fft_best.pth"

UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_FOLDER.mkdir(exist_ok=True)
PREVIEW_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}
NUM_FRAMES = 15
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["AI", "camera"]
MAX_CONTENT_LENGTH = 300 * 1024 * 1024  # 300 MB