from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from flask import url_for
from torchvision import transforms

from config import DEVICE, NUM_PHASE1_FRAMES, PHASE1_CLASS_NAMES, PHASE1_IMG_SIZE, PREVIEW_FOLDER
from phase1_model import load_phase1_model


PHASE1_MODEL = None

PHASE1_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((PHASE1_IMG_SIZE, PHASE1_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
    ]
)


def get_phase1_model():
    global PHASE1_MODEL
    if PHASE1_MODEL is None:
        PHASE1_MODEL = load_phase1_model()
    return PHASE1_MODEL


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    positions = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    return sorted(set(positions))


def sample_evenly_spaced_frames(video_path: str, num_frames: int = NUM_PHASE1_FRAMES) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read total frame count from video.")

    frame_indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    frames: list[np.ndarray] = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video.")

    return frames, frame_indices.tolist()[: len(frames)]


def make_residual(img_bgr: np.ndarray, blur_kernel: tuple[int, int] = (5, 5), sigma: int = 0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, sigma)
    residual = gray.astype(np.float32) - blurred.astype(np.float32)
    residual_norm = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
    return residual_norm.astype(np.uint8)


def make_fft_image(residual_img: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(residual_img)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_norm.astype(np.uint8)


def gray_to_rgb_pil(gray_img: np.ndarray) -> Image.Image:
    return Image.fromarray(gray_img).convert("L").convert("RGB")


def save_preview_rgb(img_rgb: np.ndarray, out_path: Path) -> None:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)


def save_preview_gray(gray_img: np.ndarray, out_path: Path) -> None:
    cv2.imwrite(str(out_path), gray_img)


@torch.no_grad()
def predict_phase1(video_path: str) -> dict[str, object]:
    model = get_phase1_model()
    frames_bgr, frame_indices = sample_evenly_spaced_frames(video_path, num_frames=NUM_PHASE1_FRAMES)

    frame_probs_camera: list[float] = []
    preview_items: list[dict[str, object]] = []

    request_id = uuid.uuid4().hex[:8]
    preview_dir = PREVIEW_FOLDER / f"phase1_{request_id}"
    preview_dir.mkdir(exist_ok=True)

    preview_targets = set(evenly_spaced_indices(len(frames_bgr), min(3, len(frames_bgr))))

    for i, (frame_bgr, frame_index) in enumerate(zip(frames_bgr, frame_indices), start=1):
        residual_img = make_residual(frame_bgr)
        fft_img = make_fft_image(residual_img)

        residual_tensor = PHASE1_TRANSFORM(gray_to_rgb_pil(residual_img)).unsqueeze(0).to(DEVICE)
        fft_tensor = PHASE1_TRANSFORM(gray_to_rgb_pil(fft_img)).unsqueeze(0).to(DEVICE)

        logits = model(residual_tensor, fft_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        prob_ai = float(probs[0])
        prob_camera = float(probs[1])
        pred_idx = int(np.argmax(probs))
        pred_key = PHASE1_CLASS_NAMES[pred_idx]

        frame_probs_camera.append(prob_camera)

        sample_zero_index = i - 1
        if sample_zero_index in preview_targets:
            orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            orig_path = preview_dir / f"frame_{i:02d}_orig.jpg"
            residual_path = preview_dir / f"frame_{i:02d}_residual.jpg"
            fft_path = preview_dir / f"frame_{i:02d}_fft.jpg"

            save_preview_rgb(orig_rgb, orig_path)
            save_preview_gray(residual_img, residual_path)
            save_preview_gray(fft_img, fft_path)

            preview_items.append(
                {
                    "index": i,
                    "frame_index": frame_index,
                    "orig": url_for("static", filename=f"previews/phase1_{request_id}/{orig_path.name}"),
                    "residual": url_for("static", filename=f"previews/phase1_{request_id}/{residual_path.name}"),
                    "fft": url_for("static", filename=f"previews/phase1_{request_id}/{fft_path.name}"),
                    "frame_prob_ai": round(prob_ai, 4),
                    "frame_prob_camera": round(prob_camera, 4),
                    "frame_pred": "Camera Captured" if pred_key == "camera" else "AI Generated",
                }
            )

    avg_prob_camera = float(np.mean(frame_probs_camera))
    avg_prob_ai = 1.0 - avg_prob_camera
    prediction_key = "camera" if avg_prob_camera >= 0.5 else "ai"
    prediction_label = "Camera Captured" if prediction_key == "camera" else "AI Generated"
    confidence = avg_prob_camera if prediction_key == "camera" else avg_prob_ai

    return {
        "prediction_key": prediction_key,
        "prediction_label": prediction_label,
        "confidence": round(confidence, 4),
        "avg_prob_ai": round(avg_prob_ai, 4),
        "avg_prob_camera": round(avg_prob_camera, 4),
        "frames_used": len(frames_bgr),
        "frame_probs_camera": [round(x, 4) for x in frame_probs_camera],
        "frame_indices": frame_indices,
        "previews": preview_items,
    }
