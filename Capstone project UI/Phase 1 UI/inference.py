import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import url_for

import torch
from torchvision import transforms

from config import NUM_FRAMES, IMG_SIZE, DEVICE, CLASS_NAMES, STATIC_FOLDER, PREVIEW_FOLDER
from model import load_model

MODEL = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]),
])


def sample_evenly_spaced_frames(video_path: str, num_frames: int = 15) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open uploaded video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read total frame count from video.")

    frame_indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames could be extracted from the uploaded video.")

    return frames


def make_residual(img_bgr: np.ndarray, blur_kernel=(5, 5), sigma=0) -> np.ndarray:
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
def predict_video(video_path: str) -> dict:
    frames_bgr = sample_evenly_spaced_frames(video_path, num_frames=NUM_FRAMES)

    frame_probs_camera = []
    preview_items = []

    request_id = uuid.uuid4().hex[:8]
    preview_dir = PREVIEW_FOLDER / f"preview_{request_id}"
    preview_dir.mkdir(exist_ok=True)

    for i, frame_bgr in enumerate(frames_bgr, start=1):
        residual_img = make_residual(frame_bgr)
        fft_img = make_fft_image(residual_img)

        residual_tensor = transform(gray_to_rgb_pil(residual_img)).unsqueeze(0).to(DEVICE)
        fft_tensor = transform(gray_to_rgb_pil(fft_img)).unsqueeze(0).to(DEVICE)

        logits = MODEL(residual_tensor, fft_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        prob_ai = float(probs[0])
        prob_camera = float(probs[1])
        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]

        frame_probs_camera.append(prob_camera)

        if i <= 3:
            orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            orig_path = preview_dir / f"frame_{i:02d}_orig.jpg"
            residual_path = preview_dir / f"frame_{i:02d}_residual.jpg"
            fft_path = preview_dir / f"frame_{i:02d}_fft.jpg"

            save_preview_rgb(orig_rgb, orig_path)
            save_preview_gray(residual_img, residual_path)
            save_preview_gray(fft_img, fft_path)

            preview_items.append({
                "index": i,
                "orig": url_for("static", filename=f"previews/preview_{request_id}/{orig_path.name}"),
                "residual": url_for("static", filename=f"previews/preview_{request_id}/{residual_path.name}"),
                "fft": url_for("static", filename=f"previews/preview_{request_id}/{fft_path.name}"),
                "frame_prob_ai": round(prob_ai, 4),
                "frame_prob_camera": round(prob_camera, 4),
                "frame_pred": pred_label,
            })

    avg_prob_camera = float(np.mean(frame_probs_camera))
    avg_prob_ai = 1.0 - avg_prob_camera
    video_pred_idx = 1 if avg_prob_camera >= 0.5 else 0
    video_pred_label = CLASS_NAMES[video_pred_idx]
    confidence = avg_prob_camera if video_pred_label == "camera" else avg_prob_ai

    return {
        "prediction": video_pred_label,
        "confidence": round(confidence, 4),
        "avg_prob_ai": round(avg_prob_ai, 4),
        "avg_prob_camera": round(avg_prob_camera, 4),
        "frames_used": len(frames_bgr),
        "frame_probs_camera": [round(x, 4) for x in frame_probs_camera],
        "previews": preview_items,
    }