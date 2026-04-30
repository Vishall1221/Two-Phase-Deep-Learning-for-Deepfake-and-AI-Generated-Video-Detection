from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from facenet_pytorch import MTCNN
from flask import url_for
from PIL import Image
from torchvision import transforms

from config import (
    DEVICE,
    NUM_PHASE2_FACE_CROPS,
    PHASE2_CANDIDATE_MULTIPLIER,
    PHASE2_EFFICIENTNET_B2_IMAGE_SIZE,
    PHASE2_EFFICIENTNET_B2_MODEL_PATH,
    PHASE2_ENSEMBLE_WEIGHTS,
    PHASE2_XCEPTION_IMAGE_SIZE,
    PHASE2_XCEPTION_MODEL_PATH,
    PREVIEW_FOLDER,
)


@dataclass(frozen=True)
class Phase2ModelBundle:
    name: str
    checkpoint_path: Path
    image_size: int
    weight: float
    tta_flips: bool
    model: torch.nn.Module
    transform: transforms.Compose


PHASE2_BUNDLES: list[Phase2ModelBundle] | None = None
PHASE2_MTCNN: MTCNN | None = None


def build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_model(model_name: str, drop_rate: float = 0.0) -> torch.nn.Module:
    return timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=drop_rate)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.squeeze(1))


def candidate_frame_indices(frame_count: int, target_images: int) -> list[int]:
    max_candidates = max(target_images * PHASE2_CANDIDATE_MULTIPLIER, target_images)
    capped_count = min(frame_count, max_candidates)
    if capped_count <= target_images:
        return list(range(capped_count))
    positions = [round(i * (frame_count - 1) / (capped_count - 1)) for i in range(capped_count)]
    return sorted(set(positions))


def get_phase2_mtcnn() -> MTCNN:
    global PHASE2_MTCNN
    if PHASE2_MTCNN is None:
        PHASE2_MTCNN = MTCNN(
            image_size=224,
            margin=16,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            select_largest=True,
            keep_all=False,
            device=DEVICE,
        )
    return PHASE2_MTCNN


def load_phase2_bundles() -> list[Phase2ModelBundle]:
    global PHASE2_BUNDLES
    if PHASE2_BUNDLES is not None:
        return PHASE2_BUNDLES

    bundle_configs = [
        ("xception", PHASE2_XCEPTION_MODEL_PATH, PHASE2_XCEPTION_IMAGE_SIZE, PHASE2_ENSEMBLE_WEIGHTS["xception"]),
        (
            "efficientnet_b2",
            PHASE2_EFFICIENTNET_B2_MODEL_PATH,
            PHASE2_EFFICIENTNET_B2_IMAGE_SIZE,
            PHASE2_ENSEMBLE_WEIGHTS["efficientnet_b2"],
        ),
    ]

    loaded: list[Phase2ModelBundle] = []
    for name, checkpoint_path, image_size, weight in bundle_configs:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint["args"]
        model_name = str(args["model_name"])
        drop_rate = float(args.get("drop_rate", 0.0))
        tta_flips = bool(args.get("eval_tta_flips", False))

        model = create_model(model_name, drop_rate=drop_rate)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device=DEVICE, memory_format=torch.channels_last)
        model.eval()

        loaded.append(
            Phase2ModelBundle(
                name=name,
                checkpoint_path=checkpoint_path,
                image_size=image_size,
                weight=weight,
                tta_flips=tta_flips,
                model=model,
                transform=build_eval_transform(image_size),
            )
        )

    PHASE2_BUNDLES = loaded
    return PHASE2_BUNDLES


def extract_faces_from_video(video_path: str, output_dir: Path) -> dict[str, object]:
    mtcnn = get_phase2_mtcnn()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Phase 2 could not open the uploaded video.")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError("Phase 2 could not read the video frame count.")

    output_dir.mkdir(parents=True, exist_ok=True)
    indices = candidate_frame_indices(frame_count, NUM_PHASE2_FACE_CROPS)
    written_paths: list[Path] = []
    source_frame_indices: list[int] = []

    try:
        for frame_index in indices:
            if len(written_paths) >= NUM_PHASE2_FACE_CROPS:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            destination = output_dir / f"face_{len(written_paths):03d}.png"
            face_tensor = mtcnn(image, save_path=str(destination))
            if face_tensor is None:
                continue
            written_paths.append(destination)
            source_frame_indices.append(frame_index)
    finally:
        capture.release()

    if not written_paths:
        raise ValueError("No faces were detected for Phase 2.")

    detected_faces_before_padding = len(written_paths)
    while len(written_paths) < NUM_PHASE2_FACE_CROPS:
        source_image = written_paths[-1]
        destination = output_dir / f"face_{len(written_paths):03d}{source_image.suffix.lower()}"
        shutil.copy2(source_image, destination)
        written_paths.append(destination)
        source_frame_indices.append(source_frame_indices[-1])

    return {
        "face_paths": [str(path) for path in written_paths],
        "source_frame_indices": source_frame_indices,
        "detected_faces_before_padding": detected_faces_before_padding,
        "padding_applied": len(written_paths) - detected_faces_before_padding,
        "frame_count": frame_count,
    }


@torch.no_grad()
def predict_phase2(video_path: str) -> dict[str, object]:
    bundles = load_phase2_bundles()
    request_id = uuid.uuid4().hex[:8]
    preview_dir = PREVIEW_FOLDER / f"phase2_{request_id}"
    preview_dir.mkdir(exist_ok=True)

    extraction = extract_faces_from_video(video_path, preview_dir)
    face_paths = extraction["face_paths"]

    per_model_frame_probs: dict[str, list[float]] = {}
    per_model_video_probs: dict[str, float] = {}

    for bundle in bundles:
        frame_probs: list[float] = []
        batch_size = 8
        for start in range(0, len(face_paths), batch_size):
            tensors = []
            for image_path in face_paths[start : start + batch_size]:
                image = Image.open(image_path).convert("RGB")
                tensors.append(bundle.transform(image))
            batch = torch.stack(tensors).to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            logits = bundle.model(batch)
            probs = logits_to_probabilities(logits)
            if bundle.tta_flips:
                flipped_logits = bundle.model(torch.flip(batch, dims=[3]))
                probs = (probs + logits_to_probabilities(flipped_logits)) / 2.0
            frame_probs.extend(float(value) for value in probs.detach().cpu().tolist())

        per_model_frame_probs[bundle.name] = frame_probs
        per_model_video_probs[bundle.name] = float(np.mean(frame_probs))

    ensemble_frame_probs = [
        float(
            PHASE2_ENSEMBLE_WEIGHTS["xception"] * per_model_frame_probs["xception"][i]
            + PHASE2_ENSEMBLE_WEIGHTS["efficientnet_b2"] * per_model_frame_probs["efficientnet_b2"][i]
        )
        for i in range(len(face_paths))
    ]
    ensemble_video_probability = float(np.mean(ensemble_frame_probs))
    prediction_key = "deepfake" if ensemble_video_probability >= 0.5 else "real"
    prediction_label = "Deepfake" if prediction_key == "deepfake" else "Real"
    confidence = ensemble_video_probability if prediction_key == "deepfake" else 1.0 - ensemble_video_probability

    face_previews = []
    for preview_number, face_path in enumerate(face_paths[:4], start=1):
        face_index = preview_number - 1
        destination = Path(face_path)
        face_previews.append(
            {
                "index": preview_number,
                "face_number": face_index + 1,
                "frame_index": extraction["source_frame_indices"][face_index],
                "image": url_for("static", filename=f"previews/phase2_{request_id}/{destination.name}"),
            }
        )

    return {
        "prediction_key": prediction_key,
        "prediction_label": prediction_label,
        "confidence": round(confidence, 4),
        "ensemble_probability": round(ensemble_video_probability, 4),
        "xception_probability": round(per_model_video_probs["xception"], 4),
        "efficientnet_b2_probability": round(per_model_video_probs["efficientnet_b2"], 4),
        "frame_probabilities": [round(value, 4) for value in ensemble_frame_probs],
        "face_previews": face_previews,
    }
