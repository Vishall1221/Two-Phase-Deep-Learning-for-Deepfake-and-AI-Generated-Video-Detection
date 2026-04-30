from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import timm
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_ENSEMBLE_CONFIG = MODELS_DIR / "ensemble_config.json"
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "runtime"
TARGET_IMAGES_PER_SAMPLE = 15
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    run_dir: Path
    checkpoint_path: Path
    image_size: int
    drop_rate: float
    tta_flips: bool
    weight: float
    best_epoch: int
    validation_video_accuracy: float
    model: torch.nn.Module
    transform: transforms.Compose


@dataclass(frozen=True)
class ModelPrediction:
    model_name: str
    weight: float
    image_size: int
    tta_flips: bool
    best_epoch: int
    validation_video_accuracy: float
    frame_probabilities: list[float]
    video_probability: float


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
    max_candidates = max(target_images * 12, target_images)
    capped_count = min(frame_count, max_candidates)
    if capped_count <= target_images:
        return list(range(capped_count))
    positions = [round(i * (frame_count - 1) / (capped_count - 1)) for i in range(capped_count)]
    return sorted(set(positions))


def resolve_device(require_cuda: bool = True) -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda:0")
    if require_cuda:
        raise RuntimeError(
            "CUDA GPU was not found in this Python environment. Launch the app with the CUDA-enabled Python311 env."
        )
    return torch.device("cpu")


def _copy_video_to_runtime(video_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_path = destination_dir / video_path.name
    if video_path.resolve() != copied_path.resolve():
        shutil.copy2(video_path, copied_path)
    return copied_path


def extract_faces_from_video(
    video_path: Path,
    output_dir: Path,
    mtcnn: MTCNN,
    images_per_sample: int = TARGET_IMAGES_PER_SAMPLE,
    progress_callback: Callable[[str, float | None], None] | None = None,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video has no readable frames: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_indices = candidate_frame_indices(frame_count, images_per_sample)
    written_paths: list[Path] = []
    source_frame_indices: list[int] = []

    try:
        for index, frame_index in enumerate(candidate_indices, start=1):
            if progress_callback is not None:
                progress_callback(
                    f"Extracting faces from frame {index}/{len(candidate_indices)}",
                    0.05 + 0.45 * (index / max(len(candidate_indices), 1)),
                )
            if len(written_paths) >= images_per_sample:
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
        raise RuntimeError("No faces were detected in the uploaded video.")

    detected_faces_before_padding = len(written_paths)
    while len(written_paths) < images_per_sample:
        source_image = written_paths[-1]
        destination = output_dir / f"face_{len(written_paths):03d}{source_image.suffix.lower()}"
        shutil.copy2(source_image, destination)
        written_paths.append(destination)
        source_frame_indices.append(source_frame_indices[-1])

    return {
        "frame_count": frame_count,
        "candidate_frame_indices": candidate_indices,
        "source_frame_indices": source_frame_indices,
        "face_paths": [str(path) for path in written_paths],
        "detected_faces_before_padding": detected_faces_before_padding,
        "padding_applied": len(written_paths) - detected_faces_before_padding,
    }


class DeepfakeEnsemblePredictor:
    def __init__(
        self,
        *,
        ensemble_config_path: Path = DEFAULT_ENSEMBLE_CONFIG,
        runtime_root: Path = DEFAULT_RUNTIME_ROOT,
        require_cuda: bool = True,
    ) -> None:
        self.ensemble_config_path = ensemble_config_path.resolve()
        self.runtime_root = runtime_root.resolve()
        self.device = resolve_device(require_cuda=require_cuda)
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "uploads").mkdir(exist_ok=True)
        (self.runtime_root / "predictions").mkdir(exist_ok=True)

        ensemble_config = json.loads(self.ensemble_config_path.read_text(encoding="utf-8"))
        self.ensemble_name = str(ensemble_config.get("ensemble_name", "xception_efficientnet_b2"))
        self.ensemble_validation_accuracy = float(ensemble_config["validation_video_accuracy"])
        self.ensemble_test_accuracy = float(ensemble_config["test_video_accuracy"])

        self.models = [
            self._load_model_bundle(model_config) for model_config in ensemble_config["models"]
        ]

        self.mtcnn = MTCNN(
            image_size=224,
            margin=16,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            select_largest=True,
            keep_all=False,
            device=self.device,
        )

    def _load_model_bundle(self, model_config: dict[str, object]) -> ModelBundle:
        checkpoint_path = (PROJECT_ROOT / str(model_config["checkpoint"])).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint["args"]
        model_name = str(args["model_name"])
        image_size = int(args.get("image_size", 299))
        drop_rate = float(args.get("drop_rate", 0.0))
        tta_flips = bool(args.get("eval_tta_flips", False))

        model = create_model(model_name, drop_rate=drop_rate)
        model.load_state_dict(checkpoint["model_state"])
        model = model.to(device=self.device, memory_format=torch.channels_last)
        model.eval()

        validation_accuracy = float(checkpoint.get("val_metrics", {}).get("video_accuracy", 0.0))
        return ModelBundle(
            model_name=model_name,
            run_dir=checkpoint_path.parent,
            checkpoint_path=checkpoint_path.resolve(),
            image_size=image_size,
            drop_rate=drop_rate,
            tta_flips=tta_flips,
            weight=float(model_config["weight"]),
            best_epoch=int(checkpoint.get("epoch", -1)),
            validation_video_accuracy=validation_accuracy,
            model=model,
            transform=build_eval_transform(image_size),
        )

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"

    def _predict_single_model(self, bundle: ModelBundle, face_paths: list[str], batch_size: int = 8) -> ModelPrediction:
        frame_probabilities: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(face_paths), batch_size):
                batch_paths = face_paths[start : start + batch_size]
                tensors = []
                for image_path in batch_paths:
                    image = Image.open(image_path).convert("RGB")
                    tensors.append(bundle.transform(image))
                batch = torch.stack(tensors).to(self.device, non_blocking=True, memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                    logits = bundle.model(batch)
                    probabilities = logits_to_probabilities(logits)
                    if bundle.tta_flips:
                        flipped_logits = bundle.model(torch.flip(batch, dims=[3]))
                        probabilities = (probabilities + logits_to_probabilities(flipped_logits)) / 2.0
                frame_probabilities.extend(float(value) for value in probabilities.detach().cpu().tolist())

        video_probability = float(np.mean(frame_probabilities))
        return ModelPrediction(
            model_name=bundle.model_name,
            weight=bundle.weight,
            image_size=bundle.image_size,
            tta_flips=bundle.tta_flips,
            best_epoch=bundle.best_epoch,
            validation_video_accuracy=bundle.validation_video_accuracy,
            frame_probabilities=frame_probabilities,
            video_probability=video_probability,
        )

    def predict_video(
        self,
        video_path: Path,
        *,
        progress_callback: Callable[[str, float | None], None] | None = None,
    ) -> dict[str, object]:
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError(f"Unsupported video format: {video_path.suffix}")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = self.runtime_root / "predictions" / f"prediction_{timestamp}_{int(time.time() * 1000) % 1000000:06d}"
        upload_dir = run_dir / "upload"
        faces_dir = run_dir / "faces"
        run_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback is not None:
            progress_callback("Preparing upload for inference", 0.02)
        stored_video_path = _copy_video_to_runtime(video_path, upload_dir)

        extraction = extract_faces_from_video(
            stored_video_path,
            faces_dir,
            self.mtcnn,
            images_per_sample=TARGET_IMAGES_PER_SAMPLE,
            progress_callback=progress_callback,
        )
        face_paths = [str(path) for path in extraction["face_paths"]]

        model_predictions: list[ModelPrediction] = []
        for index, bundle in enumerate(self.models, start=1):
            if progress_callback is not None:
                progress_callback(f"Running {bundle.model_name} branch ({index}/{len(self.models)})", 0.58 + 0.18 * index)
            model_predictions.append(self._predict_single_model(bundle, face_paths))

        ensemble_frame_probabilities = [
            float(sum(model.weight * model.frame_probabilities[frame_index] for model in model_predictions))
            for frame_index in range(len(face_paths))
        ]
        ensemble_video_probability = float(np.mean(ensemble_frame_probabilities))
        predicted_label = "Deepfake" if ensemble_video_probability >= 0.5 else "Real"
        confidence = ensemble_video_probability if predicted_label == "Deepfake" else 1.0 - ensemble_video_probability

        result = {
            "video_path": str(stored_video_path),
            "run_dir": str(run_dir),
            "device": str(self.device),
            "device_name": self.device_name,
            "ensemble_validation_accuracy": self.ensemble_validation_accuracy,
            "ensemble_test_accuracy": self.ensemble_test_accuracy,
            "face_extraction": extraction,
            "model_predictions": [asdict(prediction) for prediction in model_predictions],
            "ensemble": {
                "frame_probabilities": ensemble_frame_probabilities,
                "video_probability": ensemble_video_probability,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "decision_threshold": 0.5,
            },
        }

        (run_dir / "prediction.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if progress_callback is not None:
            progress_callback("Prediction complete", 1.0)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the best deepfake ensemble on an input video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = DeepfakeEnsemblePredictor(runtime_root=args.runtime_root, require_cuda=not args.allow_cpu)
    result = predictor.predict_video(args.video)
    print(json.dumps(result["ensemble"], indent=2))
    print(f"Saved full inference report to {Path(result['run_dir']) / 'prediction.json'}")


if __name__ == "__main__":
    main()
