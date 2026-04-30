from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_merged_classifier import (
    save_classification_report,
    save_confusion_matrix_plot,
    save_probability_histogram,
    write_csv,
)


@dataclass(frozen=True)
class VideoSampleRecord:
    sample_id: str
    label: int
    split: str
    source_dataset: str
    domain_index: int
    processed_dir: str
    image_paths: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a multi-branch generalized deepfake classifier with global and local facial regions."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/merged_generalized_deepfake_1500"))
    parser.add_argument("--global-model", type=str, default="xception")
    parser.add_argument("--local-model", type=str, default="resnet18")
    parser.add_argument("--hidden-dim", type=int, default=320)
    parser.add_argument("--global-image-size", type=int, default=224)
    parser.add_argument("--local-image-size", type=int, default=112)
    parser.add_argument("--train-frames", type=int, default=8)
    parser.add_argument("--eval-frames", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.35)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--aux-loss-weight", type=float, default=0.18)
    parser.add_argument("--domain-loss-weight", type=float, default=0.03)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_manifest(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1:
        return [0]
    return [int(round(value)) for value in np.linspace(0, length - 1, count)]


def build_domain_mapping(*manifest_groups: list[dict[str, str]]) -> dict[str, int]:
    names = sorted({row["source_dataset"] for group in manifest_groups for row in group})
    return {name: index for index, name in enumerate(names)}


def load_video_sample_records(
    manifest_rows: list[dict[str, str]],
    domain_mapping: dict[str, int],
    *,
    sample_limit: int | None = None,
) -> list[VideoSampleRecord]:
    records: list[VideoSampleRecord] = []
    for row in manifest_rows:
        sample_dir = Path(row["processed_dir"])
        image_paths = tuple(
            str(path.resolve())
            for path in sorted(sample_dir.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        records.append(
            VideoSampleRecord(
                sample_id=row["sample_id"],
                label=1 if row["label"] == "fake" else 0,
                split=row["split"],
                source_dataset=row["source_dataset"],
                domain_index=domain_mapping[row["source_dataset"]],
                processed_dir=str(sample_dir.resolve()),
                image_paths=image_paths,
            )
        )
    if sample_limit is not None:
        records = records[:sample_limit]
    return records


def make_balanced_sampler(records: list[VideoSampleRecord]) -> WeightedRandomSampler:
    groups = Counter((record.label, record.source_dataset) for record in records)
    weights = [1.0 / groups[(record.label, record.source_dataset)] for record in records]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


class MultiBranchVideoDataset(Dataset):
    def __init__(
        self,
        records: list[VideoSampleRecord],
        *,
        global_image_size: int,
        local_image_size: int,
        num_frames: int,
        training: bool,
    ) -> None:
        self.records = records
        self.global_image_size = global_image_size
        self.local_image_size = local_image_size
        self.num_frames = num_frames
        self.training = training
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.color_jitter = transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02)

    def __len__(self) -> int:
        return len(self.records)

    def select_paths(self, image_paths: tuple[str, ...]) -> list[str]:
        if len(image_paths) >= self.num_frames:
            if self.training:
                indices = sorted(random.sample(range(len(image_paths)), self.num_frames))
            else:
                indices = evenly_spaced_indices(len(image_paths), self.num_frames)
        else:
            indices = evenly_spaced_indices(len(image_paths), self.num_frames)
        return [image_paths[index] for index in indices]

    def augment_face_image(self, image: Image.Image) -> Image.Image:
        if not self.training:
            return image

        width, height = image.size
        top, left, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
            image,
            scale=(0.86, 1.0),
            ratio=(0.95, 1.05),
        )
        image = TF.resized_crop(image, top, left, crop_h, crop_w, size=[height, width], antialias=True)

        if random.random() < 0.5:
            image = TF.hflip(image)
        if random.random() < 0.8:
            image = self.color_jitter(image)
        if random.random() < 0.25:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.9)))
        return image

    @staticmethod
    def crop_region(image: Image.Image, region: str) -> Image.Image:
        width, height = image.size
        if region == "eyes":
            box = (
                int(width * 0.12),
                int(height * 0.16),
                int(width * 0.88),
                int(height * 0.48),
            )
        elif region == "mouth":
            box = (
                int(width * 0.18),
                int(height * 0.56),
                int(width * 0.82),
                int(height * 0.92),
            )
        else:
            raise ValueError(f"Unsupported region: {region}")
        return image.crop(box)

    def tensorize(self, image: Image.Image, size: int) -> torch.Tensor:
        tensor = TF.to_tensor(TF.resize(image, [size, size], antialias=True))
        return self.normalize(tensor)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        selected_paths = self.select_paths(record.image_paths)

        full_frames: list[torch.Tensor] = []
        eye_frames: list[torch.Tensor] = []
        mouth_frames: list[torch.Tensor] = []

        for image_path in selected_paths:
            image = Image.open(image_path).convert("RGB")
            image = self.augment_face_image(image)
            eye_crop = self.crop_region(image, "eyes")
            mouth_crop = self.crop_region(image, "mouth")
            full_frames.append(self.tensorize(image, self.global_image_size))
            eye_frames.append(self.tensorize(eye_crop, self.local_image_size))
            mouth_frames.append(self.tensorize(mouth_crop, self.local_image_size))

        return {
            "sample_id": record.sample_id,
            "source_dataset": record.source_dataset,
            "full_frames": torch.stack(full_frames),
            "eye_frames": torch.stack(eye_frames),
            "mouth_frames": torch.stack(mouth_frames),
            "label": torch.tensor(record.label, dtype=torch.float32),
            "domain_index": torch.tensor(record.domain_index, dtype=torch.long),
        }


class GradientReverseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = lambda_value
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


class GradientReversalLayer(nn.Module):
    def forward(self, tensor: torch.Tensor, lambda_value: float) -> torch.Tensor:
        return GradientReverseFunction.apply(tensor, lambda_value)


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        reduced_dim = max(hidden_dim // 2, 32)
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, reduced_dim),
            nn.GELU(),
            nn.Linear(reduced_dim, 1),
        )

    def forward(self, frame_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.attention(frame_features).squeeze(-1)
        attention_weights = torch.softmax(attention_logits, dim=1)
        pooled = torch.sum(attention_weights.unsqueeze(-1) * frame_features, dim=1)
        return pooled, attention_weights


def create_feature_backbone(model_name: str) -> tuple[nn.Module, int]:
    try:
        model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
    except Exception:
        model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
    feature_dim = getattr(model, "num_features", None)
    if feature_dim is None:
        raise RuntimeError(f"Unable to determine feature dimension for backbone {model_name}.")
    return model, int(feature_dim)


def build_projection_head(input_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(hidden_dim),
    )


def build_classification_head(hidden_dim: int, dropout: float) -> nn.Module:
    reduced_dim = max(hidden_dim // 2, 32)
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, reduced_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(reduced_dim, 1),
    )


class MultiBranchTemporalClassifier(nn.Module):
    def __init__(
        self,
        *,
        global_model_name: str,
        local_model_name: str,
        hidden_dim: int,
        dropout: float,
        num_domains: int,
    ) -> None:
        super().__init__()
        self.global_backbone, global_feature_dim = create_feature_backbone(global_model_name)
        self.local_backbone, local_feature_dim = create_feature_backbone(local_model_name)

        self.global_projection = build_projection_head(global_feature_dim, hidden_dim, dropout)
        self.eye_projection = build_projection_head(local_feature_dim, hidden_dim, dropout)
        self.mouth_projection = build_projection_head(local_feature_dim, hidden_dim, dropout)

        self.branch_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        self.global_pool = TemporalAttentionPool(hidden_dim)
        self.eye_pool = TemporalAttentionPool(hidden_dim)
        self.mouth_pool = TemporalAttentionPool(hidden_dim)
        self.fused_pool = TemporalAttentionPool(hidden_dim)

        self.global_head = build_classification_head(hidden_dim, dropout)
        self.eye_head = build_classification_head(hidden_dim, dropout)
        self.mouth_head = build_classification_head(hidden_dim, dropout)
        self.fused_head = build_classification_head(hidden_dim, dropout)

        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 32), num_domains),
        )

    @staticmethod
    def _reshape_frames(frames: torch.Tensor) -> tuple[int, int, torch.Tensor]:
        batch_size, num_frames, channels, height, width = frames.shape
        flattened = frames.reshape(batch_size * num_frames, channels, height, width).contiguous(
            memory_format=torch.channels_last
        )
        return batch_size, num_frames, flattened

    def encode_frames(self, frames: torch.Tensor, backbone: nn.Module, projection: nn.Module) -> torch.Tensor:
        batch_size, num_frames, flattened = self._reshape_frames(frames)
        features = backbone(flattened)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        features = projection(features)
        return features.view(batch_size, num_frames, -1)

    def forward(
        self,
        full_frames: torch.Tensor,
        eye_frames: torch.Tensor,
        mouth_frames: torch.Tensor,
        *,
        domain_lambda: float = 0.0,
        fusion_override: tuple[float, float, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        global_embeddings = self.encode_frames(full_frames, self.global_backbone, self.global_projection)
        eye_embeddings = self.encode_frames(eye_frames, self.local_backbone, self.eye_projection)
        mouth_embeddings = self.encode_frames(mouth_frames, self.local_backbone, self.mouth_projection)

        if fusion_override is None:
            gate_logits = self.branch_gate(torch.cat([global_embeddings, eye_embeddings, mouth_embeddings], dim=-1))
            branch_weights = torch.softmax(gate_logits, dim=-1)
        else:
            weights = torch.tensor(fusion_override, device=global_embeddings.device, dtype=global_embeddings.dtype)
            weights = weights / weights.sum().clamp_min(1e-6)
            branch_weights = weights.view(1, 1, 3).expand(global_embeddings.size(0), global_embeddings.size(1), -1)

        fused_embeddings = (
            branch_weights[..., 0:1] * global_embeddings
            + branch_weights[..., 1:2] * eye_embeddings
            + branch_weights[..., 2:3] * mouth_embeddings
        )

        global_video, global_temporal = self.global_pool(global_embeddings)
        eye_video, eye_temporal = self.eye_pool(eye_embeddings)
        mouth_video, mouth_temporal = self.mouth_pool(mouth_embeddings)
        fused_video, fused_temporal = self.fused_pool(fused_embeddings)

        logits = {
            "global": self.global_head(global_video).squeeze(-1),
            "eye": self.eye_head(eye_video).squeeze(-1),
            "mouth": self.mouth_head(mouth_video).squeeze(-1),
            "fused": self.fused_head(fused_video).squeeze(-1),
        }
        domain_logits = self.domain_head(self.grl(fused_video, domain_lambda))

        return {
            "logits": logits,
            "domain_logits": domain_logits,
            "branch_weights": branch_weights,
            "global_temporal": global_temporal,
            "eye_temporal": eye_temporal,
            "mouth_temporal": mouth_temporal,
            "fused_temporal": fused_temporal,
        }


def safe_auc(labels: list[int], probabilities: list[float]) -> float:
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    finite_mask = np.isfinite(probabilities_array)
    if not finite_mask.all():
        probabilities_array = probabilities_array[finite_mask]
        labels_array = labels_array[finite_mask]
    if probabilities_array.size == 0:
        return 0.0
    unique_labels = set(labels_array.tolist())
    if len(unique_labels) < 2:
        return 0.0
    return float(roc_auc_score(labels_array, probabilities_array))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_binary_metrics(
    labels: list[int],
    probabilities: list[float],
    *,
    source_names: list[str] | None = None,
) -> dict[str, Any]:
    predictions = [1 if value >= 0.5 else 0 for value in probabilities]
    metrics: dict[str, Any] = {
        "video_accuracy": float(accuracy_score(labels, predictions)),
        "video_f1": float(f1_score(labels, predictions, zero_division=0)),
        "video_auc": safe_auc(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(labels, predictions, labels=[0, 1], output_dict=True, zero_division=0),
    }

    if source_names is not None:
        source_metrics: dict[str, dict[str, Any]] = {}
        for source_name in sorted(set(source_names)):
            indices = [index for index, value in enumerate(source_names) if value == source_name]
            source_labels = [labels[index] for index in indices]
            source_probabilities = [probabilities[index] for index in indices]
            source_predictions = [predictions[index] for index in indices]
            source_metrics[source_name] = {
                "count": len(indices),
                "accuracy": float(accuracy_score(source_labels, source_predictions)),
                "f1": float(f1_score(source_labels, source_predictions, zero_division=0)),
                "auc": safe_auc(source_labels, source_probabilities),
            }
        metrics["source_metrics"] = source_metrics
    return metrics


def branch_probabilities_to_mode(
    payload: dict[str, list[Any]],
    mode_weights: tuple[float, float, float] | None,
) -> np.ndarray:
    if mode_weights is None:
        return np.array(payload["fused_probs"], dtype=np.float64)
    branch_matrix = np.stack(
        [
            np.array(payload["global_probs"], dtype=np.float64),
            np.array(payload["eye_probs"], dtype=np.float64),
            np.array(payload["mouth_probs"], dtype=np.float64),
        ],
        axis=1,
    )
    normalized = np.array(mode_weights, dtype=np.float64)
    normalized = normalized / max(normalized.sum(), 1e-6)
    return branch_matrix @ normalized


def generate_weight_grid(step: float = 0.1) -> list[tuple[float, float, float]]:
    values = [round(index * step, 10) for index in range(int(round(1.0 / step)) + 1)]
    grid: list[tuple[float, float, float]] = []
    for global_weight in values:
        for eye_weight in values:
            mouth_weight = round(1.0 - global_weight - eye_weight, 10)
            if mouth_weight < 0.0 or mouth_weight > 1.0:
                continue
            grid.append((global_weight, eye_weight, mouth_weight))
    return sorted(set(grid))


def label_smooth(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return labels
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def domain_lambda_at_progress(progress: float) -> float:
    return float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


def run_training_epoch(
    model: MultiBranchTemporalClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    epoch_index: int,
    total_epochs: int,
    grad_accum_steps: int,
    aux_loss_weight: float,
    domain_loss_weight: float,
    label_smoothing: float,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train(True)
    optimizer.zero_grad(set_to_none=True)

    total_examples = 0
    total_correct = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_domain_loss = 0.0

    for step_index, batch in enumerate(loader, start=1):
        full_frames = batch["full_frames"].to(device, non_blocking=True)
        eye_frames = batch["eye_frames"].to(device, non_blocking=True)
        mouth_frames = batch["mouth_frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        smoothed_labels = label_smooth(labels, label_smoothing)
        domain_indices = batch["domain_index"].to(device, non_blocking=True)

        progress = ((epoch_index - 1) * len(loader) + (step_index - 1)) / max(total_epochs * len(loader), 1)
        domain_lambda = domain_lambda_at_progress(progress)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(
                full_frames,
                eye_frames,
                mouth_frames,
                domain_lambda=domain_lambda,
            )
            fused_loss = F.binary_cross_entropy_with_logits(outputs["logits"]["fused"], smoothed_labels)
            aux_loss = torch.stack(
                [
                    F.binary_cross_entropy_with_logits(outputs["logits"]["global"], smoothed_labels),
                    F.binary_cross_entropy_with_logits(outputs["logits"]["eye"], smoothed_labels),
                    F.binary_cross_entropy_with_logits(outputs["logits"]["mouth"], smoothed_labels),
                ]
            ).mean()
            domain_loss = F.cross_entropy(outputs["domain_logits"], domain_indices)
            loss = fused_loss + aux_loss_weight * aux_loss + domain_loss_weight * domain_lambda * domain_loss

        scaler.scale(loss / grad_accum_steps).backward()

        if step_index % grad_accum_steps == 0 or step_index == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        probabilities = torch.nan_to_num(torch.sigmoid(outputs["logits"]["fused"]), nan=0.5, posinf=1.0, neginf=0.0)
        predictions = (probabilities >= 0.5).float()
        batch_size = labels.size(0)

        total_examples += batch_size
        total_correct += (predictions == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_cls_loss += (fused_loss.item() + aux_loss_weight * aux_loss.item()) * batch_size
        total_domain_loss += domain_loss.item() * batch_size

    return {
        "train_total_loss": total_loss / total_examples,
        "train_classification_loss": total_cls_loss / total_examples,
        "train_domain_loss": total_domain_loss / total_examples,
        "train_accuracy": total_correct / total_examples,
    }


def evaluate_model(
    model: MultiBranchTemporalClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    aux_loss_weight: float,
    domain_loss_weight: float,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    model.train(False)

    total_examples = 0
    total_loss = 0.0
    total_cls_loss = 0.0
    total_domain_loss = 0.0

    payload: dict[str, list[Any]] = {
        "sample_ids": [],
        "source_datasets": [],
        "video_labels": [],
        "fused_probs": [],
        "global_probs": [],
        "eye_probs": [],
        "mouth_probs": [],
        "learned_branch_weights": [],
        "temporal_weights": [],
    }

    with torch.no_grad():
        for batch in loader:
            full_frames = batch["full_frames"].to(device, non_blocking=True)
            eye_frames = batch["eye_frames"].to(device, non_blocking=True)
            mouth_frames = batch["mouth_frames"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            domain_indices = batch["domain_index"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(
                    full_frames,
                    eye_frames,
                    mouth_frames,
                    domain_lambda=1.0,
                )
                fused_loss = F.binary_cross_entropy_with_logits(outputs["logits"]["fused"], labels)
                aux_loss = torch.stack(
                    [
                        F.binary_cross_entropy_with_logits(outputs["logits"]["global"], labels),
                        F.binary_cross_entropy_with_logits(outputs["logits"]["eye"], labels),
                        F.binary_cross_entropy_with_logits(outputs["logits"]["mouth"], labels),
                    ]
                ).mean()
                domain_loss = F.cross_entropy(outputs["domain_logits"], domain_indices)
                loss = fused_loss + aux_loss_weight * aux_loss + domain_loss_weight * domain_loss

            batch_size = labels.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            total_cls_loss += (fused_loss.item() + aux_loss_weight * aux_loss.item()) * batch_size
            total_domain_loss += domain_loss.item() * batch_size

            payload["sample_ids"].extend(batch["sample_id"])
            payload["source_datasets"].extend(batch["source_dataset"])
            payload["video_labels"].extend(labels.int().cpu().tolist())
            payload["fused_probs"].extend(
                torch.nan_to_num(torch.sigmoid(outputs["logits"]["fused"]), nan=0.5, posinf=1.0, neginf=0.0).cpu().tolist()
            )
            payload["global_probs"].extend(
                torch.nan_to_num(torch.sigmoid(outputs["logits"]["global"]), nan=0.5, posinf=1.0, neginf=0.0).cpu().tolist()
            )
            payload["eye_probs"].extend(
                torch.nan_to_num(torch.sigmoid(outputs["logits"]["eye"]), nan=0.5, posinf=1.0, neginf=0.0).cpu().tolist()
            )
            payload["mouth_probs"].extend(
                torch.nan_to_num(torch.sigmoid(outputs["logits"]["mouth"]), nan=0.5, posinf=1.0, neginf=0.0).cpu().tolist()
            )
            payload["learned_branch_weights"].extend(
                torch.nan_to_num(outputs["branch_weights"].mean(dim=1), nan=1.0 / 3.0, posinf=1.0, neginf=0.0).cpu().tolist()
            )
            payload["temporal_weights"].extend(
                torch.nan_to_num(outputs["fused_temporal"], nan=1.0 / outputs["fused_temporal"].shape[1], posinf=1.0, neginf=0.0)
                .cpu()
                .tolist()
            )

    metrics = compute_binary_metrics(
        payload["video_labels"],
        payload["fused_probs"],
        source_names=payload["source_datasets"],
    )
    metrics.update(
        {
            "val_total_loss": total_loss / total_examples,
            "val_classification_loss": total_cls_loss / total_examples,
            "val_domain_loss": total_domain_loss / total_examples,
            "avg_branch_weights": {
                "global": float(np.mean([weights[0] for weights in payload["learned_branch_weights"]])),
                "eye": float(np.mean([weights[1] for weights in payload["learned_branch_weights"]])),
                "mouth": float(np.mean([weights[2] for weights in payload["learned_branch_weights"]])),
            },
            "branch_metrics": {
                branch: compute_binary_metrics(
                    payload["video_labels"],
                    payload[f"{branch}_probs"],
                    source_names=payload["source_datasets"],
                )
                for branch in ("global", "eye", "mouth")
            },
        }
    )
    return metrics, payload


def evaluate_feature_modes(
    val_payload: dict[str, list[Any]],
    test_payload: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[float, float, float]]:
    fixed_modes: dict[str, tuple[float, float, float] | None] = {
        "learned_fused": None,
        "global_only": (1.0, 0.0, 0.0),
        "eye_only": (0.0, 1.0, 0.0),
        "mouth_only": (0.0, 0.0, 1.0),
        "global_eye_70_30": (0.7, 0.3, 0.0),
        "global_mouth_70_30": (0.7, 0.0, 0.3),
        "global_local_equal": (0.5, 0.25, 0.25),
        "equal_all": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    }

    mode_rows: list[dict[str, Any]] = []
    for mode_name, weights in fixed_modes.items():
        val_metrics = compute_binary_metrics(
            val_payload["video_labels"],
            branch_probabilities_to_mode(val_payload, weights).tolist(),
            source_names=val_payload["source_datasets"],
        )
        test_metrics = compute_binary_metrics(
            test_payload["video_labels"],
            branch_probabilities_to_mode(test_payload, weights).tolist(),
            source_names=test_payload["source_datasets"],
        )
        mode_rows.append(
            {
                "mode": mode_name,
                "weights": "learned" if weights is None else json.dumps([round(value, 3) for value in weights]),
                "validation_video_accuracy": val_metrics["video_accuracy"],
                "validation_video_f1": val_metrics["video_f1"],
                "validation_video_auc": val_metrics["video_auc"],
                "test_video_accuracy": test_metrics["video_accuracy"],
                "test_video_f1": test_metrics["video_f1"],
                "test_video_auc": test_metrics["video_auc"],
            }
        )

    grid_rows: list[dict[str, Any]] = []
    best_weights = (1.0, 0.0, 0.0)
    best_key = (-1.0, -1.0)
    for weights in generate_weight_grid(step=0.1):
        val_probs = branch_probabilities_to_mode(val_payload, weights).tolist()
        val_metrics = compute_binary_metrics(
            val_payload["video_labels"],
            val_probs,
            source_names=val_payload["source_datasets"],
        )
        grid_rows.append(
            {
                "weights": json.dumps([round(value, 3) for value in weights]),
                "validation_video_accuracy": val_metrics["video_accuracy"],
                "validation_video_f1": val_metrics["video_f1"],
                "validation_video_auc": val_metrics["video_auc"],
            }
        )
        comparison_key = (val_metrics["video_accuracy"], val_metrics["video_auc"])
        if comparison_key > best_key:
            best_key = comparison_key
            best_weights = weights

    best_val_metrics = compute_binary_metrics(
        val_payload["video_labels"],
        branch_probabilities_to_mode(val_payload, best_weights).tolist(),
        source_names=val_payload["source_datasets"],
    )
    best_test_metrics = compute_binary_metrics(
        test_payload["video_labels"],
        branch_probabilities_to_mode(test_payload, best_weights).tolist(),
        source_names=test_payload["source_datasets"],
    )
    mode_rows.append(
        {
            "mode": "best_val_grid",
            "weights": json.dumps([round(value, 3) for value in best_weights]),
            "validation_video_accuracy": best_val_metrics["video_accuracy"],
            "validation_video_f1": best_val_metrics["video_f1"],
            "validation_video_auc": best_val_metrics["video_auc"],
            "test_video_accuracy": best_test_metrics["video_accuracy"],
            "test_video_f1": best_test_metrics["video_f1"],
            "test_video_auc": best_test_metrics["video_auc"],
        }
    )
    mode_rows.sort(key=lambda row: row["test_video_accuracy"], reverse=True)
    grid_rows.sort(key=lambda row: (row["validation_video_accuracy"], row["validation_video_auc"]), reverse=True)
    return mode_rows, grid_rows, best_weights


def save_history_plot(history: list[dict[str, float]], run_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [row["train_total_loss"] for row in history], marker="o", label="Train Total Loss")
    axes[0].plot(epochs, [row["val_total_loss"] for row in history], marker="o", label="Val Total Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, [row["val_video_accuracy"] for row in history], marker="o", label="Val Accuracy")
    axes[1].plot(epochs, [row["val_video_auc"] for row in history], marker="o", label="Val AUC")
    axes[1].set_title("Accuracy / AUC Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "loss_accuracy_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_mode_plot(mode_rows: list[dict[str, Any]], run_dir: Path) -> None:
    names = [row["mode"] for row in mode_rows]
    val_scores = [row["validation_video_accuracy"] for row in mode_rows]
    test_scores = [row["test_video_accuracy"] for row in mode_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(names))
    ax.bar(positions - 0.18, val_scores, width=0.36, label="Validation")
    ax.bar(positions + 0.18, test_scores, width=0.36, label="Test")
    ax.set_xticks(positions, names, rotation=30, ha="right")
    ax.set_ylabel("Video Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Feature Weight Mode Comparison")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "feature_mode_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_branch_weight_plot(avg_weights: dict[str, float], run_dir: Path) -> None:
    names = list(avg_weights.keys())
    values = list(avg_weights.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color=["#4C78A8", "#F58518", "#E45756"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average Learned Weight")
    ax.set_title("Average Branch Contribution")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "branch_contributions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_prediction_csv(payload: dict[str, list[Any]], run_dir: Path, filename: str) -> None:
    rows: list[dict[str, Any]] = []
    for sample_id, source_dataset, label, fused_prob, global_prob, eye_prob, mouth_prob, branch_weights in zip(
        payload["sample_ids"],
        payload["source_datasets"],
        payload["video_labels"],
        payload["fused_probs"],
        payload["global_probs"],
        payload["eye_probs"],
        payload["mouth_probs"],
        payload["learned_branch_weights"],
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": source_dataset,
                "ground_truth": "fake" if label == 1 else "real",
                "predicted_label": "fake" if fused_prob >= 0.5 else "real",
                "fused_probability": f"{fused_prob:.6f}",
                "global_probability": f"{global_prob:.6f}",
                "eye_probability": f"{eye_prob:.6f}",
                "mouth_probability": f"{mouth_prob:.6f}",
                "learned_global_weight": f"{branch_weights[0]:.6f}",
                "learned_eye_weight": f"{branch_weights[1]:.6f}",
                "learned_mouth_weight": f"{branch_weights[2]:.6f}",
            }
        )
    write_csv(run_dir / filename, rows)


def save_source_metrics_csv(metrics: dict[str, Any], run_dir: Path, filename: str) -> None:
    rows = []
    for source_name, source_metrics in metrics.get("source_metrics", {}).items():
        rows.append({"source_dataset": source_name, **source_metrics})
    write_csv(run_dir / filename, rows)


def save_branch_contribution_csv(payload: dict[str, list[Any]], run_dir: Path, filename: str) -> None:
    rows = []
    for sample_id, source_dataset, weights in zip(
        payload["sample_ids"], payload["source_datasets"], payload["learned_branch_weights"]
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": source_dataset,
                "global_weight": f"{weights[0]:.6f}",
                "eye_weight": f"{weights[1]:.6f}",
                "mouth_weight": f"{weights[2]:.6f}",
            }
        )
    write_csv(run_dir / filename, rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training run, but no active CUDA device was found.")

    run_name = f"multibranch_{args.global_model.replace('/', '-')}_{args.local_model.replace('/', '-')}_{int(time.time())}"
    run_dir = dataset_root / "training_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(dataset_root / "manifests" / "train.csv")
    val_rows = load_manifest(dataset_root / "manifests" / "val.csv")
    test_rows = load_manifest(dataset_root / "manifests" / "test.csv")
    domain_mapping = build_domain_mapping(train_rows, val_rows, test_rows)
    domain_names = {index: name for name, index in domain_mapping.items()}

    train_records = load_video_sample_records(train_rows, domain_mapping, sample_limit=args.limit_train_samples)
    val_records = load_video_sample_records(val_rows, domain_mapping, sample_limit=args.limit_val_samples)
    test_records = load_video_sample_records(test_rows, domain_mapping, sample_limit=args.limit_test_samples)

    train_dataset = MultiBranchVideoDataset(
        train_records,
        global_image_size=args.global_image_size,
        local_image_size=args.local_image_size,
        num_frames=args.train_frames,
        training=True,
    )
    val_dataset = MultiBranchVideoDataset(
        val_records,
        global_image_size=args.global_image_size,
        local_image_size=args.local_image_size,
        num_frames=args.eval_frames,
        training=False,
    )
    test_dataset = MultiBranchVideoDataset(
        test_records,
        global_image_size=args.global_image_size,
        local_image_size=args.local_image_size,
        num_frames=args.eval_frames,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_records),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    model = MultiBranchTemporalClassifier(
        global_model_name=args.global_model,
        local_model_name=args.local_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_domains=len(domain_mapping),
    ).to(device)

    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if "backbone" in name:
            backbone_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    optimizer = AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr * args.backbone_lr_scale},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr / 50)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_accuracy = -math.inf
    best_epoch = -1
    best_history_row: dict[str, float] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_training_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch_index=epoch,
            total_epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps,
            aux_loss_weight=args.aux_loss_weight,
            domain_loss_weight=args.domain_loss_weight,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics, _ = evaluate_model(
            model,
            val_loader,
            device,
            aux_loss_weight=args.aux_loss_weight,
            domain_loss_weight=args.domain_loss_weight,
        )
        scheduler.step()

        history_row = {
            "epoch": epoch,
            **train_metrics,
            "val_total_loss": float(val_metrics["val_total_loss"]),
            "val_classification_loss": float(val_metrics["val_classification_loss"]),
            "val_domain_loss": float(val_metrics["val_domain_loss"]),
            "val_video_accuracy": float(val_metrics["video_accuracy"]),
            "val_video_f1": float(val_metrics["video_f1"]),
            "val_video_auc": float(val_metrics["video_auc"]),
            "learning_rate": float(optimizer.param_groups[-1]["lr"]),
        }
        history.append(history_row)
        print(json.dumps(history_row, indent=2))

        if val_metrics["video_accuracy"] > best_val_accuracy:
            best_val_accuracy = float(val_metrics["video_accuracy"])
            best_epoch = epoch
            best_history_row = history_row
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "domain_mapping": domain_mapping,
                    "val_metrics": val_metrics,
                },
                run_dir / "best_model.pth",
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": history[-1]["epoch"],
            "domain_mapping": domain_mapping,
        },
        run_dir / "last_model.pth",
    )

    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    val_metrics, val_payload = evaluate_model(
        model,
        val_loader,
        device,
        aux_loss_weight=args.aux_loss_weight,
        domain_loss_weight=args.domain_loss_weight,
    )
    test_metrics, test_payload = evaluate_model(
        model,
        test_loader,
        device,
        aux_loss_weight=args.aux_loss_weight,
        domain_loss_weight=args.domain_loss_weight,
    )
    mode_rows, grid_rows, best_weights = evaluate_feature_modes(val_payload, test_payload)

    history_rows = [
        {
            "epoch": row["epoch"],
            "train_total_loss": f"{row['train_total_loss']:.6f}",
            "train_classification_loss": f"{row['train_classification_loss']:.6f}",
            "train_domain_loss": f"{row['train_domain_loss']:.6f}",
            "train_accuracy": f"{row['train_accuracy']:.6f}",
            "val_total_loss": f"{row['val_total_loss']:.6f}",
            "val_classification_loss": f"{row['val_classification_loss']:.6f}",
            "val_domain_loss": f"{row['val_domain_loss']:.6f}",
            "val_video_accuracy": f"{row['val_video_accuracy']:.6f}",
            "val_video_f1": f"{row['val_video_f1']:.6f}",
            "val_video_auc": f"{row['val_video_auc']:.6f}",
            "learning_rate": f"{row['learning_rate']:.8f}",
        }
        for row in history
    ]
    write_csv(run_dir / "epoch_history.csv", history_rows)
    write_csv(run_dir / "feature_mode_comparison.csv", mode_rows)
    write_csv(run_dir / "feature_weight_grid.csv", grid_rows)
    save_history_plot(history, run_dir)
    save_mode_plot(mode_rows, run_dir)
    save_branch_weight_plot(val_metrics["avg_branch_weights"], run_dir)
    save_confusion_matrix_plot(val_metrics["confusion_matrix"], run_dir, "validation_confusion_matrix.png")
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], run_dir, "test_confusion_matrix.png")
    save_probability_histogram(
        {
            "video_labels": test_payload["video_labels"],
            "video_probs": test_payload["fused_probs"],
        },
        run_dir,
        "test_probability_histogram.png",
    )
    save_classification_report(val_metrics["classification_report"], run_dir, "validation_classification_report")
    save_classification_report(test_metrics["classification_report"], run_dir, "test_classification_report")
    save_prediction_csv(val_payload, run_dir, "validation_video_predictions.csv")
    save_prediction_csv(test_payload, run_dir, "test_video_predictions.csv")
    save_source_metrics_csv(val_metrics, run_dir, "validation_source_metrics.csv")
    save_source_metrics_csv(test_metrics, run_dir, "test_source_metrics.csv")
    save_branch_contribution_csv(val_payload, run_dir, "validation_branch_contributions.csv")
    save_branch_contribution_csv(test_payload, run_dir, "test_branch_contributions.csv")
    write_csv(
        run_dir / "results_summary.csv",
        [
            {
                "global_model": args.global_model,
                "local_model": args.local_model,
                "best_epoch": best_epoch,
                "validation_video_accuracy": f"{val_metrics['video_accuracy']:.6f}",
                "validation_video_f1": f"{val_metrics['video_f1']:.6f}",
                "validation_video_auc": f"{val_metrics['video_auc']:.6f}",
                "test_video_accuracy": f"{test_metrics['video_accuracy']:.6f}",
                "test_video_f1": f"{test_metrics['video_f1']:.6f}",
                "test_video_auc": f"{test_metrics['video_auc']:.6f}",
                "best_val_grid_weights": json.dumps([round(value, 3) for value in best_weights]),
                "avg_global_weight": f"{val_metrics['avg_branch_weights']['global']:.6f}",
                "avg_eye_weight": f"{val_metrics['avg_branch_weights']['eye']:.6f}",
                "avg_mouth_weight": f"{val_metrics['avg_branch_weights']['mouth']:.6f}",
            }
        ],
    )

    artifacts = {
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_history_row": best_history_row,
        "history": history,
        "domain_mapping": domain_names,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "feature_mode_rows": mode_rows,
        "best_val_grid_weights": best_weights,
    }
    (run_dir / "metrics.json").write_text(json.dumps(to_jsonable(artifacts), indent=2), encoding="utf-8")
    (run_dir / "classification_report.txt").write_text(
        "\n".join(
            [
                "Validation Classification Report",
                json.dumps(val_metrics["classification_report"], indent=2),
                "",
                "Test Classification Report",
                json.dumps(test_metrics["classification_report"], indent=2),
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "val_predictions.json").write_text(json.dumps(to_jsonable(val_payload), indent=2), encoding="utf-8")
    (run_dir / "test_predictions.json").write_text(json.dumps(to_jsonable(test_payload), indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "validation_video_accuracy": val_metrics["video_accuracy"],
                "test_video_accuracy": test_metrics["video_accuracy"],
                "best_val_grid_weights": best_weights,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
