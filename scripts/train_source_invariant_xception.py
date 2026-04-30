from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch import Tensor, nn
from torch.autograd import Function
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_merged_classifier import save_classification_report, save_confusion_matrix_plot, save_probability_histogram, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a source-invariant Xception classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("generalized_xception_source_invariant"))
    parser.add_argument("--model-name", type=str, default="xception")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--backbone-lr", type=float, default=1.0e-4)
    parser.add_argument("--head-lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--source-loss-weight", type=float, default=0.15)
    parser.add_argument("--manip-loss-weight", type=float, default=0.12)
    parser.add_argument("--contrastive-loss-weight", type=float, default=0.06)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--sampler-oversample-cap", type=float, default=5.0)
    parser.add_argument("--eval-tta-flips", action="store_true")
    parser.add_argument("--final-eval-tta-flips", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@dataclass(frozen=True)
class FrameRecord:
    image_path: str
    label: int
    sample_id: str
    split: str
    source_dataset: str
    source_index: int
    manipulation_type: str
    manipulation_index: int


class RandomJPEGCompression:
    def __init__(self, p: float = 0.35, quality_range: tuple[int, int] = (35, 95)) -> None:
        self.p = p
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(*self.quality_range))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownscaleUpscale:
    def __init__(self, p: float = 0.30, scale_range: tuple[float, float] = (0.45, 0.85)) -> None:
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        width, height = image.size
        factor = random.uniform(*self.scale_range)
        reduced = (max(32, int(width * factor)), max(32, int(height * factor)))
        return image.resize(reduced, Image.BILINEAR).resize((width, height), Image.BICUBIC)


class AddGaussianNoise:
    def __init__(self, p: float = 0.25, std_range: tuple[float, float] = (0.0, 0.04)) -> None:
        self.p = p
        self.std_range = std_range

    def __call__(self, tensor: Tensor) -> Tensor:
        if random.random() >= self.p:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * random.uniform(*self.std_range), 0.0, 1.0)


class GradientReversal(Function):
    @staticmethod
    def forward(ctx: Any, tensor: Tensor, lambda_value: float) -> Tensor:
        ctx.lambda_value = lambda_value
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return -ctx.lambda_value * grad_output, None


def grad_reverse(tensor: Tensor, lambda_value: float) -> Tensor:
    return GradientReversal.apply(tensor, lambda_value)


class SourceInvariantXception(nn.Module):
    def __init__(self, model_name: str, num_sources: int, num_manipulations: int, drop_rate: float) -> None:
        super().__init__()
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
                drop_rate=drop_rate,
            )
        except Exception:
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg",
                drop_rate=drop_rate,
            )
        feature_dim = int(getattr(self.backbone, "num_features"))
        self.classifier = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(drop_rate), nn.Linear(feature_dim, 1))
        self.source_head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(drop_rate * 0.5), nn.Linear(feature_dim, num_sources))
        self.manipulation_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(feature_dim, num_manipulations),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, 128),
        )

    def forward(self, images: Tensor, grl_lambda: float = 0.0) -> dict[str, Tensor]:
        features = self.backbone(images)
        return {
            "binary_logits": self.classifier(features).squeeze(1),
            "source_logits": self.source_head(grad_reverse(features, grl_lambda)),
            "manipulation_logits": self.manipulation_head(features),
            "projection": F.normalize(self.projection_head(features), dim=1),
        }


class FrameDataset(Dataset):
    def __init__(self, records: list[FrameRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, str, str, str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(record.label, dtype=torch.float32),
            torch.tensor(record.source_index, dtype=torch.long),
            torch.tensor(record.manipulation_index, dtype=torch.long),
            record.sample_id,
            record.source_dataset,
            record.manipulation_type,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_mappings(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[str, int]]:
    source_names = sorted({row["source_dataset"] for row in rows})
    manipulation_names = sorted({row["fake_subtype"] for row in rows if row["fake_subtype"]})
    return (
        {name: index for index, name in enumerate(source_names)},
        {name: index for index, name in enumerate(manipulation_names)},
    )


def load_frame_records(
    manifest_rows: list[dict[str, str]],
    source_to_index: dict[str, int],
    manipulation_to_index: dict[str, int],
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for row in manifest_rows:
        sample_dir = Path(row["processed_dir"])
        image_files = sorted(
            path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        label = int(row["binary_label"])
        manipulation_index = manipulation_to_index.get(row["fake_subtype"], -1)
        for image_path in image_files:
            records.append(
                FrameRecord(
                    image_path=str(image_path),
                    label=label,
                    sample_id=row["sample_id"],
                    split=row["split"],
                    source_dataset=row["source_dataset"],
                    source_index=source_to_index[row["source_dataset"]],
                    manipulation_type=row["manipulation_type"],
                    manipulation_index=manipulation_index,
                )
            )
    return records


def make_balanced_sampler(records: list[FrameRecord], oversample_cap: float) -> WeightedRandomSampler:
    group_counts = Counter((record.label, record.source_dataset) for record in records)
    major_counts = [count for (_, source_name), count in group_counts.items() if source_name != "Extras"]
    target_count = max(int(np.median(major_counts)) if major_counts else max(group_counts.values()), 1)
    floor_value = target_count / max(oversample_cap, 1.0)
    effective_counts = {key: float(max(count, floor_value)) for key, count in group_counts.items()}
    weights = [1.0 / effective_counts[(record.label, record.source_dataset)] for record in records]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomJPEGCompression(p=0.35, quality_range=(35, 95)),
            RandomDownscaleUpscale(p=0.30, scale_range=(0.45, 0.85)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.03),
            transforms.RandomGrayscale(p=0.08),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.20),
            transforms.ToTensor(),
            AddGaussianNoise(p=0.25, std_range=(0.0, 0.04)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def supervised_contrastive_loss(embeddings: Tensor, labels: Tensor, temperature: float) -> Tensor:
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    logits = torch.matmul(embeddings, embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~identity
    exp_logits = torch.exp(logits) * (~identity)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not valid.any():
        return embeddings.new_zeros(())
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1).float()
    return -mean_log_prob_pos[valid].mean()


def logits_to_probabilities(logits: Tensor) -> Tensor:
    return torch.sigmoid(logits)


def safe_auc(labels: list[int], probabilities: list[float]) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def compute_domain_lambda(epoch: int, total_epochs: int) -> float:
    progress = epoch / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def run_epoch(
    model: SourceInvariantXception,
    loader: DataLoader,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    args: argparse.Namespace,
    *,
    epoch: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    grl_lambda = compute_domain_lambda(epoch, args.epochs) if is_train else 0.0
    totals = defaultdict(float)
    total_examples = 0

    for images, labels, source_indices, manipulation_indices, *_ in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        source_indices = source_indices.to(device, non_blocking=True)
        manipulation_indices = manipulation_indices.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images, grl_lambda=grl_lambda if is_train else 0.0)
                bce_loss = F.binary_cross_entropy_with_logits(outputs["binary_logits"], labels)
                source_loss = F.cross_entropy(outputs["source_logits"], source_indices)
                fake_mask = manipulation_indices >= 0
                if fake_mask.any():
                    manipulation_loss = F.cross_entropy(
                        outputs["manipulation_logits"][fake_mask],
                        manipulation_indices[fake_mask],
                    )
                else:
                    manipulation_loss = outputs["binary_logits"].new_zeros(())
                contrastive_loss = supervised_contrastive_loss(
                    outputs["projection"],
                    labels.long(),
                    args.contrastive_temperature,
                )
                total_loss = (
                    bce_loss
                    + args.source_loss_weight * source_loss
                    + args.manip_loss_weight * manipulation_loss
                    + args.contrastive_loss_weight * contrastive_loss
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                assert scaler is not None
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probabilities = logits_to_probabilities(outputs["binary_logits"])
        predictions = (probabilities >= 0.5).float()
        batch_size = labels.size(0)
        totals["loss"] += total_loss.item() * batch_size
        totals["bce_loss"] += bce_loss.item() * batch_size
        totals["source_loss"] += source_loss.item() * batch_size
        totals["manipulation_loss"] += manipulation_loss.item() * batch_size
        totals["contrastive_loss"] += contrastive_loss.item() * batch_size
        totals["binary_accuracy"] += (predictions == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": totals["loss"] / total_examples,
        "bce_loss": totals["bce_loss"] / total_examples,
        "source_loss": totals["source_loss"] / total_examples,
        "manipulation_loss": totals["manipulation_loss"] / total_examples,
        "contrastive_loss": totals["contrastive_loss"] / total_examples,
        "binary_accuracy": totals["binary_accuracy"] / total_examples,
        "domain_lambda": grl_lambda,
    }


def evaluate(
    model: SourceInvariantXception,
    loader: DataLoader,
    device: torch.device,
    manipulation_names_by_index: dict[int, str],
    *,
    tta_flips: bool = False,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    model.eval()
    frame_targets: list[int] = []
    frame_probs: list[float] = []
    frame_sample_ids: list[str] = []
    frame_source_names: list[str] = []
    frame_manipulation_names: list[str] = []
    fake_manipulation_targets: list[str] = []
    fake_manipulation_predictions: list[str] = []

    with torch.no_grad():
        for images, labels, _, manipulation_indices, sample_ids, source_names, manipulation_names in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            outputs = model(images, grl_lambda=0.0)
            probs_tensor = logits_to_probabilities(outputs["binary_logits"])
            if tta_flips:
                flipped_outputs = model(torch.flip(images, dims=[3]), grl_lambda=0.0)
                probs_tensor = (probs_tensor + logits_to_probabilities(flipped_outputs["binary_logits"])) / 2.0

            frame_probs.extend(probs_tensor.detach().cpu().tolist())
            frame_targets.extend(labels.int().tolist())
            frame_sample_ids.extend(sample_ids)
            frame_source_names.extend(source_names)
            frame_manipulation_names.extend(manipulation_names)

            fake_mask = manipulation_indices >= 0
            if fake_mask.any():
                predicted_indices = outputs["manipulation_logits"][fake_mask.to(device)].argmax(dim=1).detach().cpu().tolist()
                target_indices = manipulation_indices[fake_mask].tolist()
                fake_manipulation_targets.extend([manipulation_names_by_index[index] for index in target_indices])
                fake_manipulation_predictions.extend([manipulation_names_by_index[index] for index in predicted_indices])

    frame_predictions = [1 if prob >= 0.5 else 0 for prob in frame_probs]
    metrics: dict[str, Any] = {
        "frame_accuracy": float(accuracy_score(frame_targets, frame_predictions)),
        "frame_f1": float(f1_score(frame_targets, frame_predictions)),
        "frame_auc": safe_auc(frame_targets, frame_probs),
    }

    video_scores: dict[str, list[float]] = defaultdict(list)
    video_targets: dict[str, int] = {}
    video_sources: dict[str, str] = {}
    video_manipulations: dict[str, str] = {}
    for sample_id, prob, target, source_name, manipulation_name in zip(
        frame_sample_ids,
        frame_probs,
        frame_targets,
        frame_source_names,
        frame_manipulation_names,
    ):
        video_scores[sample_id].append(prob)
        video_targets[sample_id] = target
        video_sources[sample_id] = source_name
        video_manipulations[sample_id] = manipulation_name

    video_ids = sorted(video_scores)
    video_probs = [float(np.mean(video_scores[sample_id])) for sample_id in video_ids]
    video_labels = [video_targets[sample_id] for sample_id in video_ids]
    video_source_names = [video_sources[sample_id] for sample_id in video_ids]
    video_manipulation_names = [video_manipulations[sample_id] for sample_id in video_ids]
    video_predictions = [1 if prob >= 0.5 else 0 for prob in video_probs]

    source_metrics: dict[str, dict[str, float]] = {}
    for source_name in sorted(set(video_source_names)):
        indices = [index for index, value in enumerate(video_source_names) if value == source_name]
        source_labels = [video_labels[index] for index in indices]
        source_probs = [video_probs[index] for index in indices]
        source_preds = [video_predictions[index] for index in indices]
        source_metrics[source_name] = {
            "count": len(indices),
            "accuracy": float(accuracy_score(source_labels, source_preds)),
            "auc": safe_auc(source_labels, source_probs),
        }

    fake_subtype_metrics: dict[str, dict[str, float]] = {}
    for manipulation_name in sorted(set(fake_manipulation_targets)):
        indices = [index for index, value in enumerate(fake_manipulation_targets) if value == manipulation_name]
        correct = sum(fake_manipulation_targets[index] == fake_manipulation_predictions[index] for index in indices)
        fake_subtype_metrics[manipulation_name] = {
            "count": len(indices),
            "accuracy": float(correct / len(indices)),
        }

    source_accuracies = [values["accuracy"] for values in source_metrics.values()]
    metrics.update(
        {
            "video_accuracy": float(accuracy_score(video_labels, video_predictions)),
            "video_f1": float(f1_score(video_labels, video_predictions)),
            "video_auc": safe_auc(video_labels, video_probs),
            "confusion_matrix": confusion_matrix(video_labels, video_predictions).tolist(),
            "classification_report": classification_report(video_labels, video_predictions, output_dict=True),
            "source_metrics": source_metrics,
            "fake_subtype_metrics": fake_subtype_metrics,
            "mean_source_accuracy": float(np.mean(source_accuracies)) if source_accuracies else float("nan"),
            "worst_source_accuracy": float(np.min(source_accuracies)) if source_accuracies else float("nan"),
        }
    )

    payload = {
        "video_ids": video_ids,
        "video_labels": video_labels,
        "video_probs": video_probs,
        "video_source_datasets": video_source_names,
        "video_manipulation_types": video_manipulation_names,
        "frame_probs": frame_probs,
        "frame_targets": frame_targets,
    }
    return metrics, payload


def save_history_plot(history: list[dict[str, float]], run_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train Total")
    axes[0, 0].plot(epochs, [row["train_bce_loss"] for row in history], marker="o", label="BCE")
    axes[0, 0].plot(epochs, [row["train_source_loss"] for row in history], marker="o", label="Source")
    axes[0, 0].plot(epochs, [row["train_manipulation_loss"] for row in history], marker="o", label="Subtype")
    axes[0, 0].plot(epochs, [row["train_contrastive_loss"] for row in history], marker="o", label="Contrastive")
    axes[0, 0].set_title("Training Loss Components")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train Acc")
    axes[0, 1].plot(epochs, [row["video_accuracy"] for row in history], marker="o", label="Val Video Acc")
    axes[0, 1].plot(epochs, [row["mean_source_accuracy"] for row in history], marker="o", label="Val Mean Source Acc")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Accuracy Trends")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["video_auc"] for row in history], marker="o", label="Val Video AUC")
    axes[1, 0].plot(epochs, [row["frame_auc"] for row in history], marker="o", label="Val Frame AUC")
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_title("Validation AUC")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, [row["selection_score"] for row in history], marker="o", label="Selection Score")
    axes[1, 1].plot(epochs, [row["worst_source_accuracy"] for row in history], marker="o", label="Worst Source Acc")
    axes[1, 1].plot(epochs, [row["domain_lambda"] for row in history], marker="o", label="Domain Lambda")
    axes[1, 1].set_ylim(0.0, 1.05)
    axes[1, 1].set_title("Generalization Monitor")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "loss_accuracy_curves.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(payload: dict[str, list[Any]], run_dir: Path, filename: str) -> None:
    rows = []
    for sample_id, label, probability, source_dataset, manipulation_type in zip(
        payload["video_ids"],
        payload["video_labels"],
        payload["video_probs"],
        payload["video_source_datasets"],
        payload["video_manipulation_types"],
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": source_dataset,
                "manipulation_type": manipulation_type,
                "ground_truth": "fake" if label == 1 else "real",
                "predicted_fake_probability": f"{probability:.6f}",
                "predicted_label": "fake" if probability >= 0.5 else "real",
            }
        )
    write_csv(run_dir / filename, rows)


def save_metrics_table(metrics: dict[str, dict[str, float]], output_path: Path, key_name: str) -> None:
    rows = []
    for item_name, values in metrics.items():
        row = {key_name: item_name}
        row.update(values)
        rows.append(row)
    write_csv(output_path, rows)


def save_summary_dashboard(history: list[dict[str, float]], test_metrics: dict[str, Any], best_epoch: int, run_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], marker="o")
    axes[0, 0].set_title("Train Total Loss")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, [row["video_accuracy"] for row in history], marker="o", label="Val Video Acc")
    axes[0, 1].plot(epochs, [row["mean_source_accuracy"] for row in history], marker="o", label="Val Mean Source Acc")
    axes[0, 1].plot(epochs, [row["worst_source_accuracy"] for row in history], marker="o", label="Val Worst Source Acc")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_title("Validation Generalization")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    matrix = np.array(test_metrics["confusion_matrix"])
    axes[1, 0].imshow(matrix, cmap="Blues")
    axes[1, 0].set_title("Test Confusion Matrix")
    axes[1, 0].set_xticks([0, 1], labels=["Real", "Fake"])
    axes[1, 0].set_yticks([0, 1], labels=["Real", "Fake"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes[1, 0].text(j, i, str(matrix[i, j]), ha="center", va="center")

    axes[1, 1].axis("off")
    summary_text = "\n".join(
        [
            f"Best Epoch: {best_epoch}",
            f"Test Video Accuracy: {test_metrics['video_accuracy']:.4f}",
            f"Test Mean Source Acc: {test_metrics['mean_source_accuracy']:.4f}",
            f"Test Worst Source Acc: {test_metrics['worst_source_accuracy']:.4f}",
            f"Test Video F1: {test_metrics['video_f1']:.4f}",
            f"Test Video AUC: {test_metrics['video_auc']:.4f}",
        ]
    )
    axes[1, 1].text(0.02, 0.98, summary_text, va="top", fontsize=12, family="monospace")
    fig.tight_layout()
    fig.savefig(run_dir / "summary_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    manifests_dir = dataset_root / "manifests"
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Manifest directory not found: {manifests_dir}")

    all_rows = load_manifest(manifests_dir / "all_samples.csv")
    source_to_index, manipulation_to_index = build_mappings(all_rows)
    manipulation_names_by_index = {index: name for name, index in manipulation_to_index.items()}

    train_rows = load_manifest(manifests_dir / "train.csv")
    val_rows = load_manifest(manifests_dir / "val.csv")
    test_rows = load_manifest(manifests_dir / "test.csv")

    train_records = load_frame_records(train_rows, source_to_index, manipulation_to_index)
    val_records = load_frame_records(val_rows, source_to_index, manipulation_to_index)
    test_records = load_frame_records(test_rows, source_to_index, manipulation_to_index)
    train_transform, eval_transform = build_transforms(args.image_size)
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size

    train_loader = DataLoader(
        FrameDataset(train_records, train_transform),
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_records, args.sampler_oversample_cap),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        FrameDataset(val_records, eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        FrameDataset(test_records, eval_transform),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training run, but no active CUDA device was found.")

    run_name = args.run_name or f"source_invariant_{args.model_name}_{int(time.time())}"
    run_dir = dataset_root / "training_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Run directory: {run_dir}")

    model = SourceInvariantXception(
        model_name=args.model_name,
        num_sources=len(source_to_index),
        num_manipulations=len(manipulation_to_index),
        drop_rate=args.drop_rate,
    ).to(device=device, memory_format=torch.channels_last)

    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {
                "params": list(model.classifier.parameters())
                + list(model.source_head.parameters())
                + list(model.manipulation_head.parameters())
                + list(model.projection_head.parameters()),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.backbone_lr / 50.0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_score = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, args, epoch=epoch)
        val_metrics, _ = evaluate(
            model,
            val_loader,
            device,
            manipulation_names_by_index,
            tta_flips=args.eval_tta_flips,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        scheduler.step()

        selection_score = float(
            0.5 * val_metrics["video_accuracy"]
            + 0.35 * val_metrics["mean_source_accuracy"]
            + 0.15 * val_metrics["worst_source_accuracy"]
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_bce_loss": train_metrics["bce_loss"],
            "train_source_loss": train_metrics["source_loss"],
            "train_manipulation_loss": train_metrics["manipulation_loss"],
            "train_contrastive_loss": train_metrics["contrastive_loss"],
            "train_accuracy": train_metrics["binary_accuracy"],
            "domain_lambda": train_metrics["domain_lambda"],
            "selection_score": selection_score,
            **val_metrics,
        }
        history.append(row)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{row['train_loss']:.6f}",
                "train_bce_loss": f"{row['train_bce_loss']:.6f}",
                "train_source_loss": f"{row['train_source_loss']:.6f}",
                "train_manipulation_loss": f"{row['train_manipulation_loss']:.6f}",
                "train_contrastive_loss": f"{row['train_contrastive_loss']:.6f}",
                "train_accuracy": f"{row['train_accuracy']:.6f}",
                "val_video_accuracy": f"{row['video_accuracy']:.6f}",
                "val_video_auc": f"{row['video_auc']:.6f}",
                "val_mean_source_accuracy": f"{row['mean_source_accuracy']:.6f}",
                "val_worst_source_accuracy": f"{row['worst_source_accuracy']:.6f}",
                "selection_score": f"{selection_score:.6f}",
                "domain_lambda": f"{row['domain_lambda']:.6f}",
            }
        )
        write_csv(run_dir / "epoch_history.csv", history_rows)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation_video_accuracy": val_metrics["video_accuracy"],
                    "validation_video_auc": val_metrics["video_auc"],
                    "validation_mean_source_accuracy": val_metrics["mean_source_accuracy"],
                    "validation_worst_source_accuracy": val_metrics["worst_source_accuracy"],
                    "selection_score": selection_score,
                },
                indent=2,
            )
        )

        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "selection_score": selection_score,
                    "validation_metrics": val_metrics,
                    "source_to_index": source_to_index,
                    "manipulation_to_index": manipulation_to_index,
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
            "source_to_index": source_to_index,
            "manipulation_to_index": manipulation_to_index,
        },
        run_dir / "last_model.pth",
    )

    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics, val_payload = evaluate(
        model,
        val_loader,
        device,
        manipulation_names_by_index,
        tta_flips=args.final_eval_tta_flips,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_metrics, test_payload = evaluate(
        model,
        test_loader,
        device,
        manipulation_names_by_index,
        tta_flips=args.final_eval_tta_flips,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    write_csv(run_dir / "epoch_history.csv", history_rows)
    write_csv(
        run_dir / "results_summary.csv",
        [
            {
                "model_name": args.model_name,
                "best_epoch": best_epoch,
                "selection_score": f"{best_score:.6f}",
                "validation_video_accuracy": f"{val_metrics['video_accuracy']:.6f}",
                "validation_mean_source_accuracy": f"{val_metrics['mean_source_accuracy']:.6f}",
                "test_video_accuracy": f"{test_metrics['video_accuracy']:.6f}",
                "test_mean_source_accuracy": f"{test_metrics['mean_source_accuracy']:.6f}",
                "test_worst_source_accuracy": f"{test_metrics['worst_source_accuracy']:.6f}",
                "test_video_f1": f"{test_metrics['video_f1']:.6f}",
                "test_video_auc": f"{test_metrics['video_auc']:.6f}",
            }
        ],
    )
    save_history_plot(history, run_dir)
    save_confusion_matrix_plot(val_metrics["confusion_matrix"], run_dir, "validation_confusion_matrix.png")
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], run_dir, "test_confusion_matrix.png")
    save_probability_histogram(test_payload, run_dir, "test_probability_histogram.png")
    save_classification_report(val_metrics["classification_report"], run_dir, "validation_classification_report")
    save_classification_report(test_metrics["classification_report"], run_dir, "test_classification_report")
    save_predictions_csv(val_payload, run_dir, "validation_video_predictions.csv")
    save_predictions_csv(test_payload, run_dir, "test_video_predictions.csv")
    save_metrics_table(val_metrics["source_metrics"], run_dir / "validation_source_metrics.csv", "source_dataset")
    save_metrics_table(test_metrics["source_metrics"], run_dir / "test_source_metrics.csv", "source_dataset")
    save_metrics_table(val_metrics["fake_subtype_metrics"], run_dir / "validation_fake_subtype_metrics.csv", "fake_subtype")
    save_metrics_table(test_metrics["fake_subtype_metrics"], run_dir / "test_fake_subtype_metrics.csv", "fake_subtype")
    save_summary_dashboard(history, test_metrics, best_epoch, run_dir)

    artifacts = {
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_frames": len(train_records),
        "val_frames": len(val_records),
        "test_frames": len(test_records),
        "source_to_index": source_to_index,
        "manipulation_to_index": manipulation_to_index,
    }
    (run_dir / "metrics.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
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
    (run_dir / "val_predictions.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")
    (run_dir / "test_predictions.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "dataset_root": str(dataset_root),
                "run_dir": str(run_dir),
                "source_to_index": source_to_index,
                "manipulation_to_index": manipulation_to_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run_dir": str(run_dir), **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
