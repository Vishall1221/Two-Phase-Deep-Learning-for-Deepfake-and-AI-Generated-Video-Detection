from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a merged deepfake classifier.")
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/merged_generalized_deepfake_1500"))
    parser.add_argument("--model-name", type=str, default="xception")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--eval-tta-flips", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@dataclass(frozen=True)
class FrameRecord:
    image_path: str
    label: int
    sample_id: str
    split: str
    source_dataset: str


class FrameDataset(Dataset):
    def __init__(self, records: list[FrameRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(record.label, dtype=torch.float32),
            record.sample_id,
            record.source_dataset,
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


def load_frame_records(manifest_rows: list[dict[str, str]]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for row in manifest_rows:
        sample_dir = Path(row["processed_dir"])
        image_files = sorted(
            path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        label = 1 if row["label"] == "fake" else 0
        for image_path in image_files:
            records.append(
                FrameRecord(
                    image_path=str(image_path),
                    label=label,
                    sample_id=row["sample_id"],
                    split=row["split"],
                    source_dataset=row["source_dataset"],
                )
            )
    return records


def make_balanced_sampler(records: list[FrameRecord]) -> WeightedRandomSampler:
    groups = Counter((record.label, record.source_dataset) for record in records)
    weights = [1.0 / groups[(record.label, record.source_dataset)] for record in records]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
            transforms.ToTensor(),
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


def create_model(model_name: str, drop_rate: float = 0.0) -> nn.Module:
    try:
        return timm.create_model(model_name, pretrained=True, num_classes=1, drop_rate=drop_rate)
    except Exception:
        return timm.create_model(model_name, pretrained=False, num_classes=1, drop_rate=drop_rate)


def get_head_module_names(model: nn.Module) -> list[str]:
    candidate_names = ["fc", "classifier", "head"]
    return [name for name in candidate_names if hasattr(model, name) and getattr(model, name) is not None]


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    backbone_params = []
    head_params = []
    head_prefixes = tuple(f"{name}." for name in get_head_module_names(model))
    for name, parameter in model.named_parameters():
        if head_prefixes and name.startswith(head_prefixes):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    return backbone_params, head_params


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    head_module_names = set(get_head_module_names(model))
    for name, module in model.named_children():
        if name in head_module_names:
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad = True
            continue
        for parameter in module.parameters():
            parameter.requires_grad = not frozen
        module.train(not frozen)


def logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.squeeze(1))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits.squeeze(1), labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probabilities = logits_to_probabilities(logits)
        predictions = (probabilities >= 0.5).float()
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    tta_flips: bool = False,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    model.eval()
    frame_targets: list[int] = []
    frame_probs: list[float] = []
    frame_sample_ids: list[str] = []
    frame_source_names: list[str] = []

    with torch.no_grad():
        for images, labels, sample_ids, source_names in loader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            logits = model(images)
            probs_tensor = logits_to_probabilities(logits)
            if tta_flips:
                flipped_logits = model(torch.flip(images, dims=[3]))
                probs_tensor = (probs_tensor + logits_to_probabilities(flipped_logits)) / 2.0
            probs = probs_tensor.detach().cpu().tolist()
            frame_probs.extend(probs)
            frame_targets.extend(labels.int().tolist())
            frame_sample_ids.extend(sample_ids)
            frame_source_names.extend(source_names)

    frame_preds = [1 if prob >= 0.5 else 0 for prob in frame_probs]
    metrics = {
        "frame_accuracy": accuracy_score(frame_targets, frame_preds),
        "frame_f1": f1_score(frame_targets, frame_preds),
        "frame_auc": roc_auc_score(frame_targets, frame_probs),
    }

    video_scores: dict[str, list[float]] = defaultdict(list)
    video_targets: dict[str, int] = {}
    video_sources: dict[str, str] = {}
    for sample_id, prob, target, source_name in zip(frame_sample_ids, frame_probs, frame_targets, frame_source_names):
        video_scores[sample_id].append(prob)
        video_targets[sample_id] = target
        video_sources[sample_id] = source_name

    video_ids = sorted(video_scores)
    video_probs = [float(np.mean(video_scores[sample_id])) for sample_id in video_ids]
    video_labels = [video_targets[sample_id] for sample_id in video_ids]
    video_source_names = [video_sources[sample_id] for sample_id in video_ids]
    video_preds = [1 if prob >= 0.5 else 0 for prob in video_probs]
    metrics.update(
        {
            "video_accuracy": accuracy_score(video_labels, video_preds),
            "video_f1": f1_score(video_labels, video_preds),
            "video_auc": roc_auc_score(video_labels, video_probs),
            "confusion_matrix": confusion_matrix(video_labels, video_preds).tolist(),
            "classification_report": classification_report(video_labels, video_preds, output_dict=True),
            "source_metrics": {
                source_name: {
                    "count": sum(1 for value in video_source_names if value == source_name),
                    "accuracy": accuracy_score(
                        [video_labels[index] for index, value in enumerate(video_source_names) if value == source_name],
                        [video_preds[index] for index, value in enumerate(video_source_names) if value == source_name],
                    ),
                }
                for source_name in sorted(set(video_source_names))
            },
        }
    )
    payload = {
        "video_ids": video_ids,
        "video_labels": video_labels,
        "video_probs": video_probs,
        "video_source_datasets": video_source_names,
        "frame_probs": frame_probs,
        "frame_targets": frame_targets,
    }
    return metrics, payload


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    header = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def save_history_plot(history: list[dict[str, float]], run_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, [row["video_accuracy"] for row in history], marker="o", label="Val Video Accuracy")
    axes[1].plot(epochs, [row["frame_accuracy"] for row in history], marker="o", label="Val Frame Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(run_dir / "loss_accuracy_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_val_accuracy_plot(history: list[dict[str, float]], run_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["video_accuracy"] for row in history], marker="o", label="Val Video Accuracy", color="#4C78A8")
    ax.plot(epochs, [row["frame_accuracy"] for row in history], marker="o", label="Val Frame Accuracy", color="#F58518")
    ax.set_title("Validation Accuracy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "val_accuracy_vs_epoch.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(conf_matrix: list[list[int]], run_dir: Path, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array(conf_matrix)
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_title("Video-Level Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["Real", "Fake"])
    ax.set_yticks([0, 1], labels=["Real", "Fake"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_probability_histogram(payload: dict[str, list[float]], run_dir: Path, filename: str) -> None:
    labels = np.array(payload["video_labels"])
    probs = np.array(payload["video_probs"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[labels == 0], bins=20, alpha=0.7, label="Real", color="#4C78A8")
    ax.hist(probs[labels == 1], bins=20, alpha=0.7, label="Fake", color="#E45756")
    ax.set_title("Video Probability Distribution")
    ax.set_xlabel("Predicted fake probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(report: dict[str, dict[str, float]], run_dir: Path, filename_prefix: str) -> None:
    rows: list[dict[str, object]] = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            row = {"label": label}
            row.update(metrics)
            rows.append(row)
    write_csv(run_dir / f"{filename_prefix}.csv", rows)

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.6)))
    ax.axis("off")
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row.get("label", ""),
                f"{float(row.get('precision', 0.0)):.4f}" if "precision" in row else "",
                f"{float(row.get('recall', 0.0)):.4f}" if "recall" in row else "",
                f"{float(row.get('f1-score', 0.0)):.4f}" if "f1-score" in row else "",
                f"{float(row.get('support', 0.0)):.0f}" if "support" in row else "",
            ]
        )
    table = ax.table(
        cellText=table_rows,
        colLabels=["Label", "Precision", "Recall", "F1", "Support"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(run_dir / f"{filename_prefix}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(payload: dict[str, list[float]], run_dir: Path, filename: str) -> None:
    rows = []
    source_datasets = payload.get("video_source_datasets", [""] * len(payload["video_ids"]))
    for sample_id, label, prob, source_dataset in zip(
        payload["video_ids"], payload["video_labels"], payload["video_probs"], source_datasets
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": source_dataset,
                "ground_truth": "fake" if label == 1 else "real",
                "predicted_fake_probability": f"{prob:.6f}",
                "predicted_label": "fake" if prob >= 0.5 else "real",
            }
        )
    write_csv(run_dir / filename, rows)


def save_source_metrics_csv(metrics: dict[str, object], run_dir: Path, filename: str) -> None:
    rows = []
    for source_name, source_metrics in metrics.get("source_metrics", {}).items():
        row = {"source_dataset": source_name}
        row.update(source_metrics)
        rows.append(row)
    write_csv(run_dir / filename, rows)


def save_summary_dashboard(
    history: list[dict[str, float]],
    test_metrics: dict[str, float],
    best_epoch: int,
    run_dir: Path,
) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, [row["train_loss"] for row in history], marker="o", color="#4C78A8")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="Train Acc")
    axes[0, 1].plot(epochs, [row["video_accuracy"] for row in history], marker="o", label="Val Video Acc")
    axes[0, 1].set_title("Accuracy Trend")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

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
            f"Test Video F1: {test_metrics['video_f1']:.4f}",
            f"Test Video AUC: {test_metrics['video_auc']:.4f}",
            f"Test Frame Accuracy: {test_metrics['frame_accuracy']:.4f}",
            f"Test Frame AUC: {test_metrics['frame_auc']:.4f}",
        ]
    )
    axes[1, 1].text(0.02, 0.98, summary_text, va="top", fontsize=12, family="monospace")

    fig.suptitle("Deepfake Training Summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(run_dir / "summary_dashboard.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    run_dir = dataset_root / "training_runs" / (args.run_name or f"{args.model_name}_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(dataset_root / "manifests" / "train.csv")
    val_rows = load_manifest(dataset_root / "manifests" / "val.csv")
    test_rows = load_manifest(dataset_root / "manifests" / "test.csv")

    train_transform, eval_transform = build_transforms(args.image_size)
    train_records = load_frame_records(train_rows)
    val_records = load_frame_records(val_rows)
    test_records = load_frame_records(test_rows)

    train_dataset = FrameDataset(train_records, train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_records) if args.balanced_sampler else None,
        shuffle=not args.balanced_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        FrameDataset(val_records, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        FrameDataset(test_records, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training run, but no active CUDA device was found.")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    model = create_model(args.model_name, drop_rate=args.drop_rate).to(device=device, memory_format=torch.channels_last)
    criterion = nn.BCEWithLogitsLoss()
    backbone_params, head_params = split_parameters(model)
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.lr / 3.0 if backbone_params else args.lr},
            {"params": head_params if head_params else model.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr / 50)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    history_rows: list[dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        set_backbone_frozen(model, freeze_backbone)
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics, _ = evaluate(model, val_loader, device, tta_flips=args.eval_tta_flips)
        scheduler.step()

        log_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "freeze_backbone": freeze_backbone,
            **val_metrics,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(log_row)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_acc:.6f}",
                "freeze_backbone": str(freeze_backbone),
                "val_frame_accuracy": f"{val_metrics['frame_accuracy']:.6f}",
                "val_frame_f1": f"{val_metrics['frame_f1']:.6f}",
                "val_frame_auc": f"{val_metrics['frame_auc']:.6f}",
                "val_video_accuracy": f"{val_metrics['video_accuracy']:.6f}",
                "val_video_f1": f"{val_metrics['video_f1']:.6f}",
                "val_video_auc": f"{val_metrics['video_auc']:.6f}",
                "backbone_lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                "head_lr": f"{optimizer.param_groups[1]['lr']:.8f}",
            }
        )
        print(json.dumps(log_row, indent=2))

        if val_metrics["video_accuracy"] > best_val:
            best_val = val_metrics["video_accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                run_dir / "best_model.pth",
            )
        else:
            epochs_without_improvement += 1

        if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "epoch": history[-1]["epoch"],
        },
        run_dir / "last_model.pth",
    )

    checkpoint = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_metrics, val_payload = evaluate(model, val_loader, device, tta_flips=args.eval_tta_flips)
    test_metrics, test_payload = evaluate(model, test_loader, device, tta_flips=args.eval_tta_flips)

    write_csv(run_dir / "epoch_history.csv", history_rows)
    write_csv(
        run_dir / "results_summary.csv",
        [
            {
                "model_name": args.model_name,
                "best_epoch": best_epoch,
                "validation_video_accuracy": f"{val_metrics['video_accuracy']:.6f}",
                "validation_video_f1": f"{val_metrics['video_f1']:.6f}",
                "validation_video_auc": f"{val_metrics['video_auc']:.6f}",
                "test_video_accuracy": f"{test_metrics['video_accuracy']:.6f}",
                "test_video_f1": f"{test_metrics['video_f1']:.6f}",
                "test_video_auc": f"{test_metrics['video_auc']:.6f}",
                "test_frame_accuracy": f"{test_metrics['frame_accuracy']:.6f}",
                "test_frame_f1": f"{test_metrics['frame_f1']:.6f}",
                "test_frame_auc": f"{test_metrics['frame_auc']:.6f}",
            }
        ],
    )
    save_history_plot(history, run_dir)
    save_val_accuracy_plot(history, run_dir)
    save_confusion_matrix_plot(val_metrics["confusion_matrix"], run_dir, "validation_confusion_matrix.png")
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], run_dir, "test_confusion_matrix.png")
    save_probability_histogram(test_payload, run_dir, "test_probability_histogram.png")
    save_classification_report(val_metrics["classification_report"], run_dir, "validation_classification_report")
    save_classification_report(test_metrics["classification_report"], run_dir, "test_classification_report")
    save_predictions_csv(val_payload, run_dir, "validation_video_predictions.csv")
    save_predictions_csv(test_payload, run_dir, "test_video_predictions.csv")
    save_source_metrics_csv(val_metrics, run_dir, "validation_source_metrics.csv")
    save_source_metrics_csv(test_metrics, run_dir, "test_source_metrics.csv")
    save_summary_dashboard(history, test_metrics, best_epoch, run_dir)

    artifacts = {
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_frames": len(train_records),
        "val_frames": len(val_records),
        "test_frames": len(test_records),
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
    print(json.dumps({"run_dir": str(run_dir), **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
