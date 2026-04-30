from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_merged_classifier import (
    FrameDataset,
    create_model,
    evaluate,
    load_frame_records,
    load_manifest,
    save_classification_report,
    save_confusion_matrix_plot,
    save_history_plot,
    save_predictions_csv,
    save_probability_histogram,
    save_source_metrics_csv,
    save_summary_dashboard,
    set_seed,
    write_csv,
)


class RandomJPEGCompression:
    def __init__(self, p: float = 0.25, quality_range: tuple[int, int] = (45, 95)) -> None:
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
    def __init__(self, p: float = 0.20, scale_range: tuple[float, float] = (0.65, 0.90)) -> None:
        self.p = p
        self.scale_range = scale_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        width, height = image.size
        scale = random.uniform(*self.scale_range)
        reduced_size = (max(48, int(width * scale)), max(48, int(height * scale)))
        downscaled = image.resize(reduced_size, Image.BILINEAR)
        return downscaled.resize((width, height), Image.BICUBIC)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DFDC-focused Xception specialist on the merged dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/merged_generalized_deepfake_1500"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-lr", type=float, default=7.5e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--drop-rate", type=float, default=0.2)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=3)
    parser.add_argument("--dfdc-weight", type=float, default=1.5)
    parser.add_argument("--extras-weight", type=float, default=0.35)
    parser.add_argument("--max-source-oversample", type=float, default=4.0)
    parser.add_argument("--selection-source", type=str, default=None)
    parser.add_argument("--final-eval-tta-flips", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size + 20, image_size + 20)),
            transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomJPEGCompression(p=0.25, quality_range=(45, 95)),
            RandomDownscaleUpscale(p=0.20, scale_range=(0.65, 0.90)),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.9))], p=0.10),
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


def make_dfdc_specialist_sampler(
    records: list,
    *,
    dfdc_weight: float,
    extras_weight: float,
    max_source_oversample: float,
) -> WeightedRandomSampler:
    group_counts = Counter((record.label, record.source_dataset) for record in records)
    reference_groups = [count for (_, source_name), count in group_counts.items() if source_name != "Extras"]
    reference_count = max(int(sorted(reference_groups)[len(reference_groups) // 2]), 1) if reference_groups else max(group_counts.values())
    min_effective_count = reference_count / max(max_source_oversample, 1.0)

    source_multipliers = {
        "DFDC": dfdc_weight,
        "Extras": extras_weight,
    }
    weights = []
    for record in records:
        group_key = (record.label, record.source_dataset)
        effective_count = max(group_counts[group_key], min_effective_count)
        source_multiplier = source_multipliers.get(record.source_dataset, 1.0)
        weights.append(source_multiplier / effective_count)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def split_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    backbone_params = []
    head_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    return backbone_params, head_params


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    for name, module in model.named_children():
        if name == "fc":
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad = True
            continue
        for parameter in module.parameters():
            parameter.requires_grad = not frozen
        module.train(not frozen)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    *,
    freeze_backbone: bool,
) -> tuple[float, float]:
    model.train(True)
    if freeze_backbone:
        set_backbone_frozen(model, frozen=True)
    else:
        set_backbone_frozen(model, frozen=False)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits.squeeze(1), labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probabilities = torch.sigmoid(logits.squeeze(1))
        predictions = (probabilities >= 0.5).float()
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def get_selection_metric(val_metrics: dict, selection_source: str | None) -> float:
    if selection_source:
        source_metric = val_metrics["source_metrics"].get(selection_source)
        if source_metric is None:
            available = ", ".join(sorted(val_metrics["source_metrics"]))
            raise KeyError(f"Selection source '{selection_source}' not found in validation metrics. Available: {available}")
        return float(source_metric["accuracy"])
    return float(val_metrics["video_accuracy"])


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    run_dir = dataset_root / "training_runs" / (args.run_name or f"xception_dfdc_specialist_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_manifest(dataset_root / "manifests" / "train.csv")
    val_rows = load_manifest(dataset_root / "manifests" / "val.csv")
    test_rows = load_manifest(dataset_root / "manifests" / "test.csv")

    train_transform, eval_transform = build_transforms(args.image_size)
    train_records = load_frame_records(train_rows)
    val_records = load_frame_records(val_rows)
    test_records = load_frame_records(test_rows)

    train_loader = DataLoader(
        FrameDataset(train_records, train_transform),
        batch_size=args.batch_size,
        sampler=make_dfdc_specialist_sampler(
            train_records,
            dfdc_weight=args.dfdc_weight,
            extras_weight=args.extras_weight,
            max_source_oversample=args.max_source_oversample,
        ),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        FrameDataset(val_records, eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        FrameDataset(test_records, eval_transform),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training run.")

    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Run directory: {run_dir}")

    model = create_model("xception", drop_rate=args.drop_rate).to(device=device, memory_format=torch.channels_last)
    backbone_params, head_params = split_parameters(model)
    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.backbone_lr / 20.0)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()

    best_val = -math.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history = []
    history_rows = []

    for epoch in range(1, args.epochs + 1):
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            freeze_backbone=freeze_backbone,
        )
        val_metrics, _ = evaluate(model, val_loader, device, tta_flips=False)
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "freeze_backbone": freeze_backbone,
            **val_metrics,
            "selection_metric": get_selection_metric(val_metrics, args.selection_source),
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
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
                "val_dfdc_accuracy": f"{val_metrics['source_metrics'].get('DFDC', {}).get('accuracy', float('nan')):.6f}",
                "selection_metric": f"{get_selection_metric(val_metrics, args.selection_source):.6f}",
                "val_mean_source_accuracy": f"{sum(metric['accuracy'] for metric in val_metrics['source_metrics'].values()) / len(val_metrics['source_metrics']):.6f}",
                "backbone_lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                "head_lr": f"{optimizer.param_groups[1]['lr']:.8f}",
            }
        )
        write_csv(run_dir / "epoch_history.csv", history_rows)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "freeze_backbone": freeze_backbone,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "validation_video_accuracy": val_metrics["video_accuracy"],
                    "validation_video_auc": val_metrics["video_auc"],
                    "validation_source_metrics": val_metrics["source_metrics"],
                    "selection_source": args.selection_source,
                    "selection_metric": get_selection_metric(val_metrics, args.selection_source),
                    "backbone_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                },
                indent=2,
            )
        )

        current_selection_metric = get_selection_metric(val_metrics, args.selection_source)
        if current_selection_metric > best_val:
            best_val = current_selection_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_metric": current_selection_metric,
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
    val_metrics, val_payload = evaluate(model, val_loader, device, tta_flips=args.final_eval_tta_flips)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    test_metrics, test_payload = evaluate(model, test_loader, device, tta_flips=args.final_eval_tta_flips)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    write_csv(
        run_dir / "results_summary.csv",
        [
            {
                "model_name": "xception",
                "best_epoch": best_epoch,
                "selection_source": args.selection_source or "",
                "best_selection_metric": f"{best_val:.6f}",
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

    metrics = {
        "model_name": "xception",
        "best_epoch": best_epoch,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "train_frames": len(train_records),
        "val_frames": len(val_records),
        "test_frames": len(test_records),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
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
    (run_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (run_dir / "val_predictions.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")
    (run_dir / "test_predictions.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
