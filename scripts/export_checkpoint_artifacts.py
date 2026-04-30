from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_merged_classifier import (
    FrameDataset,
    build_transforms,
    create_model,
    evaluate,
    load_frame_records,
    load_manifest,
    save_classification_report,
    save_confusion_matrix_plot,
    save_history_plot,
    save_predictions_csv,
    save_probability_histogram,
    save_summary_dashboard,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export evaluation artifacts from a saved checkpoint.")
    parser.add_argument("--dataset-root", type=Path, default=Path("artifacts/merged_generalized_deepfake_1500"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="xception")
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--history-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    run_dir = args.run_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for artifact export.")

    _, eval_transform = build_transforms(args.image_size)
    val_rows = load_manifest(dataset_root / "manifests" / "val.csv")
    test_rows = load_manifest(dataset_root / "manifests" / "test.csv")

    val_loader = DataLoader(
        FrameDataset(load_frame_records(val_rows), eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        FrameDataset(load_frame_records(test_rows), eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(args.model_name).to(device=device, memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state"])

    val_metrics, val_payload = evaluate(model, val_loader, device)
    test_metrics, test_payload = evaluate(model, test_loader, device)

    history: list[dict[str, float]] = []
    if args.history_json and args.history_json.exists():
        history = json.loads(args.history_json.read_text(encoding="utf-8"))
        write_csv(
            run_dir / "epoch_history.csv",
            [
                {
                    "epoch": row["epoch"],
                    "train_loss": f"{row['train_loss']:.6f}",
                    "train_accuracy": f"{row['train_accuracy']:.6f}",
                    "val_frame_accuracy": f"{row['frame_accuracy']:.6f}",
                    "val_frame_f1": f"{row['frame_f1']:.6f}",
                    "val_frame_auc": f"{row['frame_auc']:.6f}",
                    "val_video_accuracy": f"{row['video_accuracy']:.6f}",
                    "val_video_f1": f"{row['video_f1']:.6f}",
                    "val_video_auc": f"{row['video_auc']:.6f}",
                    "learning_rate": f"{row['learning_rate']:.8f}",
                }
                for row in history
            ],
        )
        save_history_plot(history, run_dir)

    best_epoch = int(checkpoint.get("epoch", 0))
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
    save_confusion_matrix_plot(val_metrics["confusion_matrix"], run_dir, "validation_confusion_matrix.png")
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], run_dir, "test_confusion_matrix.png")
    save_probability_histogram(test_payload, run_dir, "test_probability_histogram.png")
    save_classification_report(val_metrics["classification_report"], run_dir, "validation_classification_report")
    save_classification_report(test_metrics["classification_report"], run_dir, "test_classification_report")
    save_predictions_csv(val_payload, run_dir, "validation_video_predictions.csv")
    save_predictions_csv(test_payload, run_dir, "test_video_predictions.csv")
    if history:
        save_summary_dashboard(history, test_metrics, best_epoch, run_dir)

    metrics = {
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
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
    (run_dir / "val_predictions.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")
    (run_dir / "test_predictions.json").write_text(json.dumps(test_payload, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
