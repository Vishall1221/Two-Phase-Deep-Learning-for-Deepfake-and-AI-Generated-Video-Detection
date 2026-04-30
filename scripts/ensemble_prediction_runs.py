from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_merged_classifier import (
    save_classification_report,
    save_confusion_matrix_plot,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble two trained runs using validation-selected weights.")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_predictions(run_dir: Path, split: str) -> dict[str, Any]:
    path = run_dir / f"{split}_predictions.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def align_payloads(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    mapping_b = {
        sample_id: (label, prob, source)
        for sample_id, label, prob, source in zip(
            payload_b["video_ids"],
            payload_b["video_labels"],
            payload_b["video_probs"],
            payload_b.get("video_source_datasets", [""] * len(payload_b["video_ids"])),
        )
    }
    sample_ids: list[str] = []
    labels: list[int] = []
    probs_a: list[float] = []
    probs_b: list[float] = []
    sources: list[str] = []
    for sample_id, label, prob_a, source in zip(
        payload_a["video_ids"],
        payload_a["video_labels"],
        payload_a["video_probs"],
        payload_a.get("video_source_datasets", [""] * len(payload_a["video_ids"])),
    ):
        label_b, prob_b, source_b = mapping_b[sample_id]
        if label != label_b:
            raise ValueError(f"Label mismatch for {sample_id}")
        sample_ids.append(sample_id)
        labels.append(label)
        probs_a.append(prob_a)
        probs_b.append(prob_b)
        sources.append(source or source_b)
    return sample_ids, np.array(labels), np.array(probs_a), np.array(probs_b), sources


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.0
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, probabilities))


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, sources: list[str]) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "video_accuracy": float(accuracy_score(labels, predictions)),
        "video_f1": float(f1_score(labels, predictions)),
        "video_auc": safe_auc(labels, probabilities),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, output_dict=True),
        "source_metrics": {},
    }
    for source_name in sorted(set(sources)):
        indices = [index for index, value in enumerate(sources) if value == source_name]
        source_labels = labels[indices]
        source_probs = probabilities[indices]
        source_preds = predictions[indices]
        metrics["source_metrics"][source_name] = {
            "count": len(indices),
            "accuracy": float((source_preds == source_labels).mean()),
            "auc": safe_auc(source_labels, source_probs),
        }
    return metrics


def find_best_weight(labels: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, sources: list[str]) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    search_rows: list[dict[str, Any]] = []
    best_weight = 0.5
    best_metrics: dict[str, Any] | None = None
    best_key = (-1.0, -1.0)
    for weight in np.linspace(0.0, 1.0, 101):
        probabilities = weight * probs_a + (1.0 - weight) * probs_b
        metrics = compute_metrics(labels, probabilities, sources)
        search_rows.append(
            {
                "weight_run_a": float(weight),
                "weight_run_b": float(1.0 - weight),
                "validation_video_accuracy": metrics["video_accuracy"],
                "validation_video_f1": metrics["video_f1"],
                "validation_video_auc": metrics["video_auc"],
            }
        )
        key = (metrics["video_accuracy"], metrics["video_auc"])
        if key > best_key:
            best_key = key
            best_weight = float(weight)
            best_metrics = metrics
    assert best_metrics is not None
    return best_weight, best_metrics, search_rows


def save_search_plot(search_rows: list[dict[str, Any]], output_dir: Path) -> None:
    weights = [row["weight_run_a"] for row in search_rows]
    accuracies = [row["validation_video_accuracy"] for row in search_rows]
    aucs = [row["validation_video_auc"] for row in search_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(weights, accuracies, label="Validation Accuracy")
    ax.plot(weights, aucs, label="Validation AUC")
    ax.set_xlabel("Weight for Run A")
    ax.set_ylabel("Score")
    ax.set_title("Ensemble Weight Search")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ensemble_weight_search.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_predictions_csv(
    sample_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    sources: list[str],
    output_path: Path,
) -> None:
    rows = []
    for sample_id, label, probability, source_name in zip(sample_ids, labels.tolist(), probabilities.tolist(), sources):
        rows.append(
            {
                "sample_id": sample_id,
                "source_dataset": source_name,
                "ground_truth": "fake" if label == 1 else "real",
                "predicted_fake_probability": f"{probability:.6f}",
                "predicted_label": "fake" if probability >= 0.5 else "real",
            }
        )
    write_csv(output_path, rows)


def save_source_metrics_csv(metrics: dict[str, Any], output_path: Path) -> None:
    rows = []
    for source_name, source_metrics in metrics.get("source_metrics", {}).items():
        row = {"source_dataset": source_name}
        row.update(source_metrics)
        rows.append(row)
    write_csv(output_path, rows)


def main() -> None:
    args = parse_args()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    output_dir = (args.output_dir or (run_a.parent / f"ensemble_{run_a.name}_{run_b.name}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    val_a = load_predictions(run_a, "val")
    val_b = load_predictions(run_b, "val")
    test_a = load_predictions(run_a, "test")
    test_b = load_predictions(run_b, "test")

    val_ids, val_labels, val_probs_a, val_probs_b, val_sources = align_payloads(val_a, val_b)
    test_ids, test_labels, test_probs_a, test_probs_b, test_sources = align_payloads(test_a, test_b)

    best_weight, best_val_metrics, search_rows = find_best_weight(val_labels, val_probs_a, val_probs_b, val_sources)
    test_probabilities = best_weight * test_probs_a + (1.0 - best_weight) * test_probs_b
    test_metrics = compute_metrics(test_labels, test_probabilities, test_sources)

    write_csv(output_dir / "ensemble_weight_search.csv", search_rows)
    save_search_plot(search_rows, output_dir)
    save_confusion_matrix_plot(best_val_metrics["confusion_matrix"], output_dir, "validation_confusion_matrix.png")
    save_confusion_matrix_plot(test_metrics["confusion_matrix"], output_dir, "test_confusion_matrix.png")
    save_classification_report(best_val_metrics["classification_report"], output_dir, "validation_classification_report")
    save_classification_report(test_metrics["classification_report"], output_dir, "test_classification_report")
    save_predictions_csv(val_ids, val_labels, best_weight * val_probs_a + (1.0 - best_weight) * val_probs_b, val_sources, output_dir / "validation_video_predictions.csv")
    save_predictions_csv(test_ids, test_labels, test_probabilities, test_sources, output_dir / "test_video_predictions.csv")
    save_source_metrics_csv(best_val_metrics, output_dir / "validation_source_metrics.csv")
    save_source_metrics_csv(test_metrics, output_dir / "test_source_metrics.csv")

    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "weight_run_a": best_weight,
        "weight_run_b": 1.0 - best_weight,
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        output_dir / "results_summary.csv",
        [
            {
                "run_a": run_a.name,
                "run_b": run_b.name,
                "weight_run_a": f"{best_weight:.2f}",
                "weight_run_b": f"{1.0 - best_weight:.2f}",
                "validation_video_accuracy": f"{best_val_metrics['video_accuracy']:.6f}",
                "validation_video_auc": f"{best_val_metrics['video_auc']:.6f}",
                "test_video_accuracy": f"{test_metrics['video_accuracy']:.6f}",
                "test_video_auc": f"{test_metrics['video_auc']:.6f}",
            }
        ],
    )
    print(json.dumps({"output_dir": str(output_dir), **test_metrics}, indent=2))


if __name__ == "__main__":
    main()
