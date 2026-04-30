from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.metrics import auc, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "artifacts" / "merged_generalized_deepfake_1500" / "training_runs"
REPORT_ROOT = PROJECT_ROOT / "final_results_report"

RUN_MAPPING = {
    "xception_baseline": RUNS_ROOT / "xception_1776369649",
    "efficientnet_b2": RUNS_ROOT / "efficientnet_b2_1776402608",
    "ensemble_xception_efficientnet_b2": RUNS_ROOT / "ensemble_xception_efficientnet_b2_1776418065",
}

PREDICTION_FILES = [
    "validation_video_predictions.csv",
    "test_video_predictions.csv",
]


def ensure_prediction_csvs(report_dir: Path, source_run_dir: Path) -> None:
    for filename in PREDICTION_FILES:
        target_path = report_dir / filename
        if target_path.exists():
            continue
        source_path = source_run_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_path)


def load_labels_and_scores(csv_path: Path) -> tuple[list[int], list[float]]:
    frame = pd.read_csv(csv_path)
    labels = [1 if str(value).strip().lower() == "fake" else 0 for value in frame["ground_truth"]]
    scores = frame["predicted_fake_probability"].astype(float).tolist()
    return labels, scores


def save_roc_curve(csv_path: Path, output_path: Path, title: str) -> None:
    labels, scores = load_labels_and_scores(csv_path)
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot(fpr, tpr, color="#2E86DE", linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#95A5A6", label="Chance")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for report_folder_name, source_run_dir in RUN_MAPPING.items():
        report_dir = REPORT_ROOT / report_folder_name
        report_dir.mkdir(parents=True, exist_ok=True)
        ensure_prediction_csvs(report_dir, source_run_dir)

        validation_csv = report_dir / "validation_video_predictions.csv"
        test_csv = report_dir / "test_video_predictions.csv"

        if validation_csv.exists():
            save_roc_curve(validation_csv, report_dir / "validation_roc_curve.png", "Validation ROC Curve")
        if test_csv.exists():
            save_roc_curve(test_csv, report_dir / "test_roc_curve.png", "Test ROC Curve")

    print(f"ROC curves generated in {REPORT_ROOT}")


if __name__ == "__main__":
    main()
