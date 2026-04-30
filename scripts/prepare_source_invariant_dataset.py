from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE_ROOT = Path("artifacts/merged_generalized_deepfake_1500")
DEFAULT_OUTPUT_ROOT = Path("generalized_xception_source_invariant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a source-aware manifest root for source-invariant deepfake training."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_manipulation_type(row: dict[str, str]) -> str:
    if row["label"] == "real":
        return "real"
    mapping = {
        "CelebDF": "CelebDF_fake",
        "DFDC": "DFDC_fake",
        "FaceForensics": "FaceForensics_deepfake",
        "Extras": "Extras_fake",
    }
    return mapping.get(row["source_dataset"], f"{row['source_dataset']}_fake")


def enrich_row(row: dict[str, str]) -> dict[str, object]:
    manipulation_type = infer_manipulation_type(row)
    is_fake = row["label"] == "fake"
    enriched = dict(row)
    enriched.update(
        {
            "binary_label": 1 if is_fake else 0,
            "source_domain": row["source_dataset"],
            "fake_subtype": manipulation_type if is_fake else "",
            "manipulation_type": manipulation_type,
            "is_minor_source": int(row["source_dataset"] == "Extras"),
        }
    )
    return enriched


def summarize(rows_by_split: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "splits": {},
        "manipulation_types": sorted(
            {row["manipulation_type"] for rows in rows_by_split.values() for row in rows}
        ),
        "source_datasets": sorted({row["source_dataset"] for rows in rows_by_split.values() for row in rows}),
    }
    for split_name, rows in rows_by_split.items():
        source_counts = Counter((row["source_dataset"], row["label"]) for row in rows)
        manipulation_counts = Counter(row["manipulation_type"] for row in rows)
        summary["splits"][split_name] = {
            "num_samples": len(rows),
            "source_label_counts": {
                f"{source}_{label}": count for (source, label), count in sorted(source_counts.items())
            },
            "manipulation_counts": dict(sorted(manipulation_counts.items())),
        }
    return summary


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    fieldnames: list[str] | None = None

    for split_name in ["train", "val", "test"]:
        rows = load_manifest(source_root / "manifests" / f"{split_name}.csv")
        enriched_rows = [enrich_row(row) for row in rows]
        rows_by_split[split_name] = enriched_rows
        if fieldnames is None:
            fieldnames = list(enriched_rows[0].keys())
        write_csv(manifests_dir / f"{split_name}.csv", enriched_rows, fieldnames=fieldnames)

    all_rows = [row for split_rows in rows_by_split.values() for row in split_rows]
    write_csv(manifests_dir / "all_samples.csv", all_rows, fieldnames=fieldnames or [])

    summary = summarize(rows_by_split)
    summary["source_root"] = str(source_root)
    summary["output_root"] = str(output_root)
    (output_root / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
