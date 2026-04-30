from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import torch
from facenet_pytorch import MTCNN
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGET_IMAGES_PER_SAMPLE = 15
DEFAULT_OUTPUT = Path("artifacts/merged_generalized_deepfake_1500")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_dataset: str
    label: str
    source_type: str
    source_path: str
    subject_id: str
    basename: str
    force_split: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a merged deepfake dataset from DFDC, CelebDF, FaceForensics, and Extras."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("Dataset"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-per-sample", type=int, default=TARGET_IMAGES_PER_SAMPLE)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def make_sample(
    source_dataset: str,
    label: str,
    source_type: str,
    path: Path,
    *,
    subject_id: str | None = None,
    force_split: str | None = None,
) -> Sample:
    stem = path.stem if source_type == "video" else path.name
    sample_id = f"{source_dataset}_{label}_{stem}"
    return Sample(
        sample_id=sample_id,
        source_dataset=source_dataset,
        label=label,
        source_type=source_type,
        source_path=str(path.resolve()),
        subject_id=subject_id or stem.split("_")[0],
        basename=stem,
        force_split=force_split,
    )


def list_video_samples(
    directory: Path,
    source_dataset: str,
    label: str,
    *,
    force_split: str | None = None,
) -> list[Sample]:
    return [
        make_sample(source_dataset, label, "video", path, force_split=force_split)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def list_face_directory_samples(directory: Path, source_dataset: str, label: str) -> list[Sample]:
    return [
        make_sample(source_dataset, label, "faces_dir", path)
        for path in sorted(directory.iterdir())
        if path.is_dir()
    ]


def load_all_sources(dataset_root: Path) -> dict[str, list[Sample]]:
    celeb_root = dataset_root / "CelebDF"
    dfdc_root = dataset_root / "DFDC Dataset" / "deep-fake-detection-dfd-entire-original-dataset"
    ff_root = dataset_root / "FaceForensic"
    extras_root = dataset_root / "Extras" / "fake"

    return {
        "CelebDF_real": list_video_samples(celeb_root / "Celeb-real", "CelebDF", "real"),
        "CelebDF_fake": list_video_samples(celeb_root / "Celeb-synthesis", "CelebDF", "fake"),
        "DFDC_real": list_video_samples(dfdc_root / "DFD_original sequences", "DFDC", "real"),
        "DFDC_fake": list_video_samples(dfdc_root / "DFD_manipulated_sequences", "DFDC", "fake"),
        "FaceForensics_real": list_face_directory_samples(ff_root / "real_MTCNN_faces", "FaceForensics", "real"),
        "FaceForensics_fake": list_face_directory_samples(ff_root / "DeepFake_MTCNN_Faces", "FaceForensics", "fake"),
        "Extras_fake": list_video_samples(extras_root, "Extras", "fake", force_split="train"),
    }


def pick_samples(pool: list[Sample], count: int, rng: random.Random) -> list[Sample]:
    if len(pool) < count:
        raise ValueError(f"Requested {count} samples from a pool of {len(pool)}.")
    chosen = rng.sample(pool, count)
    chosen.sort(key=lambda item: item.sample_id)
    return chosen


def largest_remainder_split(counts: dict[str, int], forced_train: dict[str, int] | None = None) -> dict[str, dict[str, int]]:
    forced_train = forced_train or {}
    split_counts: dict[str, dict[str, int]] = {}
    total_counts = {split: 0 for split in SPLIT_RATIOS}
    remainders: list[tuple[float, str, str]] = []

    for source_name, count in counts.items():
        forced = forced_train.get(source_name, 0)
        free_count = count - forced
        raw = {split: free_count * ratio for split, ratio in SPLIT_RATIOS.items()}
        base = {split: math.floor(value) for split, value in raw.items()}
        leftover = free_count - sum(base.values())
        order = sorted(((raw[split] - base[split], split) for split in SPLIT_RATIOS), reverse=True)
        for _, split in order[:leftover]:
            base[split] += 1
        base["train"] += forced
        split_counts[source_name] = base
        for split, value in base.items():
            total_counts[split] += value
            remainders.append((raw.get(split, 0.0) - math.floor(raw.get(split, 0.0)), source_name, split))

    expected_total = sum(counts.values())
    expected = {
        "train": round(expected_total * SPLIT_RATIOS["train"]),
        "val": round(expected_total * SPLIT_RATIOS["val"]),
        "test": expected_total - round(expected_total * SPLIT_RATIOS["train"]) - round(expected_total * SPLIT_RATIOS["val"]),
    }

    for split in SPLIT_RATIOS:
        while total_counts[split] > expected[split]:
            for _, source_name, split_name in sorted(remainders):
                if split_name != split or split_counts[source_name][split] <= 0:
                    continue
                if split == "train" and split_counts[source_name]["train"] <= forced_train.get(source_name, 0):
                    continue
                destination = "val" if split == "train" else "train"
                split_counts[source_name][split] -= 1
                split_counts[source_name][destination] += 1
                total_counts[split] -= 1
                total_counts[destination] += 1
                break
        while total_counts[split] < expected[split]:
            for _, source_name, split_name in sorted(remainders, reverse=True):
                if split_name != split:
                    continue
                donor = "train" if split != "train" else "val"
                if split_counts[source_name][donor] <= forced_train.get(source_name, 0):
                    continue
                split_counts[source_name][donor] -= 1
                split_counts[source_name][split] += 1
                total_counts[donor] -= 1
                total_counts[split] += 1
                break

    return split_counts


def assign_splits(
    samples_by_source: dict[str, list[Sample]],
    counts_by_source: dict[str, int],
    rng: random.Random,
) -> dict[str, list[Sample]]:
    forced_train = {
        source_name: sum(1 for item in items if item.force_split == "train")
        for source_name, items in samples_by_source.items()
    }
    split_sizes = largest_remainder_split(counts_by_source, forced_train=forced_train)
    assigned = {split: [] for split in SPLIT_RATIOS}

    for source_name, samples in samples_by_source.items():
        forced_items = [item for item in sorted(samples, key=lambda item: item.sample_id) if item.force_split == "train"]
        free_items = [item for item in samples if item.force_split is None]
        rng.shuffle(free_items)

        counts = split_sizes[source_name]
        train_needed = counts["train"] - len(forced_items)
        val_needed = counts["val"]
        test_needed = counts["test"]

        train_items = forced_items + free_items[:train_needed]
        val_items = free_items[train_needed : train_needed + val_needed]
        test_items = free_items[train_needed + val_needed : train_needed + val_needed + test_needed]

        if len(train_items) != counts["train"] or len(val_items) != counts["val"] or len(test_items) != counts["test"]:
            raise RuntimeError(f"Split assignment mismatch for {source_name}.")

        assigned["train"].extend(sorted(train_items, key=lambda item: item.sample_id))
        assigned["val"].extend(sorted(val_items, key=lambda item: item.sample_id))
        assigned["test"].extend(sorted(test_items, key=lambda item: item.sample_id))

    return assigned


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    positions = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    return sorted(set(positions))


def copy_face_directory(sample: Sample, output_dir: Path, images_per_sample: int) -> list[Path]:
    source_dir = Path(sample.source_path)
    image_files = sorted(
        [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_files:
        raise RuntimeError(f"No face crops found in {source_dir}")

    selected_indices = evenly_spaced_indices(len(image_files), min(images_per_sample, len(image_files)))
    written_paths: list[Path] = []
    for write_index, image_index in enumerate(selected_indices):
        source_image = image_files[image_index]
        destination = output_dir / f"face_{write_index:03d}{source_image.suffix.lower()}"
        shutil.copy2(source_image, destination)
        written_paths.append(destination)
    return written_paths


def candidate_frame_indices(frame_count: int, target_images: int) -> list[int]:
    max_candidates = max(target_images * 12, target_images)
    capped_count = min(frame_count, max_candidates)
    if capped_count <= target_images:
        return list(range(capped_count))
    return evenly_spaced_indices(frame_count, capped_count)


def extract_faces_from_video(sample: Sample, output_dir: Path, mtcnn: MTCNN, images_per_sample: int) -> list[Path]:
    capture = cv2.VideoCapture(sample.source_path)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video {sample.source_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video {sample.source_path} has no frames.")

    written_paths: list[Path] = []
    try:
        for frame_index in candidate_frame_indices(frame_count, images_per_sample):
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
    finally:
        capture.release()

    if not written_paths:
        raise RuntimeError(f"No faces extracted from {sample.source_path}.")
    while len(written_paths) < images_per_sample:
        source_image = written_paths[-1]
        destination = output_dir / f"face_{len(written_paths):03d}{source_image.suffix.lower()}"
        shutil.copy2(source_image, destination)
        written_paths.append(destination)
    return written_paths


def ensure_output_root(output_root: Path, keep_existing: bool) -> None:
    if output_root.exists() and not keep_existing:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for child in ("processed", "manifests", "logs"):
        (output_root / child).mkdir(exist_ok=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for manifest {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_selection(rng: random.Random, sources: dict[str, list[Sample]]) -> dict[str, list[Sample]]:
    quotas = {
        "DFDC_real": 363,
        "CelebDF_real": 568,
        "FaceForensics_real": 569,
        "DFDC_fake": 499,
        "CelebDF_fake": 499,
        "FaceForensics_fake": 500,
        "Extras_fake": 2,
    }
    selected: dict[str, list[Sample]] = defaultdict(list)
    for source_name, count in quotas.items():
        selected[source_name] = pick_samples(sources[source_name], count, rng)
    return selected


def summarize(selected: dict[str, list[Sample]], split_assignment: dict[str, list[Sample]]) -> dict[str, object]:
    summary: dict[str, object] = {
        "selected_counts": {source: len(items) for source, items in selected.items()},
        "split_counts": {},
        "split_by_source": {},
    }
    for split, items in split_assignment.items():
        summary["split_counts"][split] = dict(Counter(item.label for item in items))
        summary["split_by_source"][split] = dict(Counter(item.source_dataset for item in items))
    return summary


def process_sample(sample: Sample, split: str, output_root: Path, mtcnn: MTCNN, images_per_sample: int) -> dict[str, object]:
    destination_dir = output_root / "processed" / split / sample.label / sample.sample_id
    destination_dir.mkdir(parents=True, exist_ok=True)

    if sample.source_type == "faces_dir":
        written_paths = copy_face_directory(sample, destination_dir, images_per_sample)
    elif sample.source_type == "video":
        written_paths = extract_faces_from_video(sample, destination_dir, mtcnn, images_per_sample)
    else:
        raise ValueError(f"Unsupported source type: {sample.source_type}")

    return {
        **asdict(sample),
        "split": split,
        "processed_dir": str(destination_dir.resolve()),
        "num_images": len(written_paths),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    rng = random.Random(args.seed)

    sources = load_all_sources(dataset_root)
    selected = build_selection(rng, sources)
    grouped = {
        "real": {name: items for name, items in selected.items() if items and items[0].label == "real"},
        "fake": {name: items for name, items in selected.items() if items and items[0].label == "fake"},
    }
    split_assignment = {"train": [], "val": [], "test": []}
    for label in ("real", "fake"):
        assigned = assign_splits(grouped[label], {name: len(items) for name, items in grouped[label].items()}, rng)
        for split in split_assignment:
            split_assignment[split].extend(assigned[split])
    for split in split_assignment:
        split_assignment[split] = sorted(split_assignment[split], key=lambda item: (item.label, item.sample_id))

    ensure_output_root(output_root, keep_existing=args.keep_existing)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(
        image_size=224,
        margin=16,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        select_largest=True,
        keep_all=False,
        device=device,
    )

    manifest_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total_items = sum(len(items) for items in split_assignment.values())
    done = 0

    for split in ("train", "val", "test"):
        for sample in split_assignment[split]:
            done += 1
            print(f"[{done}/{total_items}] Processing {split} {sample.label} {sample.sample_id}")
            try:
                manifest_rows.append(process_sample(sample, split, output_root, mtcnn, args.images_per_sample))
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": sample.sample_id,
                        "split": split,
                        "label": sample.label,
                        "source_dataset": sample.source_dataset,
                        "source_path": sample.source_path,
                        "error": repr(exc),
                    }
                )
                print(f"FAILED {sample.sample_id}: {exc}")

    if failures:
        failure_path = output_root / "logs" / "build_failures.json"
        failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"Dataset build finished with {len(failures)} failures. See {failure_path}.")

    write_manifest(output_root / "manifests" / "all_samples.csv", manifest_rows)
    for split in ("train", "val", "test"):
        write_manifest(output_root / "manifests" / f"{split}.csv", [row for row in manifest_rows if row["split"] == split])

    summary = summarize(selected, split_assignment)
    (output_root / "manifests" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
