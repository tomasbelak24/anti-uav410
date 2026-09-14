#!/usr/bin/env python3
"""
Anti-UAV Data Preparation Script
=================================
Converts Anti-UAV video or JPEG sequences with JSON annotations to YOLO format.

The Anti-UAV dataset structure:
    Anti-UAV-RGBT/
    ├── train/
    │   └── <sequence_name>/
    │       ├── infrared.mp4 + infrared.json, or
    │       ├── 000001.jpg ... + IR_label.json
    │       └── visible.mp4 + visible.json (optional)
    ├── val/
    └── test/

Output YOLO format:
    processed/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
        ├── train/
        └── val/
    ├── manifest.jsonl
    └── drone.yaml

Usage:
    # Extract a downloaded ZIP and convert it
    python scripts/prepare_data.py --input data/Anti-UAV-RGBT.zip --output data/processed

    # Extract and convert full dataset
    python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed

    # Convert only infrared (thermal) images
    python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed --modality ir

    # Sample every 5th frame (faster, less data)
    python scripts/prepare_data.py --input data/raw/Anti-UAV-RGBT --output data/processed --sample-rate 5

Author: Anti-UAV Project
"""

import argparse
import json
import math
import shutil
import zipfile
from pathlib import Path

import cv2
from tqdm import tqdm


def has_requested_splits(root: Path, splits: list[str]) -> bool:
    """Return whether root directly contains every requested dataset split."""
    return root.is_dir() and all((root / split).is_dir() for split in splits)


def find_dataset_root(base_dir: Path, splits: list[str]) -> Path | None:
    """Find a dataset root at base_dir or one directory below it."""
    if has_requested_splits(base_dir, splits):
        return base_dir
    if not base_dir.is_dir():
        return None

    candidates = sorted(path for path in base_dir.iterdir() if has_requested_splits(path, splits))
    if len(candidates) > 1:
        formatted = "\n  ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple dataset roots found under {base_dir}:\n  {formatted}")
    return candidates[0] if candidates else None


def resolve_dataset_root(input_path: Path, splits: list[str]) -> Path:
    """Resolve or extract an Anti-UAV archive and return its split root."""
    existing_root = find_dataset_root(input_path, splits)
    if existing_root is not None:
        return existing_root

    # Accept the legacy case where a root-level ZIP was already extracted beside the
    # requested wrapper directory (for example raw/train instead of raw/dataset/train).
    if not input_path.exists() and has_requested_splits(input_path.parent, splits):
        return input_path.parent

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        zip_path = input_path
        extraction_dir = input_path.with_suffix("")
    else:
        zip_candidates = [
            input_path.with_suffix(".zip"),
            input_path / "Anti-UAV-RGBT.zip",
            input_path.parent / "Anti-UAV-RGBT.zip",
        ]
        zip_path = next((path for path in zip_candidates if path.is_file()), None)
        extraction_dir = input_path

    extracted_root = find_dataset_root(extraction_dir, splits)
    if extracted_root is not None:
        return extracted_root

    if zip_path is None:
        raise FileNotFoundError(
            f"Dataset splits {splits} and Anti-UAV-RGBT.zip were not found at {input_path}"
        )

    if extraction_dir.exists():
        unrelated = [path for path in extraction_dir.iterdir() if path != zip_path]
        if unrelated:
            raise FileExistsError(
                f"Refusing to merge archive contents into non-empty directory: {extraction_dir}"
            )
    else:
        extraction_dir.mkdir(parents=True)

    print(f"\n📦 Extracting {zip_path} to {extraction_dir}...")
    destination = extraction_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        archive.extractall(extraction_dir)
    print("  ✅ Extraction complete")

    resolved_root = find_dataset_root(extraction_dir, splits)
    if resolved_root is None:
        raise FileNotFoundError(
            f"Requested dataset splits {splits} were not found after extracting {zip_path}"
        )
    return resolved_root


def parse_annotation_json(json_path: Path) -> dict:
    """
    Parse Anti-UAV JSON annotation file.

    Format:
    {
        "exist": [1, 1, 1, 0, 0, ...],  # 1 = visible, 0 = not visible
        "gt_rect": [[x, y, w, h], [x, y, w, h], ...]  # bounding boxes
    }
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Annotation must be a JSON object: {json_path}")
    for field in ("exist", "gt_rect"):
        if field not in data:
            raise ValueError(f"Annotation is missing '{field}': {json_path}")
        if not isinstance(data[field], list):
            raise ValueError(f"Annotation field '{field}' must be a list: {json_path}")
    return data


def bbox_to_yolo(bbox: list[float], img_width: int, img_height: int) -> str | None:
    """
    Convert [x, y, w, h] to YOLO format [class, x_center, y_center, width, height].

    YOLO format uses normalized coordinates (0-1).
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(bbox) != 4:
        return None

    try:
        x, y, w, h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    # Skip invalid boxes
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        return None

    # Clip the box geometry to the image before normalizing. Clamping each normalized
    # value independently can otherwise describe a box that never existed.
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(img_width), x + w)
    y2 = min(float(img_height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    clipped_width = x2 - x1
    clipped_height = y2 - y1
    x_center = (x1 + clipped_width / 2) / img_width
    y_center = (y1 + clipped_height / 2) / img_height
    norm_w = clipped_width / img_width
    norm_h = clipped_height / img_height

    # Class 0 = drone
    return f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def write_label_and_manifest(
    *,
    frame_idx: int,
    img_width: int,
    img_height: int,
    img_path: Path,
    label_path: Path,
    exist_flags: list,
    gt_rects: list,
    sequence_name: str,
    modality: str,
    source_layout: str,
    source_image: str | None,
    manifest_records: list[dict[str, object]] | None,
) -> bool:
    """Write one YOLO label and preserve its Anti-UAV source semantics."""
    has_object = False
    target_present: bool | None = None
    source_bbox: object = gt_rects[frame_idx] if frame_idx < len(gt_rects) else None
    annotation_status = "missing_exist"

    if frame_idx < len(exist_flags):
        target_present = exist_flags[frame_idx] == 1
        annotation_status = "absent"
        if target_present:
            annotation_status = "missing_bbox"
            if isinstance(source_bbox, list):
                yolo_line = bbox_to_yolo(source_bbox, img_width, img_height)
                annotation_status = "invalid_bbox"
                if yolo_line:
                    has_object = True
                    annotation_status = "present"
                    label_path.write_text(yolo_line + "\n", encoding="utf-8")

    if not has_object:
        label_path.write_text("", encoding="utf-8")

    if manifest_records is not None:
        output_root = img_path.parent.parent.parent
        record: dict[str, object] = {
            "annotation_status": annotation_status,
            "image": img_path.relative_to(output_root).as_posix(),
            "label": label_path.relative_to(output_root).as_posix(),
            "label_written": has_object,
            "modality": modality,
            "sequence": sequence_name,
            "source_bbox_xywh": source_bbox,
            "source_frame": frame_idx,
            "source_layout": source_layout,
            "split": img_path.parent.name,
            "target_present": target_present,
        }
        if source_image is not None:
            record["source_image"] = source_image
        manifest_records.append(record)

    return has_object


def extract_frames_with_annotations(
    video_path: Path,
    json_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    sequence_name: str,
    modality: str,
    sample_rate: int = 1,
    max_frames: int | None = None,
    manifest_records: list[dict[str, object]] | None = None,
) -> tuple[int, int]:
    """
    Extract frames from video and create YOLO label files.

    Returns:
        (num_frames_extracted, num_frames_with_objects)
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")

    # Parse annotations
    annotations = parse_annotation_json(json_path)
    exist_flags = annotations["exist"]
    gt_rects = annotations["gt_rect"]

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if img_width <= 0 or img_height <= 0:
        cap.release()
        raise RuntimeError(f"Video reports invalid dimensions: {video_path}")

    if len(exist_flags) != total_frames or len(gt_rects) != total_frames:
        print(
            "  ⚠️  Annotation/video length mismatch for "
            f"{sequence_name}/{modality}: video={total_frames}, "
            f"exist={len(exist_flags)}, gt_rect={len(gt_rects)}"
        )

    # Ensure output directories exist
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    frames_extracted = 0
    frames_with_objects = 0

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Sample rate
        if frame_idx % sample_rate != 0:
            frame_idx += 1
            continue

        # Max frames limit
        if max_frames is not None and frames_extracted >= max_frames:
            break

        # Generate filename
        filename = f"{sequence_name}_{modality}_{frame_idx:06d}"

        # Save image
        img_path = output_images_dir / f"{filename}.jpg"
        if not cv2.imwrite(str(img_path), frame):
            cap.release()
            raise RuntimeError(f"Could not write image: {img_path}")

        label_path = output_labels_dir / f"{filename}.txt"
        has_object = write_label_and_manifest(
            frame_idx=frame_idx,
            img_width=img_width,
            img_height=img_height,
            img_path=img_path,
            label_path=label_path,
            exist_flags=exist_flags,
            gt_rects=gt_rects,
            sequence_name=sequence_name,
            modality=modality,
            source_layout="video",
            source_image=None,
            manifest_records=manifest_records,
        )

        frames_extracted += 1
        if has_object:
            frames_with_objects += 1

        frame_idx += 1

    cap.release()
    return frames_extracted, frames_with_objects


def prepare_jpeg_sequence(
    sequence_dir: Path,
    json_path: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    sample_rate: int = 1,
    max_frames: int | None = None,
    manifest_records: list[dict[str, object]] | None = None,
) -> tuple[int, int]:
    """Copy sampled Anti-UAV JPEG frames and create YOLO labels."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero when provided")

    annotations = parse_annotation_json(json_path)
    exist_flags = annotations["exist"]
    gt_rects = annotations["gt_rect"]
    images = sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )

    if len(images) != len(exist_flags) or len(images) != len(gt_rects):
        raise ValueError(
            "JPEG/annotation length mismatch for "
            f"{sequence_dir.name}: images={len(images)}, "
            f"exist={len(exist_flags)}, gt_rect={len(gt_rects)}"
        )

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    frames_prepared = 0
    frames_with_objects = 0
    for frame_idx, source_path in enumerate(images):
        if frame_idx % sample_rate != 0:
            continue
        if max_frames is not None and frames_prepared >= max_frames:
            break

        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not read image: {source_path}")
        img_height, img_width = image.shape[:2]

        filename = f"{sequence_dir.name}_ir_{frame_idx:06d}"
        img_path = output_images_dir / f"{filename}.jpg"
        label_path = output_labels_dir / f"{filename}.txt"
        shutil.copy2(source_path, img_path)

        source_image = (
            Path(output_images_dir.name) / sequence_dir.name / source_path.name
        ).as_posix()
        has_object = write_label_and_manifest(
            frame_idx=frame_idx,
            img_width=img_width,
            img_height=img_height,
            img_path=img_path,
            label_path=label_path,
            exist_flags=exist_flags,
            gt_rects=gt_rects,
            sequence_name=sequence_dir.name,
            modality="ir",
            source_layout="jpeg",
            source_image=source_image,
            manifest_records=manifest_records,
        )
        frames_prepared += 1
        if has_object:
            frames_with_objects += 1

    return frames_prepared, frames_with_objects


def process_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    modality: str,
    sample_rate: int,
    max_frames_per_sequence: int | None,
    manifest_records: list[dict[str, object]] | None = None,
) -> dict:
    """Process all sequences in a split (train/val/test)."""
    split_dir = input_dir / split
    if not split_dir.exists():
        print(f"  ⚠️  Split directory not found: {split_dir}")
        return {
            "sequences": 0,
            "processed": 0,
            "skipped": 0,
            "video_sources": 0,
            "jpeg_sources": 0,
            "frames": 0,
            "objects": 0,
            "absent": 0,
            "invalid": 0,
        }

    # Get all sequence directories
    sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"\n  Found {len(sequences)} sequences in {split}/")

    output_images = output_dir / "images" / split
    output_labels = output_dir / "labels" / split

    total_frames = 0
    total_objects = 0
    processed_sequences = 0
    video_sources = 0
    jpeg_sources = 0
    split_records = manifest_records if manifest_records is not None else []
    first_record = len(split_records)

    for seq_dir in tqdm(sequences, desc=f"  Processing {split}"):
        sequence_processed = False

        if modality in {"ir", "infrared", "thermal", "both"}:
            video_path = seq_dir / "infrared.mp4"
            annotation_path = seq_dir / "infrared.json"
            jpeg_annotation_path = seq_dir / "IR_label.json"
            jpeg_images = any(
                path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
                for path in seq_dir.iterdir()
            )

            if video_path.is_file() and annotation_path.is_file():
                frames, objects = extract_frames_with_annotations(
                    video_path,
                    annotation_path,
                    output_images,
                    output_labels,
                    seq_dir.name,
                    "ir",
                    sample_rate,
                    max_frames_per_sequence,
                    split_records,
                )
                video_sources += 1
                sequence_processed = True
                total_frames += frames
                total_objects += objects
            elif jpeg_images and jpeg_annotation_path.is_file():
                frames, objects = prepare_jpeg_sequence(
                    seq_dir,
                    jpeg_annotation_path,
                    output_images,
                    output_labels,
                    sample_rate,
                    max_frames_per_sequence,
                    split_records,
                )
                jpeg_sources += 1
                sequence_processed = True
                total_frames += frames
                total_objects += objects
            else:
                print(
                    f"  ⚠️  Skipping {seq_dir.name}/ir: expected "
                    "infrared.mp4 + infrared.json or JPEG files + IR_label.json"
                )

        if modality in {"rgb", "visible", "both"}:
            video_path = seq_dir / "visible.mp4"
            annotation_path = seq_dir / "visible.json"
            if video_path.is_file() and annotation_path.is_file():
                frames, objects = extract_frames_with_annotations(
                    video_path,
                    annotation_path,
                    output_images,
                    output_labels,
                    seq_dir.name,
                    "rgb",
                    sample_rate,
                    max_frames_per_sequence,
                    split_records,
                )
                video_sources += 1
                sequence_processed = True
                total_frames += frames
                total_objects += objects
            else:
                print(f"  ⚠️  Skipping {seq_dir.name}/rgb: expected visible.mp4 + visible.json")

        if sequence_processed:
            processed_sequences += 1

    new_records = split_records[first_record:]
    absent = sum(record["annotation_status"] == "absent" for record in new_records)
    invalid = sum(
        record["annotation_status"] in {"invalid_bbox", "missing_bbox", "missing_exist"}
        for record in new_records
    )
    return {
        "sequences": len(sequences),
        "processed": processed_sequences,
        "skipped": len(sequences) - processed_sequences,
        "video_sources": video_sources,
        "jpeg_sources": jpeg_sources,
        "frames": total_frames,
        "objects": total_objects,
        "absent": absent,
        "invalid": invalid,
    }


def write_manifest(output_dir: Path, records: list[dict[str, object]]) -> Path:
    """Write deterministic JSON Lines metadata linking samples to source frames."""
    manifest_path = output_dir / "manifest.jsonl"
    content = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    manifest_path.write_text(content, encoding="utf-8")
    return manifest_path


def prepare_output(output_dir: Path, overwrite: bool) -> None:
    """Reject stale derived data, or explicitly remove the whole derived dataset."""
    derived_paths = [
        output_dir / "images",
        output_dir / "labels",
        output_dir / "manifest.jsonl",
        output_dir / "drone.yaml",
    ]
    existing = [path for path in derived_paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "Prepared output already exists. Choose another --output or pass --overwrite:\n  "
            + formatted
        )

    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)


def create_dataset_yaml(output_dir: Path, dataset_name: str = "drone") -> Path:
    """Create YOLO dataset configuration file."""
    yaml_content = f"""# Anti-UAV Dataset Configuration
# Auto-generated by prepare_data.py

path: {output_dir.absolute()}
train: images/train
val: images/val

# Classes
nc: 1
names: ['drone']

# Dataset info
# Generated from Anti-UAV-RGBT dataset
"""
    yaml_path = output_dir / f"{dataset_name}.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Anti-UAV dataset to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/raw/Anti-UAV-RGBT"),
        help="Input directory containing Anti-UAV dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for YOLO format data",
    )
    parser.add_argument(
        "--modality",
        "-m",
        choices=["ir", "rgb", "both"],
        default="ir",
        help="Which modality to extract (default: ir for thermal)",
    )
    parser.add_argument(
        "--sample-rate", "-s", type=int, default=5, help="Extract every Nth frame (default: 5)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames per sequence (default: unlimited)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val"],
        help="Which splits to process (default: train val)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the derived dataset (source videos/images are never removed)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Anti-UAV Data Preparation")
    print("=" * 60)
    print(f"\n  Input:       {args.input}")
    print(f"  Output:      {args.output}")
    print(f"  Modality:    {args.modality}")
    print(f"  Sample rate: every {args.sample_rate} frames")
    print(f"  Splits:      {', '.join(args.splits)}")

    if args.sample_rate <= 0:
        parser.error("--sample-rate must be greater than zero")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be greater than zero")

    input_dir = resolve_dataset_root(args.input, args.splits)
    if input_dir != args.input:
        print(f"  Resolved input: {input_dir}")

    prepare_output(args.output, args.overwrite)

    # Process each split
    stats = {}
    manifest_records: list[dict[str, object]] = []
    for split in args.splits:
        stats[split] = process_split(
            input_dir,
            args.output,
            split,
            args.modality,
            args.sample_rate,
            args.max_frames,
            manifest_records,
        )

    if not manifest_records:
        raise RuntimeError("No frames were prepared; check the requested splits and modality")

    # Create dataset YAML
    yaml_path = create_dataset_yaml(args.output)
    manifest_path = write_manifest(args.output, manifest_records)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total_frames = 0
    total_objects = 0
    for split, s in stats.items():
        print(f"\n  {split}:")
        print(f"    Sequences discovered: {s['sequences']}")
        print(f"    Sequences processed:  {s['processed']}")
        print(f"    Sequences skipped:    {s['skipped']}")
        print(f"    Video sources:        {s['video_sources']}")
        print(f"    JPEG sources:         {s['jpeg_sources']}")
        print(f"    Frames:               {s['frames']}")
        print(f"    With UAV:             {s['objects']}")
        print(f"    Absent:               {s['absent']}")
        print(f"    Invalid:              {s['invalid']}")
        total_frames += s["frames"]
        total_objects += s["objects"]

    print(f"\n  Total frames:  {total_frames}")
    print(f"  Total with UAV: {total_objects}")
    print(f"  Dataset YAML:  {yaml_path}")
    print(f"  Source manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("\nNext steps:")
    print(f"  1. Verify images in: {args.output / 'images'}")
    print(f"  2. Verify labels in: {args.output / 'labels'}")
    print(f"  3. Inspect source metadata in: {manifest_path}")
    print("  Modern detector training is introduced in milestone M2.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
