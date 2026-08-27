"""
Unit tests for scripts/prepare_data.py

Tests the data preparation functions that convert Anti-UAV dataset to YOLO format.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from prepare_data import (
    bbox_to_yolo,
    create_dataset_yaml,
    parse_annotation_json,
    prepare_jpeg_sequence,
    prepare_output,
    process_split,
    write_manifest,
)


class TestBboxToYolo:
    """Tests for bbox_to_yolo conversion function."""

    def test_basic_conversion(self):
        """Test basic bounding box conversion."""
        # Box at (100, 100) with size 50x50 in 640x480 image
        result = bbox_to_yolo([100, 100, 50, 50], 640, 480)

        assert result is not None
        parts = result.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Expected: center at (125, 125) -> normalized (125/640, 125/480)
        assert abs(x_center - 125 / 640) < 0.0001
        assert abs(y_center - 125 / 480) < 0.0001
        assert abs(width - 50 / 640) < 0.0001
        assert abs(height - 50 / 480) < 0.0001

    def test_center_box(self):
        """Test box at image center."""
        result = bbox_to_yolo([270, 190, 100, 100], 640, 480)

        parts = result.split()
        x_center = float(parts[1])
        y_center = float(parts[2])

        # Center should be at (320, 240) -> normalized (0.5, 0.5)
        assert abs(x_center - 0.5) < 0.0001
        assert abs(y_center - 0.5) < 0.0001

    def test_edge_box(self):
        """Test box at image edge is geometrically clipped."""
        # Box partially outside image
        result = bbox_to_yolo([600, 450, 100, 100], 640, 480)

        parts = result.split()
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        assert x_center == pytest.approx(620 / 640)
        assert y_center == pytest.approx(465 / 480)
        assert width == pytest.approx(40 / 640)
        assert height == pytest.approx(30 / 480)

    def test_box_completely_outside_image(self):
        """Test that a box without image intersection is rejected."""
        assert bbox_to_yolo([700, 500, 20, 20], 640, 480) is None

    def test_invalid_zero_width(self):
        """Test that zero-width box returns None."""
        result = bbox_to_yolo([100, 100, 0, 50], 640, 480)
        assert result is None

    def test_invalid_zero_height(self):
        """Test that zero-height box returns None."""
        result = bbox_to_yolo([100, 100, 50, 0], 640, 480)
        assert result is None

    def test_invalid_negative_dimensions(self):
        """Test that negative dimensions return None."""
        result = bbox_to_yolo([100, 100, -10, 50], 640, 480)
        assert result is None

        result = bbox_to_yolo([100, 100, 50, -10], 640, 480)
        assert result is None

    def test_invalid_image_dimensions(self):
        """Test that unusable video dimensions fail clearly."""
        with pytest.raises(ValueError, match="dimensions"):
            bbox_to_yolo([100, 100, 50, 50], 0, 480)

    def test_small_box(self):
        """Test very small bounding box (typical for distant drones)."""
        result = bbox_to_yolo([300, 200, 5, 5], 640, 480)

        assert result is not None
        parts = result.split()
        width = float(parts[3])
        height = float(parts[4])

        # Should be small but valid
        assert width > 0
        assert height > 0
        assert width < 0.1
        assert height < 0.1

    def test_output_format(self):
        """Test that output has correct YOLO format."""
        result = bbox_to_yolo([100, 100, 50, 50], 640, 480)

        # Should be: "class x_center y_center width height"
        parts = result.split()
        assert len(parts) == 5

        # All values should be valid floats
        class_id = int(parts[0])
        assert class_id == 0

        for val in parts[1:]:
            f = float(val)
            assert 0 <= f <= 1


class TestParseAnnotationJson:
    """Tests for JSON annotation parsing."""

    def test_parse_valid_json(self, temp_dir):
        """Test parsing a valid annotation file."""
        annotations = {
            "exist": [1, 1, 0, 1],
            "gt_rect": [
                [100, 100, 50, 50],
                [110, 105, 48, 52],
                [0, 0, 0, 0],
                [120, 110, 46, 54],
            ],
        }

        json_path = temp_dir / "test_annotations.json"
        with open(json_path, "w") as f:
            json.dump(annotations, f)

        result = parse_annotation_json(json_path)

        assert "exist" in result
        assert "gt_rect" in result
        assert len(result["exist"]) == 4
        assert len(result["gt_rect"]) == 4
        assert result["exist"] == [1, 1, 0, 1]

    def test_parse_empty_annotations(self, temp_dir):
        """Test parsing annotations with no detections."""
        annotations = {"exist": [0, 0, 0], "gt_rect": [[0, 0, 0, 0]] * 3}

        json_path = temp_dir / "empty_annotations.json"
        with open(json_path, "w") as f:
            json.dump(annotations, f)

        result = parse_annotation_json(json_path)

        assert all(e == 0 for e in result["exist"])

    def test_parse_missing_file(self, temp_dir):
        """Test that missing file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            parse_annotation_json(temp_dir / "nonexistent.json")

    def test_parse_missing_required_field(self, temp_dir):
        """Test that malformed source annotations are not silently accepted."""
        json_path = temp_dir / "missing_gt.json"
        json_path.write_text('{"exist": [1]}', encoding="utf-8")

        with pytest.raises(ValueError, match="gt_rect"):
            parse_annotation_json(json_path)


class TestDataPipelineIntegration:
    """Integration tests for the data preparation pipeline."""

    def test_full_pipeline(self, mock_video_dataset, temp_dir):
        """Test the full data preparation pipeline."""
        from prepare_data import extract_frames_with_annotations

        input_dir = mock_video_dataset
        output_images = temp_dir / "images" / "train"
        output_labels = temp_dir / "labels" / "train"

        video_path = input_dir / "train" / "test_sequence" / "infrared.mp4"
        json_path = input_dir / "train" / "test_sequence" / "infrared.json"

        frames_extracted, frames_with_objects = extract_frames_with_annotations(
            video_path=video_path,
            json_path=json_path,
            output_images_dir=output_images,
            output_labels_dir=output_labels,
            sequence_name="test_sequence",
            modality="ir",
            sample_rate=1,  # Extract every frame
        )

        # Check that frames were extracted
        assert frames_extracted == 10
        assert frames_with_objects == 10

        # Check output files exist
        image_files = list(output_images.glob("*.jpg"))
        label_files = list(output_labels.glob("*.txt"))

        assert len(image_files) == 10
        assert len(label_files) == 10

        # Check label content
        sample_label = label_files[0].read_text().strip()
        parts = sample_label.split()
        assert len(parts) == 5
        assert parts[0] == "0"  # class id

    def test_sample_rate(self, mock_video_dataset, temp_dir):
        """Test that sample_rate correctly reduces frames."""
        from prepare_data import extract_frames_with_annotations

        input_dir = mock_video_dataset
        output_images = temp_dir / "images" / "train"
        output_labels = temp_dir / "labels" / "train"

        video_path = input_dir / "train" / "test_sequence" / "infrared.mp4"
        json_path = input_dir / "train" / "test_sequence" / "infrared.json"

        frames_extracted, _ = extract_frames_with_annotations(
            video_path=video_path,
            json_path=json_path,
            output_images_dir=output_images,
            output_labels_dir=output_labels,
            sequence_name="test_sequence",
            modality="ir",
            sample_rate=5,  # Extract every 5th frame
        )

        # 10 frames / 5 = 2 frames
        assert frames_extracted == 2

    def test_invalid_sample_rate(self, mock_video_dataset, temp_dir):
        """Test that zero cannot reach the frame modulo operation."""
        from prepare_data import extract_frames_with_annotations

        sequence = mock_video_dataset / "train" / "test_sequence"
        with pytest.raises(ValueError, match="sample_rate"):
            extract_frames_with_annotations(
                video_path=sequence / "infrared.mp4",
                json_path=sequence / "infrared.json",
                output_images_dir=temp_dir / "images" / "train",
                output_labels_dir=temp_dir / "labels" / "train",
                sequence_name="test_sequence",
                modality="ir",
                sample_rate=0,
            )

    def test_absent_target_has_empty_label_and_manifest_record(self, mock_video_dataset, temp_dir):
        """Test that source absence survives the derived empty YOLO label."""
        from prepare_data import extract_frames_with_annotations

        sequence = mock_video_dataset / "train" / "test_sequence"
        json_path = sequence / "infrared.json"
        annotations = json.loads(json_path.read_text(encoding="utf-8"))
        annotations["exist"][1] = 0
        annotations["gt_rect"][1] = [0, 0, 0, 0]
        json_path.write_text(json.dumps(annotations), encoding="utf-8")

        output_images = temp_dir / "output" / "images" / "train"
        output_labels = temp_dir / "output" / "labels" / "train"
        records = []
        frames, objects = extract_frames_with_annotations(
            video_path=sequence / "infrared.mp4",
            json_path=json_path,
            output_images_dir=output_images,
            output_labels_dir=output_labels,
            sequence_name="test_sequence",
            modality="ir",
            manifest_records=records,
        )

        assert frames == 10
        assert objects == 9
        assert (output_labels / "test_sequence_ir_000001.txt").read_text() == ""
        assert records[1]["target_present"] is False
        assert records[1]["annotation_status"] == "absent"
        assert records[1]["source_frame"] == 1
        assert records[1]["source_bbox_xywh"] == [0, 0, 0, 0]
        assert records[1]["source_layout"] == "video"

    def test_jpeg_sequence_sampling_and_manifest(self, mock_jpeg_dataset, temp_dir):
        """Test the benchmark JPEG layout without importing legacy tooling."""
        sequence = mock_jpeg_dataset / "train" / "jpeg_sequence"
        output_images = temp_dir / "prepared" / "images" / "train"
        output_labels = temp_dir / "prepared" / "labels" / "train"
        records = []

        frames, objects = prepare_jpeg_sequence(
            sequence,
            sequence / "IR_label.json",
            output_images,
            output_labels,
            sample_rate=5,
            manifest_records=records,
        )

        assert frames == 2
        assert objects == 2
        assert sorted(path.name for path in output_images.glob("*.jpg")) == [
            "jpeg_sequence_ir_000000.jpg",
            "jpeg_sequence_ir_000005.jpg",
        ]
        assert (output_images / "jpeg_sequence_ir_000000.jpg").read_bytes() == (
            sequence / "000001.jpg"
        ).read_bytes()
        assert [record["source_frame"] for record in records] == [0, 5]
        assert records[0]["source_layout"] == "jpeg"
        assert records[0]["source_image"] == "train/jpeg_sequence/000001.jpg"

    def test_jpeg_absence_and_max_frames(self, mock_jpeg_dataset, temp_dir):
        """Test that JPEG source absence produces an empty detector label."""
        sequence = mock_jpeg_dataset / "train" / "jpeg_sequence"
        output = temp_dir / "prepared"
        records = []

        frames, objects = prepare_jpeg_sequence(
            sequence,
            sequence / "IR_label.json",
            output / "images" / "train",
            output / "labels" / "train",
            max_frames=2,
            manifest_records=records,
        )

        assert frames == 2
        assert objects == 1
        assert (output / "labels/train/jpeg_sequence_ir_000001.txt").read_text() == ""
        assert records[1]["annotation_status"] == "absent"
        assert records[1]["target_present"] is False

    def test_jpeg_annotation_mismatch_fails(self, mock_jpeg_dataset, temp_dir):
        """Test that JPEG frames cannot silently shift against annotations."""
        sequence = mock_jpeg_dataset / "train" / "jpeg_sequence"
        annotations = json.loads((sequence / "IR_label.json").read_text(encoding="utf-8"))
        annotations["exist"].pop()
        (sequence / "IR_label.json").write_text(json.dumps(annotations), encoding="utf-8")

        with pytest.raises(ValueError, match="length mismatch"):
            prepare_jpeg_sequence(
                sequence,
                sequence / "IR_label.json",
                temp_dir / "prepared/images/train",
                temp_dir / "prepared/labels/train",
            )

    def test_unreadable_jpeg_fails(self, mock_jpeg_dataset, temp_dir):
        """Test that corrupt source images are reported instead of copied."""
        sequence = mock_jpeg_dataset / "train" / "jpeg_sequence"
        (sequence / "000001.jpg").write_bytes(b"not a jpeg")

        with pytest.raises(RuntimeError, match="Could not read image"):
            prepare_jpeg_sequence(
                sequence,
                sequence / "IR_label.json",
                temp_dir / "prepared/images/train",
                temp_dir / "prepared/labels/train",
            )

    def test_mixed_video_and_jpeg_sequences_are_processed(
        self, mock_video_dataset, mock_jpeg_dataset, temp_dir
    ):
        """Test automatic layout selection and honest sequence accounting."""
        unlabeled = temp_dir / "train" / "unlabelled_jpeg"
        unlabeled.mkdir()
        shutil.copy2(temp_dir / "train/jpeg_sequence/000001.jpg", unlabeled / "000001.jpg")
        records = []

        stats = process_split(
            temp_dir,
            temp_dir / "prepared",
            "train",
            "ir",
            sample_rate=5,
            max_frames_per_sequence=None,
            manifest_records=records,
        )

        assert stats["sequences"] == 3
        assert stats["processed"] == 2
        assert stats["skipped"] == 1
        assert stats["video_sources"] == 1
        assert stats["jpeg_sources"] == 1
        assert stats["frames"] == 4
        assert {record["source_layout"] for record in records} == {"video", "jpeg"}

    def test_train_val_outputs_and_yaml(self, mock_video_dataset, temp_dir):
        """Test split separation and the generated detector data contract."""
        shutil.copytree(mock_video_dataset / "train", mock_video_dataset / "val")
        output = temp_dir / "prepared"
        records = []

        for split in ("train", "val"):
            stats = process_split(
                mock_video_dataset,
                output,
                split,
                "ir",
                sample_rate=5,
                max_frames_per_sequence=None,
                manifest_records=records,
            )
            assert stats["frames"] == 2

        yaml_path = create_dataset_yaml(output)
        assert len(list((output / "images" / "train").glob("*.jpg"))) == 2
        assert len(list((output / "images" / "val").glob("*.jpg"))) == 2
        assert "train: images/train" in yaml_path.read_text(encoding="utf-8")
        assert "val: images/val" in yaml_path.read_text(encoding="utf-8")
        assert {record["split"] for record in records} == {"train", "val"}

    def test_both_modalities_are_preserved(self, mock_video_dataset, temp_dir):
        """Test the existing IR/RGB combined path."""
        sequence = mock_video_dataset / "train" / "test_sequence"
        shutil.copy2(sequence / "infrared.mp4", sequence / "visible.mp4")
        shutil.copy2(sequence / "infrared.json", sequence / "visible.json")
        records = []

        stats = process_split(
            mock_video_dataset,
            temp_dir / "prepared",
            "train",
            "both",
            sample_rate=1,
            max_frames_per_sequence=None,
            manifest_records=records,
        )

        assert stats["frames"] == 20
        assert stats["objects"] == 20
        assert {record["modality"] for record in records} == {"ir", "rgb"}


class TestPreparedArtifacts:
    """Tests for deterministic derived-output handling."""

    def test_manifest_is_deterministic_jsonl(self, temp_dir):
        records = [
            {
                "sequence": "seq",
                "source_frame": 0,
                "target_present": False,
            }
        ]
        first = write_manifest(temp_dir, records).read_text(encoding="utf-8")
        second = write_manifest(temp_dir, records).read_text(encoding="utf-8")

        assert first == second
        assert json.loads(first) == records[0]

    def test_stale_output_requires_explicit_overwrite(self, temp_dir):
        output = temp_dir / "prepared"
        derived = output / "images" / "train"
        derived.mkdir(parents=True)
        (derived / "stale.jpg").write_text("stale", encoding="utf-8")
        source_video = temp_dir / "infrared.mp4"
        source_video.write_text("source", encoding="utf-8")

        with pytest.raises(FileExistsError, match="--overwrite"):
            prepare_output(output, overwrite=False)

        prepare_output(output, overwrite=True)
        assert not derived.exists()
        assert source_video.read_text(encoding="utf-8") == "source"

    def test_overwrite_removes_unselected_stale_split(self, temp_dir):
        output = temp_dir / "prepared"
        stale_val = output / "labels" / "val" / "stale.txt"
        stale_val.parent.mkdir(parents=True)
        stale_val.write_text("stale", encoding="utf-8")

        prepare_output(output, overwrite=True)

        assert not (output / "labels").exists()
