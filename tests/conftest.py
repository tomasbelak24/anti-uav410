"""
Pytest fixtures for Anti-UAV test suite.
"""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def device():
    """Return the best available device (CUDA if available, else CPU)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def sample_image() -> np.ndarray:
    """Create a sample test image with a synthetic UAV-like object."""
    # Create a 640x480 grayscale image (thermal-like)
    img = np.zeros((480, 640), dtype=np.uint8)
    # Add some background noise
    img = img + np.random.randint(20, 40, img.shape, dtype=np.uint8)
    # Add a bright spot (simulating a drone)
    cv2.circle(img, (320, 240), 15, 200, -1)
    # Convert to 3-channel for YOLO
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_rgb


@pytest.fixture(scope="function")
def sample_dataset(temp_dir, sample_image) -> Path:
    """
    Create a minimal YOLO-format dataset for testing.

    Structure:
        temp_dir/
        ├── images/
        │   ├── train/
        │   │   └── sample_0.jpg
        │   └── val/
        │       └── sample_1.jpg
        ├── labels/
        │   ├── train/
        │   │   └── sample_0.txt
        │   └── val/
        │       └── sample_1.txt
        └── dataset.yaml
    """
    # Create directories
    for split in ["train", "val"]:
        (temp_dir / "images" / split).mkdir(parents=True)
        (temp_dir / "labels" / split).mkdir(parents=True)

    # Create sample images and labels
    for split in ["train", "val"]:
        for i in range(3):  # 3 images per split
            img_name = f"sample_{split}_{i}.jpg"
            label_name = f"sample_{split}_{i}.txt"

            # Vary the image slightly
            img = sample_image.copy()
            # Move the "drone" to different positions
            img_modified = np.zeros_like(img)
            cv2.circle(img_modified, (200 + i * 100, 200 + i * 50), 15, (200, 200, 200), -1)
            img = cv2.addWeighted(img, 0.5, img_modified, 0.5, 0)

            cv2.imwrite(str(temp_dir / "images" / split / img_name), img)

            # YOLO format label: class x_center y_center width height (normalized)
            # Drone at roughly center with small size
            x_center = (200 + i * 100) / 640
            y_center = (200 + i * 50) / 480
            width = 30 / 640
            height = 30 / 480
            label_content = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

            (temp_dir / "labels" / split / label_name).write_text(label_content)

    # Create dataset YAML
    dataset_yaml = {
        "path": str(temp_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["drone"],
    }

    yaml_path = temp_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f)

    return temp_dir


@pytest.fixture(scope="session")
def hyp_config() -> dict:
    """Return minimal hyperparameters for smoke testing."""
    return {
        "lr0": 0.01,
        "lrf": 0.2,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 0.1,  # Minimal warmup for testing
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 0.05,
        "cls": 0.5,
        "cls_pw": 1.0,
        "obj": 1.0,
        "obj_pw": 1.0,
        "iou_t": 0.20,
        "anchor_t": 4.0,
        "fl_gamma": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,  # Disable mosaic for testing (faster)
        "mixup": 0.0,
    }


@pytest.fixture(scope="function")
def mock_video_dataset(temp_dir) -> Path:
    """
    Create a mock Anti-UAV style video dataset for testing prepare_data.py.

    Structure:
        temp_dir/
        └── train/
            └── test_sequence/
                ├── infrared.mp4
                └── infrared.json
    """
    seq_dir = temp_dir / "train" / "test_sequence"
    seq_dir.mkdir(parents=True)

    # Create a small test video (10 frames)
    video_path = seq_dir / "infrared.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 30, (640, 480), isColor=False)

    for i in range(10):
        frame = np.zeros((480, 640), dtype=np.uint8) + 30
        # Add a moving "drone"
        x = 100 + i * 40
        y = 200 + i * 20
        cv2.circle(frame, (x, y), 10, 200, -1)
        out.write(frame)

    out.release()

    # Create annotation JSON
    import json

    annotations = {
        "exist": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "gt_rect": [[100 + i * 40 - 10, 200 + i * 20 - 10, 20, 20] for i in range(10)],
    }

    json_path = seq_dir / "infrared.json"
    with open(json_path, "w") as f:
        json.dump(annotations, f)

    return temp_dir
