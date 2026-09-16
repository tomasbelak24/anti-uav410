"""Lightweight tests for the modern M3 training entry point."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from scripts import train


@pytest.fixture
def training_config(tmp_path: Path) -> Path:
    config = {
        "model": "yolo11n.pt",
        "epochs": 20,
        "imgsz": 640,
        "batch": 8,
        "device": None,
        "workers": 2,
        "project": "runs/train",
        "name": "anti-uav-test",
    }
    path = tmp_path / "training.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_baseline_config_and_cli_override() -> None:
    args = train.parse_args(
        ["--data", "data/processed/drone.yaml", "--epochs", "1", "--batch", "4"]
    )

    assert args.model == "yolo11n.pt"
    assert args.epochs == 1
    assert args.imgsz == 640
    assert args.batch == 4
    assert args.workers == 2
    assert args.project == "runs/train"
    assert args.name == "anti-uav-baseline"


def test_new_training_forwards_supported_arguments(
    monkeypatch: pytest.MonkeyPatch, training_config: Path
) -> None:
    model = Mock()
    model.train.return_value = "metrics"
    model.trainer = SimpleNamespace(save_dir=Path("runs/train/smoke"))
    yolo = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=yolo))

    args = train.parse_args(
        [
            "--config",
            str(training_config),
            "--data",
            "prepared/drone.yaml",
            "--epochs",
            "1",
            "--imgsz",
            "512",
            "--batch",
            "4",
            "--device",
            "cpu",
            "--workers",
            "0",
            "--project",
            "outputs",
            "--name",
            "smoke",
        ]
    )

    result = train.run_training(args)

    yolo.assert_called_once_with("yolo11n.pt")
    model.train.assert_called_once_with(
        data="prepared/drone.yaml",
        epochs=1,
        imgsz=512,
        batch=4,
        device="cpu",
        workers=0,
        project=str((train.PROJECT_ROOT / "outputs").resolve()),
        name="smoke",
        exist_ok=True,
    )
    assert result == "metrics"


def test_resume_loads_checkpoint_and_uses_library_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = Mock()
    yolo = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=yolo))

    resume_config = tmp_path / "resume.yaml"
    resume_config.write_text("{}\n", encoding="utf-8")

    args = train.parse_args(
        [
            "--config",
            str(resume_config),
            "--resume",
            "runs/train/anti-uav/weights/last.pt",
            "--device",
            "0",
        ]
    )
    train.run_training(args)

    yolo.assert_called_once_with("runs/train/anti-uav/weights/last.pt")
    model.train.assert_called_once_with(resume=True, device="0")


def test_new_training_requires_dataset(training_config: Path) -> None:
    with pytest.raises(SystemExit):
        train.parse_args(["--config", str(training_config)])


@pytest.mark.parametrize("name", ["runs/train/smoke", "runs\\train\\smoke", ".", ".."])
def test_training_name_must_not_be_a_path(training_config: Path, name: str) -> None:
    with pytest.raises(SystemExit):
        train.parse_args(
            [
                "--config",
                str(training_config),
                "--data",
                "prepared/drone.yaml",
                "--name",
                name,
            ]
        )


def test_help_does_not_import_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)

    with pytest.raises(SystemExit) as error:
        train.parse_args(["--help"])

    assert error.value.code == 0
    assert "ultralytics" not in sys.modules


def test_training_entry_point_has_no_legacy_detector_imports() -> None:
    source = Path(train.__file__).read_text(encoding="utf-8")

    assert "sys.path" not in source
    assert "from models" not in source
    assert "from utils" not in source
