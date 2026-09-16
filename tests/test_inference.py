"""Lightweight tests for the modern M4 inference entry point."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import infer
from src.detection import Detection


class FakeTensor:
    """Minimal tensor field used by the Ultralytics result adapter."""

    def __init__(self, values: list) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


def make_result(
    coordinates: list[list[float]], confidences: list[float], class_ids: list[float]
) -> SimpleNamespace:
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=FakeTensor(coordinates),
            conf=FakeTensor(confidences),
            cls=FakeTensor(class_ids),
        )
    )


def test_cli_accepts_supported_options_and_legacy_threshold_aliases() -> None:
    args = infer.parse_args(
        [
            "--weights",
            "runs/train/drone/weights/best.pt",
            "--source",
            "sample.mp4",
            "--output",
            "artifacts/demo",
            "--img-size",
            "512",
            "--conf-thres",
            "0.4",
            "--iou-thres",
            "0.6",
            "--device",
            "cpu",
        ]
    )

    assert args.weights == Path("runs/train/drone/weights/best.pt")
    assert args.source == "sample.mp4"
    assert args.output == Path("artifacts/demo")
    assert args.imgsz == 512
    assert args.conf == 0.4
    assert args.iou == 0.6
    assert args.device == "cpu"


@pytest.mark.parametrize(
    ("option", "value"),
    [("--imgsz", "0"), ("--conf", "1.1"), ("--iou", "-0.1")],
)
def test_cli_rejects_invalid_numeric_values(option: str, value: str) -> None:
    with pytest.raises(SystemExit):
        infer.parse_args(["--weights", "best.pt", "--source", "image.jpg", option, value])


def test_streaming_inference_saves_output_and_yields_detections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = [make_result([[1, 2, 10, 20]], [0.9], [0]), make_result([], [], [])]
    model = Mock()
    model.predict.return_value = iter(results)
    yolo = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=yolo))

    args = infer.parse_args(
        [
            "--weights",
            "best.pt",
            "--source",
            "input/video.mp4",
            "--output",
            "runs/detect/demo",
            "--conf",
            "0.3",
            "--iou",
            "0.5",
            "--device",
            "0",
        ]
    )

    frames = list(infer.iter_detection_frames(args))

    yolo.assert_called_once_with("best.pt")
    model.predict.assert_called_once_with(
        source="input/video.mp4",
        imgsz=640,
        conf=0.3,
        iou=0.5,
        stream=True,
        save=True,
        project="runs/detect",
        name="demo",
        exist_ok=True,
        verbose=False,
        device="0",
    )
    assert frames == [
        [Detection(xyxy=(1.0, 2.0, 10.0, 20.0), confidence=0.9, class_id=0)],
        [],
    ]


def test_run_inference_counts_frames_and_empty_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = infer.parse_args(["--weights", "best.pt", "--source", "images"])
    frames = iter(
        [
            [Detection(xyxy=(1, 2, 3, 4), confidence=0.8, class_id=0)],
            [],
            [
                Detection(xyxy=(2, 3, 4, 5), confidence=0.7, class_id=0),
                Detection(xyxy=(3, 4, 5, 6), confidence=0.6, class_id=0),
            ],
        ]
    )
    monkeypatch.setattr(infer, "iter_detection_frames", lambda _args: frames)

    assert infer.run_inference(args) == (3, 3)


def test_help_does_not_import_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)

    with pytest.raises(SystemExit) as error:
        infer.parse_args(["--help"])

    assert error.value.code == 0
    assert "ultralytics" not in sys.modules


def test_inference_entry_point_has_no_legacy_detector_imports() -> None:
    source = Path(infer.__file__).read_text(encoding="utf-8")

    assert "sys.path" not in source
    assert "from models" not in source
    assert "from utils" not in source
