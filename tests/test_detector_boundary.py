"""Tests for the modern detector boundary introduced in M2."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.detection import Detection, UltralyticsDetector, detections_from_result


class FakeTensor:
    """Minimal stand-in for the tensor fields exposed by Ultralytics boxes."""

    def __init__(self, values: list) -> None:
        self.values = values

    def tolist(self) -> list:
        return self.values


def make_result(
    coordinates: list[list[float]], confidences: list[float], class_ids: list[float]
) -> SimpleNamespace:
    boxes = SimpleNamespace(
        xyxy=FakeTensor(coordinates),
        conf=FakeTensor(confidences),
        cls=FakeTensor(class_ids),
    )
    return SimpleNamespace(boxes=boxes)


def test_detection_holds_framework_independent_values() -> None:
    detection = Detection(xyxy=(1.0, 2.0, 10.0, 20.0), confidence=0.75, class_id=0)

    assert detection.xyxy == (1.0, 2.0, 10.0, 20.0)
    assert detection.confidence == 0.75
    assert detection.class_id == 0


def test_converts_ultralytics_result() -> None:
    result = make_result(
        coordinates=[[1, 2, 10, 20], [5.5, 6.5, 15.5, 16.5]],
        confidences=[0.75, 0.5],
        class_ids=[0, 2],
    )

    assert detections_from_result(result) == [
        Detection(xyxy=(1.0, 2.0, 10.0, 20.0), confidence=0.75, class_id=0),
        Detection(xyxy=(5.5, 6.5, 15.5, 16.5), confidence=0.5, class_id=2),
    ]


def test_empty_ultralytics_result_returns_empty_list() -> None:
    assert detections_from_result(SimpleNamespace(boxes=None)) == []
    assert detections_from_result(make_result([], [], [])) == []


def test_adapter_loads_model_and_predicts_one_frame() -> None:
    result = make_result([[1, 2, 10, 20]], [0.9], [0])
    model = Mock()
    model.predict.return_value = [result]
    yolo = Mock(return_value=model)
    fake_ultralytics = SimpleNamespace(YOLO=yolo)

    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        detector = UltralyticsDetector("weights/drone.pt")

    frame = object()
    detections = detector.predict(frame, conf=0.4, device="cpu")

    yolo.assert_called_once_with("weights/drone.pt")
    model.predict.assert_called_once_with(source=frame, conf=0.4, device="cpu")
    assert detections == [Detection(xyxy=(1.0, 2.0, 10.0, 20.0), confidence=0.9, class_id=0)]


def test_adapter_rejects_multi_source_results() -> None:
    model = Mock()
    model.predict.return_value = [make_result([], [], []), make_result([], [], [])]
    fake_ultralytics = SimpleNamespace(YOLO=Mock(return_value=model))

    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        detector = UltralyticsDetector("weights/drone.pt")

    with pytest.raises(ValueError, match="pass one image or video frame at a time"):
        detector.predict("image-directory")
