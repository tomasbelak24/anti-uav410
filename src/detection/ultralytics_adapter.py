"""Small adapter from Ultralytics results to project-owned detections."""

from pathlib import Path
from typing import Any

from src.detection.results import Detection


def detections_from_result(result: Any) -> list[Detection]:
    """Convert one Ultralytics ``Results`` object into detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    coordinates = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()

    if not (len(coordinates) == len(confidences) == len(class_ids)):
        raise ValueError("Ultralytics result fields have different lengths")

    detections = []
    for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids, strict=True):
        if len(xyxy) != 4:
            raise ValueError("Ultralytics bounding boxes must contain four xyxy values")
        detections.append(
            Detection(
                xyxy=tuple(float(value) for value in xyxy),
                confidence=float(confidence),
                class_id=int(class_id),
            )
        )
    return detections


class UltralyticsDetector:
    """Load an Ultralytics model and predict on one image or video frame."""

    def __init__(self, model: str | Path) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model))

    def predict(self, source: Any, **kwargs: Any) -> list[Detection]:
        """Predict one image/frame and return framework-independent detections."""
        results = self._model.predict(source=source, **kwargs)
        if len(results) != 1:
            raise ValueError(
                f"Expected one prediction result, received {len(results)}; "
                "pass one image or video frame at a time"
            )
        return detections_from_result(results[0])
