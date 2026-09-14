"""Detector interfaces owned by the Anti-UAV project."""

from src.detection.results import Detection
from src.detection.ultralytics_adapter import UltralyticsDetector, detections_from_result

__all__ = ["Detection", "UltralyticsDetector", "detections_from_result"]
