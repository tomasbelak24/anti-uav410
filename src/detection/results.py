"""Framework-independent detector results."""

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    """One object detection in absolute image coordinates."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
