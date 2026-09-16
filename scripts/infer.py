"""Run modern Anti-UAV detector inference on images, videos, or directories."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.detection import Detection


def build_parser() -> argparse.ArgumentParser:
    """Build the supported M4 inference command."""
    parser = argparse.ArgumentParser(
        description="Run Anti-UAV detector inference with Ultralytics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True, help="trained model checkpoint")
    parser.add_argument(
        "--source",
        required=True,
        help="image, video, directory, webcam index, or stream URL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/detect"),
        help="directory for annotated output",
    )
    parser.add_argument(
        "--imgsz",
        "--img-size",
        dest="imgsz",
        type=int,
        default=640,
        help="inference image size",
    )
    parser.add_argument(
        "--conf",
        "--conf-thres",
        dest="conf",
        type=float,
        default=0.25,
        help="minimum detection confidence",
    )
    parser.add_argument(
        "--iou",
        "--iou-thres",
        dest="iou",
        type=float,
        default=0.7,
        help="IoU threshold for non-maximum suppression",
    )
    parser.add_argument("--device", help="device such as 0, cpu, or mps")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate the small supported inference interface."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.imgsz < 1:
        parser.error("--imgsz must be at least 1")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if not 0.0 <= args.iou <= 1.0:
        parser.error("--iou must be between 0 and 1")
    if not args.output.name:
        parser.error("--output must name a directory")

    return args


def _prediction_options(args: argparse.Namespace) -> dict[str, Any]:
    """Translate project arguments to the public Ultralytics predict API."""
    output = args.output.expanduser()
    options: dict[str, Any] = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "stream": True,
        "save": True,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "verbose": False,
    }
    if args.device is not None:
        options["device"] = args.device
    return options


def iter_detection_frames(args: argparse.Namespace) -> Iterator[list[Detection]]:
    """Yield framework-independent detections for each saved image or video frame."""
    from ultralytics import YOLO

    from src.detection import detections_from_result

    model = YOLO(str(args.weights))
    for result in model.predict(**_prediction_options(args)):
        yield detections_from_result(result)


def run_inference(args: argparse.Namespace) -> tuple[int, int]:
    """Run inference and return the processed frame and detection counts."""
    print(f"Model:  {args.weights}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output.expanduser()}")

    frame_count = 0
    detection_count = 0
    for detections in iter_detection_frames(args):
        frame_count += 1
        detection_count += len(detections)

    print(f"Processed {frame_count} frame(s), found {detection_count} detection(s)")
    print(f"Annotated output: {args.output.expanduser()}")
    return frame_count, detection_count


def main(argv: Sequence[str] | None = None) -> None:
    """Run detector inference from the command line."""
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
