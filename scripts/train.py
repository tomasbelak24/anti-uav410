"""Thin Ultralytics training entry point for Anti-UAV detectors."""

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "training" / "anti_uav_baseline.yaml"
CONFIG_KEYS = {
    "batch",
    "data",
    "device",
    "epochs",
    "imgsz",
    "model",
    "name",
    "project",
    "workers",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the supported M3 training command."""
    parser = argparse.ArgumentParser(
        description="Train an Anti-UAV detector with Ultralytics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="small project training configuration",
    )
    parser.add_argument("--data", type=Path, help="prepared dataset YAML")
    parser.add_argument("--model", help="Ultralytics model name or checkpoint")
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--imgsz", type=int, help="training image size")
    parser.add_argument("--batch", type=int, help="images per batch; use -1 for auto batch")
    parser.add_argument("--device", help="device such as 0, cpu, or mps")
    parser.add_argument("--workers", type=int, help="data-loading worker processes")
    parser.add_argument("--project", type=Path, help="training output root")
    parser.add_argument("--name", help="experiment name")
    parser.add_argument(
        "--resume",
        type=Path,
        help="Ultralytics last.pt checkpoint to resume",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments and overlay them on the baseline YAML configuration."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    except OSError as error:
        parser.error(f"cannot read training config {args.config}: {error}")

    if not isinstance(config, dict):
        parser.error(f"training config must contain a mapping: {args.config}")

    unknown_keys = sorted(set(config) - CONFIG_KEYS)
    if unknown_keys:
        parser.error(f"unsupported training config keys: {', '.join(unknown_keys)}")

    for key, value in config.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.resume is None:
        if args.data is None:
            parser.error("--data is required for a new training run")
        if not args.model:
            parser.error("--model is required for a new training run")
        if args.project is None:
            parser.error("--project is required for a new training run")
        if not args.name:
            parser.error("--name is required for a new training run")
        if args.name in {".", ".."} or "/" in args.name or "\\" in args.name:
            parser.error("--name must be a plain run name, not a path")

        for key in ("epochs", "imgsz", "batch", "workers"):
            value = getattr(args, key)
            if value is None or not isinstance(value, int):
                parser.error(f"{key} must be an integer in {args.config}")
        if args.epochs < 1:
            parser.error("epochs must be at least 1")
        if args.imgsz < 1:
            parser.error("imgsz must be at least 1")
        if args.batch == 0 or args.batch < -1:
            parser.error("batch must be -1 or a positive integer")
        if args.workers < 0:
            parser.error("workers cannot be negative")

    return args


def run_training(args: argparse.Namespace) -> Any:
    """Run one new or resumed training job through the public Ultralytics API."""
    from ultralytics import YOLO

    if args.resume is not None:
        model_source = str(args.resume)
        train_options: dict[str, Any] = {"resume": True}
        if args.device is not None:
            train_options["device"] = args.device
    else:
        model_source = str(args.model)
        project = Path(args.project).expanduser()
        if not project.is_absolute():
            project = PROJECT_ROOT / project
        train_options = {
            "data": str(args.data),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": str(project.resolve()),
            "name": args.name,
            "exist_ok": True,
        }
        if args.device is not None:
            train_options["device"] = args.device

    effective_config = {"model": model_source, **train_options}
    print("Effective training configuration:")
    print(yaml.safe_dump(effective_config, sort_keys=False).strip())

    model = YOLO(model_source)
    result = model.train(**train_options)

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is not None:
        print(f"Training artifacts: {save_dir}")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Train from the command line."""
    run_training(parse_args(argv))


if __name__ == "__main__":
    main()
