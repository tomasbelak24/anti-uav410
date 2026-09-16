# Anti-UAV: Detection and Tracking

A UAV (drone) detection and tracking project with a modern Ultralytics detector path and
preserved Anti-UAV tracking code, supporting RGB and thermal infrared video.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://docs.astral.sh/uv/)

<div align="center">
  <img src="Fig/1.gif" width="100%"/>
</div>

## Features

- **Modern Detection**: Ultralytics-based detector training and inference
- **Siamese Tracking**: Robust single-object tracking with occlusion handling
- **Multi-Modal**: Support for RGB and thermal infrared (IR) imagery
- **GPU Accelerated**: Uses the Torch/CUDA environment available on the target machine
- **Modern Tooling**: Uses `uv` for fast, reproducible dependency management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/pgryko/anti-uav410.git
cd anti-uav410

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (automatically handles PyTorch CUDA)
uv sync

# Verify GPU support
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Data Preparation

1. **Download the dataset** (choose one):

   | Dataset | Description | Download |
   |---------|-------------|----------|
   | Anti-UAV300 | RGB + IR videos | [Google Drive](https://drive.google.com/file/d/1NPYaop35ocVTYWHOYQQHn8YHsM9jmLGr/view) |
   | Anti-UAV410 | IR only, more sequences | [Baidu (wfds)](https://pan.baidu.com/s/1PbINXhxc-722NWoO8P2AdQ) |
   | Anti-UAV600 | IR only, largest | [ModelScope](https://modelscope.cn/datasets/ly261666/3rd_Anti-UAV/files) |

2. **Extract and convert to YOLO format**:

   ```bash
   # Extract dataset
   cd data/raw && unzip Anti-UAV-RGBT.zip && cd ../..

   # Convert to YOLO format (thermal IR, every 5th frame)
   uv run python scripts/prepare_data.py \
       --input data/raw \
       --output data/processed \
       --modality ir \
       --sample-rate 5
   ```

### Training

```bash
# Basic training
uv run python scripts/train.py \
    --imgsz 640 \
    --batch 16 \
    --epochs 100 \
    --data data/processed/drone.yaml \
    --name drone_v1

# Monitor with TensorBoard
tensorboard --logdir runs/train
```

### Inference

```bash
# Run detection on video
uv run python scripts/infer.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --source path/to/video.mp4 \
    --output runs/detect/drone_v1 \
    --device 0

# Annotated media is written inside runs/detect/drone_v1.

# Legacy detection + tracking path (modern integration is a later milestone)
uv run python Codes/demo_detect_track.py \
    --video path/to/video.mp4 \
    --detector runs/train/drone_v1/weights/best.pt
```

## Project Structure

```
anti-uav410/
├── Codes/                      # Core implementation
│   ├── detect_wrapper/         # YOLOv5 detection module
│   │   ├── models/             # Model architectures
│   │   └── utils/              # Detection utilities
│   └── tracking_wrapper/       # Siamese tracking module
│       ├── dronetracker/       # Main tracker implementation
│       └── drtracker/          # Alternative tracker
├── scripts/                    # Training & utility scripts
│   ├── prepare_data.py         # Dataset conversion
│   ├── train.py                # Modern detector training
│   ├── infer.py                # Modern image/video inference
│   ├── evaluate.py             # Evaluation script
│   └── export.py               # Model export (ONNX, TensorRT)
├── configs/                    # Configuration files
│   ├── models/                 # Model configs
│   └── training/               # Training hyperparameters
├── data/                       # Dataset directory (gitignored)
│   ├── raw/                    # Original videos
│   └── processed/              # YOLO format images/labels
├── docs/                       # Documentation
│   └── TRAINING.md             # Detailed training guide
└── pyproject.toml              # Project configuration
```

## Cloud Training (NVIDIA Brev)

For GPU training on [NVIDIA Brev](https://brev.nvidia.com/):

```bash
# On Brev instance
git clone https://github.com/pgryko/anti-uav410.git
cd anti-uav410
uv sync

# Upload processed data from local machine
# (run on local): rsync -avz data/processed/ user@brev:~/anti-uav410/data/processed/

# Train
uv run python scripts/train.py \
    --imgsz 640 --batch 16 --epochs 100 \
    --data data/processed/drone.yaml \
    --name drone_v1
```

See [docs/TRAINING.md](docs/TRAINING.md) for the supported baseline, Colab setup,
checkpoint artifacts, and resume workflow.

## Model Export

```bash
# Install export dependencies
uv sync --extra export

# Export to ONNX
uv run python scripts/export.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --include onnx \
    --simplify

# Export to TensorRT (NVIDIA GPUs)
uv run python scripts/export.py \
    --weights runs/train/drone_v1/weights/best.pt \
    --include engine
```

## Evaluation Metrics

The tracking accuracy is computed as:

<div align="center">
  <img src="Fig/3.png" width="80%"/>
</div>

Where IoU_t is the Intersection over Union between predicted and ground-truth boxes, with visibility flags handling target disappearance.

## Dataset Information

<div align="center">
  <img src="Fig/2.gif" width="32%"/>
  <img src="Fig/3.gif" width="32%"/>
  <img src="Fig/4.gif" width="32%"/>
</div>

The Anti-UAV datasets feature:
- **Multi-scale targets**: From distant specs to close-up drones
- **Challenging conditions**: Dynamic backgrounds, occlusions, fast motion
- **Thermal + RGB**: Dual-modality for day/night operation
- **Dense annotations**: Frame-by-frame bounding boxes with visibility flags

## Performance

| Model | Dataset | mAP@0.5 | mAP@0.5:0.95 | Inference |
|-------|---------|---------|--------------|-----------|
| YOLOv5s | Anti-UAV410 (IR) | ~75% | ~45% | 5ms (RTX 3080) |
| YOLOv5m | Anti-UAV410 (IR) | ~80% | ~50% | 8ms (RTX 3080) |

## References

This project builds on the Anti-UAV benchmark series:

- [Anti-UAV410 Paper (T-PAMI 2023)](https://ieeexplore.ieee.org/document/9615243)
- [Anti-UAV Challenge Paper (arXiv)](https://arxiv.org/abs/2306.15767)
- [Original Anti-UAV Repository](https://github.com/ZhaoJ9014/Anti-UAV)

### Citation

```bibtex
@article{huang2023anti,
  title={Anti-UAV410: A Thermal Infrared Benchmark and Customized Scheme for Tracking Drones in the Wild},
  author={Huang, Bo and Li, Jianan and Chen, Junjie and Wang, Gang and Zhao, Jian and Xu, Tingfa},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023}
}

@article{jiang2021anti,
  title={Anti-UAV: A Large-Scale Benchmark for Vision-based UAV Tracking},
  author={Jiang, Nan and Wang, Kuiran and Peng, Xiaoke and Yu, Xuehui and Wang, Qiang and others},
  journal={IEEE Transactions on Multimedia},
  year={2021}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

- Original [Anti-UAV](https://github.com/ZhaoJ9014/Anti-UAV) project by Zhao et al.
- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- [PySOT](https://github.com/STVIR/pysot) tracking toolkit
