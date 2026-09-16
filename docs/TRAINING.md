# Anti-UAV detector training

The supported training path uses the public Ultralytics API through the thin
`scripts/train.py` entry point. The copied historical YOLO training loops are not the
supported default.

Ultralytics is the prototype detector dependency. Its use here is not the final
commercial licensing decision.

## Prepare the dataset

Create the YOLO dataset and `drone.yaml` with M1 before training. For example:

```bash
python scripts/prepare_data.py \
    --input data/raw/Anti-UAV-RGBT \
    --output data/processed \
    --modality ir \
    --sample-rate 5
```

## Baseline training

The version-controlled baseline is `configs/training/anti_uav_baseline.yaml`. The data
location stays outside that file because local and Colab paths differ.

Run a one-epoch smoke test:

```bash
python scripts/train.py \
    --data data/processed/drone.yaml \
    --epochs 1 \
    --name anti-uav-smoke
```

Run the 100-epoch baseline:

```bash
python scripts/train.py --data data/processed/drone.yaml
```

The baseline starts from `yolo11n.pt`, trains at 640 pixels with batch size 16, and
writes under `runs/train/anti-uav-baseline`. Ultralytics may append a number rather than
overwrite an existing run directory.

Useful overrides are intentionally limited:

```text
--model       model name or starting checkpoint
--epochs      training epochs
--imgsz       image size
--batch       batch size; -1 requests automatic sizing
--device      0, cpu, mps, and other Ultralytics-supported values
--workers     data-loader workers
--project     output root
--name        run name
```

The script prints the effective configuration before calling Ultralytics. Framework
internals such as the training loop, optimizer implementation, loss, AMP, validation,
and checkpoint serialization remain library responsibilities.

## Artifacts

For a run named `anti-uav-smoke`, the important outputs are normally:

```text
runs/train/anti-uav-smoke/weights/best.pt
runs/train/anti-uav-smoke/weights/last.pt
runs/train/anti-uav-smoke/args.yaml
runs/train/anti-uav-smoke/results.csv
```

Use `best.pt` for later validation/inference and `last.pt` to resume an interrupted run.

## Resume

Resume through Ultralytics' checkpoint state:

```bash
python scripts/train.py \
    --resume runs/train/anti-uav-smoke/weights/last.pt \
    --device 0
```

The checkpoint restores the prior run settings, optimizer, scheduler, and epoch. The
project wrapper forwards `resume=True` instead of reconstructing that state itself.

## Google Colab

Current Colab Python 3.13 is supported by the modern M1-M3 path. Colab already provides
a GPU-enabled Torch installation, so the lightweight notebook setup is:

```python
%pip install -q "ultralytics>=8.4,<9"
%pip install -q -e . --no-deps
```

Then train against the prepared dataset:

```python
!python scripts/train.py \
    --data /content/data/processed/drone.yaml \
    --epochs 1 \
    --device 0 \
    --name anti-uav-smoke
```

The one-epoch GPU run is manual validation. Automated tests mock the heavy library call.

## Current migration boundary

Detector training and standalone inference now use the modern path. Detector validation
and export remain legacy until M5. Tracker integration and Anti-UAV benchmark evaluation
have not changed.
