# Anti-UAV410 Repository Architecture

**Status:** Current-state architecture and modernization boundaries
**Basis:** Public `pgryko/anti-uav410` `main` commit
`001b79550070d6ec1b3d81bda1d93a85302234f7`, checked out and verified on 2026-08-26
**Purpose:** Give humans and Codex a shared map of the repository before changing it

---

## 1. Why this document exists

This project should be modernized **in place**.

The repository already contains valuable Anti-UAV-specific behavior, historical
tracking code, tests, data conversion logic, configuration, documentation, and demos.
The modernization therefore should not start from the assumption that the repository
must be replaced.

The core architectural problem is narrower:

> A modern project shell still depends on legacy YOLOv5-era detector internals through
> path manipulation and copied framework code.

The modernization goal is to replace that generic detector framework while preserving
domain-specific Anti-UAV knowledge.

---

## 2. High-level view

The repository can be understood as three layers.

```text
┌───────────────────────────────────────────────────────────────┐
│                     MODERN PROJECT SHELL                      │
│ pyproject.toml / uv / pytest / Ruff / Pyright / docs / tests │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                   USEFUL ANTI-UAV LOGIC                       │
│ dataset conversion / target existence / tracking / evaluation│
│ demos / experiment knowledge                                 │
└──────────────────────────────┬────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────┐
│                    LEGACY ML FRAMEWORK                        │
│ copied YOLOv5 models/utils/training/inference/export          │
│ old detector wrapper + old environments + path hacks          │
└───────────────────────────────────────────────────────────────┘
```

The correct migration is therefore:

```text
preserve domain behavior
        +
replace generic detector framework
        +
verify parity
        +
delete superseded detector code
```

not:

```text
rewrite everything
```

---

## 3. Current execution flows

### 3.1 Data preparation

Current useful path:

```text
Anti-UAV source videos + JSON
          │
          ▼
scripts/prepare_data.py
          │
          ├── frame sampling
          ├── IR/RGB modality handling
          ├── `exist` target-presence handling
          ├── bbox conversion
          ├── image extraction
          ├── YOLO label generation
          └── dataset YAML generation
          │
          ▼
prepared detector-training dataset
```

This is primarily **domain/project logic** and should be preserved.

### 3.2 Detector training

Current path:

```text
scripts/train.py
       │
       ├── inserts `src/detection` into sys.path
       ├── imports `models.*`
       ├── imports `utils.*`
       ├── constructs YOLO model
       ├── owns data loader
       ├── owns loss/optimizer/scheduler
       ├── owns checkpointing
       └── calls legacy validation
```

This is primarily **generic YOLO framework code** and should be replaced.

### 3.3 Detector inference

Current path:

```text
scripts/infer.py
       │
       ├── inserts `src/detection` into sys.path
       ├── imports `models.experimental.attempt_load`
       ├── imports legacy image/stream loaders
       ├── imports NMS
       ├── imports coordinate scaling / plotting
       └── performs detector inference
```

This is generic detector-framework behavior and should be replaced.

### 3.4 Detector evaluation

Current path:

```text
scripts/evaluate.py
       │
       ├── imports legacy YOLO detector internals
       ├── builds detector data loaders
       ├── runs NMS
       ├── computes detector AP/IoU metrics
       └── produces detector-centric validation output
```

This should be replaced by the maintained detector library's validation API.

This is **not the same concern** as Anti-UAV tracking-benchmark evaluation.

### 3.5 Detector + tracker demo

Current integration area:

```text
Codes/
├── detect_wrapper/
├── tracking_wrapper/
├── demo_detect_track.py
└── detect_tracking.py
```

The important migration rule is:

> Replace the detector side first and keep the tracker side stable until the detector
> has been proven independently.

### 3.6 Anti-UAV tracking evaluation

The Anti-UAV410 benchmark implementation used by the bundled Jittor experiment is:

```text
anti_uav_jittor/anti_uav410_jit/experiments/anti_uav.py
```

`src/evaluation/toolkit/` is a separate generic tracking toolkit, not the verified
Anti-UAV benchmark source of truth. Both areas remain frozen during detector migration.

They must not be casually replaced while detector modernization is underway because
Anti-UAV tracking includes target-existence/absence semantics that are not identical to
generic multi-object tracking.

### 3.7 M0 verified flow inventory

| Flow | Entry point and dependencies | Coupling | Tests | Expected artifacts |
|---|---|---|---|---|
| Data preparation | `scripts/prepare_data.py`; stdlib JSON/ZIP, OpenCV, tqdm | YOLO text/YAML output only | `tests/test_prepare_data.py` | source `infrared.mp4`/`.json`, JPEG + `IR_label.json`, optionally visible video; derived images, labels, YAML, manifest |
| Detector training | `scripts/train.py`; Torch plus copied `src/detection/models` and `utils` | hard `sys.path` injection and legacy YOLOv5 internals | `tests/test_training_smoke.py` exercises the different `Codes/detect_wrapper` copy, not this entry point | pretrained `.pt`; `last.pt`, `best.pt`, run metrics |
| Inference | `scripts/infer.py`; Torch/OpenCV plus copied `src/detection` loaders, NMS, plotting | hard legacy YOLOv5 coupling | no entry-point test | default `weights/best.pt`; annotated run output |
| Detector validation/export | `scripts/evaluate.py`, `scripts/export.py`; Torch plus `src/detection` | hard legacy model/checkpoint and graph coupling | no entry-point tests | defaults differ (`best_drone.pt` vs `weights/best.pt`); metrics and TorchScript/ONNX/CoreML artifacts |
| Detector + tracker | `Codes/demo_detect_track.py`, `Codes/detect_tracking.py`; `Codes/detect_wrapper` and `Codes/tracking_wrapper` | path injection, copied detector, historical tracker environments | no integration test | expects missing `Codes/detect_wrapper/weights/best.pt`; input/output videos |
| Anti-UAV benchmark | `anti_uav_jittor/anti_uav410_jit/experiments/anti_uav.py`; Jittor experiment dataset/tracker stack, NumPy | Jittor and bundled tracker layout | no focused metric tests | per-sequence result text and report `performance.json` |

The public tree contains three detector-framework copies, not two:
`src/detection/`, `Codes/detect_wrapper/`, and
`anti_uav_jittor/anti_uav_edtc_jit/yolov5/`. The root `demo_ui.py` imports the third
copy and expects `weights/drone_detector.pt`.

---

## 4. Repository map and disposition

Legend:

- **KEEP** — useful now; no broad rewrite required.
- **ADAPT** — preserve behavior, improve implementation incrementally.
- **REPLACE** — old generic framework responsibility should be substituted.
- **FREEZE** — do not change until a later milestone.
- **DELETE AFTER PARITY** — retain temporarily for rollback/comparison; remove only
  after explicit acceptance criteria pass.
- **REVIEW** — value or ownership must be established before action.

### 4.1 Root files

```text
README.md
```

**Disposition: ADAPT**

The README should eventually describe the supported modern workflow rather than legacy
YOLOv5 commands. Performance claims should only be retained if backed by reproducible
artifacts or newly reproduced experiments.

```text
pyproject.toml
```

**Disposition: ADAPT**

This is already a useful modern shell:

- Python 3.10–3.12 metadata;
- Hatchling;
- pytest;
- Ruff;
- mypy;
- Pyright;
- uv;
- optional training/export dependencies.

Important current signals:

- console scripts point to legacy `scripts.*` entry points;
- CUDA-specific Torch indexes are configured;
- broad Ruff ignores document legacy patterns;
- Pyright explicitly excludes `Codes`, `src/tracking`,
  `src/evaluation/toolkit`, and `src/detection`.

Do not rewrite this file from scratch. Tighten it as the migration removes legacy code.

```text
LICENSE
```

**Disposition: REVIEW / KEEP**

Keep existing repository licensing information, but do not assume a top-level license
automatically resolves provenance of every copied third-party file or model weight.

---

## 5. Data preparation architecture

### 5.1 `scripts/prepare_data.py`

**Disposition: KEEP + ADAPT**

This file already contains valuable project-specific behavior.

Observed behavior includes:

- parses Anti-UAV JSON annotations;
- reads `exist` flags;
- reads `gt_rect` bounding boxes;
- converts `[x, y, w, h]` boxes to normalized YOLO format;
- samples video frames;
- defaults to `sample_rate=5`;
- supports `ir`, `rgb`, and `both`;
- creates labels only when the target exists and has a valid box;
- handles absent-target samples;
- processes train/validation splits;
- creates detector dataset YAML.

This logic should not be rewritten merely because the detector framework changes.

### 5.2 Data-source semantics

The current locally audited Colab archive is the 5.6 GB `Anti-UAV-RGBT.zip` associated
with Google Drive file `1VZFq7g-z5-k0VEkGLKW2i29RzGd47dRD`. It has no enclosing
dataset directory: `train`, `val`, `test`, `label_new`, and `framecut.py` are directly
at the ZIP root. It contains 160 train, 67 validation, and 91 test sequences. Every
sequence has `infrared.mp4`, `infrared.json`, `visible.mp4`, and `visible.json`; the
archive contains no JPEG sequence data.

`label_new/{train,val,test}.json` maps every sequence to challenge-attribute codes; it
is analysis metadata and is not needed to generate detector labels. `framecut.py` is a
legacy all-frames MP4-to-JPEG utility and is superseded by `scripts/prepare_data.py` for
the modern detector workflow.

Google Drive file `1F0nGafdWP4PddmqVDLFNpHmukEFRpX8Y` was previously verified by the
user as downloadable with the expected RGBT names, but its archive contents were not
audited locally.

The separate legacy Anti-UAV410 benchmark path in this repository uses numbered JPEGs
plus `IR_label.json`. `scripts/prepare_data.py` accepts that representation for
compatibility, but it is not part of the audited RGBT ZIP. A prior Colab directory with
210 train and 93 validation folders had been formed by overlaying sources; those counts
must not be treated as an archive contract.

For either supported representation, the converter prefers a complete video/annotation
pair if both are present. ZIP extraction resolves root-level or singly wrapped split
directories and refuses to merge into a non-empty unrelated directory.

All audited IR annotations have matching `exist` and `gt_rect` lengths. However, 179
train, 72 validation, and 43 test frames say `exist=1` while supplying a zero-size box.
Preparation records these contradictions as `invalid_bbox` rather than silently
changing them to true target absence.

The Anti-UAV domain representation contains more information than YOLO labels.

At minimum, source semantics include:

```text
sequence
source frame
modality
target exists / absent
bounding box when present
split
```

Therefore:

```text
Anti-UAV source annotations
          │
          ├──────────────► tracking/evaluation semantics
          │
          └──────────────► derived detector training labels
```

YOLO label files are **derived artifacts**.

### 5.3 Required derived-data manifest

A low-cost, high-value enhancement is a generated manifest, for example:

```json
{
  "sequence": "example_sequence",
  "source_frame": 125,
  "modality": "ir",
  "target_present": true,
  "image": "images/train/example_sequence_ir_000125.jpg",
  "label": "labels/train/example_sequence_ir_000125.txt"
}
```

A manifest helps later with:

- linking predictions back to source video;
- tracking evaluation;
- failure analysis;
- recall by target size;
- auditing train/validation membership.

M0 found that an empty YOLO label alone conflates an absent target with a missing or
invalid annotation. M1 therefore requires this manifest and an annotation status while
retaining empty labels for detector-training compatibility.

---

## 6. Detector architecture

### 6.1 Current problem

The detector stack is duplicated or spread across several legacy areas.

The active scripts manipulate `sys.path` so they can import generic names such as:

```text
models.experimental
models.yolo
utils.datasets
utils.general
utils.torch_utils
```

This has several problems:

- repository execution depends on path position;
- generic module names are ambiguous;
- framework internals are maintained locally;
- training/inference/export are tied to a historical implementation;
- old checkpoint/framework compatibility leaks into normal application code;
- static analysis excludes the affected code.

### 6.2 Legacy detector areas

```text
src/detection/
Codes/detect_wrapper/
anti_uav_jittor/anti_uav_edtc_jit/yolov5/
scripts/train.py
scripts/infer.py
scripts/evaluate.py
scripts/export.py
scripts/test.py
```

These are the primary modernization target.

### 6.3 Target detector boundary

The project needs only a **small** detector boundary.

Conceptually:

```python
from dataclasses import dataclass

@dataclass(slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
```

The adapter is responsible for:

```text
framework result
      │
      ▼
project Detection[]
```

The rest of the application should not need to understand an Ultralytics-specific
`Results` object.

### 6.4 Initial detector implementation

The initial prototype implementation is:

```text
Ultralytics
```

for:

- model loading;
- training;
- resume;
- validation;
- prediction;
- detector post-processing;
- standard augmentations;
- export where practical.

This is an R&D/productivity decision, not a final commercial licensing decision.

### 6.5 Future detector substitution

A future permissive implementation such as YOLOX may be evaluated later.

The repository should **not** implement a generalized plugin system now.

A future adapter should only need to satisfy the same small detection boundary:

```text
Ultralytics ─┐
             ├──► Detection[] ─► tracker / evaluator / renderer
YOLOX ───────┘
```

---

## 7. Training architecture

### 7.1 Historical `scripts/train.py` before M3

**Disposition: REPLACED**

The current training script owns generic detector-framework concerns such as:

- model construction;
- YOLO-specific dataloading;
- loss;
- optimizer;
- LR scheduling;
- anchors;
- AMP;
- DDP;
- checkpoint lifecycle;
- validation integration;
- hyperparameter evolution.

Those responsibilities should not remain project-owned.

### 7.2 Current `scripts/train.py`

The replacement is a thin project entry point responsible for:

- parsing project-relevant arguments/config;
- selecting model/checkpoint;
- selecting data YAML;
- selecting output/run name;
- forwarding supported training options;
- invoking the maintained public training API;
- reporting where artifacts are written.

It delegates the training loop to `ultralytics.YOLO.train` and does not import copied
`models.*` or `utils.*` modules.

### 7.3 Configuration

Useful UAV-specific experiment knowledge belongs in version-controlled config.

Examples:

```text
model/checkpoint
image size
epochs
batch
device
workers
optimizer selection
augmentation choices
run name
output root
```

Do not migrate every historical YOLO flag simply because the old script exposes it.

The supported baseline configuration is `configs/training/anti_uav_baseline.yaml`.
Historical `default.yaml` and `finetune.yaml` files remain only for legacy reference and
are not consumed by the modern entry point.

---

## 8. Inference architecture

### 8.1 Current `scripts/infer.py`

**Disposition: REPLACE**

It currently owns:

- loading legacy checkpoints;
- image/video stream loading;
- image-size checking;
- NMS;
- coordinate scaling;
- plotting.

These are generic detector responsibilities.

### 8.2 Target behavior

The modern inference path should support the useful external behavior:

```text
weights/model
source image/video/directory
confidence threshold
IoU threshold
device
output location
annotated output
```

Internally, generic detector work belongs to the external library.

For integration code, normalize output into `Detection[]`.

---

## 9. Detector validation and export

### 9.1 `scripts/evaluate.py`

**Disposition: REPLACE FOR DETECTOR VALIDATION**

Detector validation should use the maintained detector library's validation API.

Preserve the ability to obtain useful detector metrics such as:

- precision;
- recall;
- mAP;
- saved metrics/artifacts.

Do not use this migration as permission to change tracking-benchmark evaluation.

### 9.2 `scripts/export.py`

**Disposition: REPLACE**

Model graph surgery and framework-specific export internals should not be maintained
locally when the detector library provides a supported export path.

Export formats can be introduced when there is a concrete deployment requirement.

### 9.3 `scripts/test.py`

**Disposition: DELETE AFTER PARITY**

If it only exists to support the old YOLO training/validation path, it becomes obsolete
after modern training and detector validation are verified.

---

## 10. Tracking architecture

### 10.1 Current tracking

Relevant areas include:

```text
Codes/tracking_wrapper/
src/tracking/
Codes/demo_detect_track.py
Codes/detect_tracking.py
```

The historical `Codes` documentation references an older PyTorch environment, which is
a sign that the tracker may require compatibility work.

### 10.2 Migration policy

Tracking is **frozen during detector migration**.

Sequence:

```text
1. modern detector works alone
2. modern detector produces project Detection[]
3. adapter feeds existing tracking path
4. combined demo works
5. only then benchmark alternatives
```

Changing detector and tracker simultaneously would make failures difficult to isolate.

### 10.3 Later tracker comparison

Possible later candidates include:

- existing Siamese/SOT tracker;
- ByteTrack;
- BoT-SORT;
- detector + dedicated single-object tracker;
- hybrid redetection/tracking policies.

No candidate is selected merely on architectural taste. Benchmark them against the
actual Anti-UAV requirement.

---

## 11. Anti-UAV evaluation architecture

### 11.1 Important semantic distinction

Anti-UAV tracking is not equivalent to standard detector validation or generic MOT.

The benchmark includes the state in which the target is not visible / does not exist
in the frame.

Therefore evaluation must preserve:

```text
target present + prediction
target present + missing prediction
target absent + no prediction
target absent + false positive
```

### 11.2 Benchmark implementation and generic toolkit

**Disposition: FREEZE / PRESERVE / TEST LATER**

The verified Anti-UAV benchmark implementation is
`anti_uav_jittor/anti_uav410_jit/experiments/anti_uav.py`. The separate
`src/evaluation/toolkit/` contains generic tracking evaluation utilities. Do not
rewrite either during detector modernization or treat their formulas as interchangeable.

Later modernization should:

1. identify benchmark formulas and file formats;
2. write small hand-computable tests;
3. refactor behind project-owned types;
4. compare old and new outputs;
5. remove old code only after deterministic parity.

---

## 12. `Codes/` disposition

### 12.1 `Codes/detect_wrapper`

**Disposition: DELETE AFTER DETECTOR PARITY**

This is legacy generic detector infrastructure.

Keep it temporarily only for:

- historical comparison;
- old checkpoint compatibility;
- rollback while modern detector work is incomplete.

It should not remain part of the normal runtime after parity is established.

### 12.2 `Codes/tracking_wrapper`

**Disposition: FREEZE**

Do not delete before the existing tracker has been evaluated against the modern
detector path.

### 12.3 `Codes/demo_detect_track.py` / `Codes/detect_tracking.py`

**Disposition: ADAPT LATER**

The demo concept is valuable.

The migration target is:

```text
modern detector
      │
      ▼
Detection[]
      │
      ▼
tracker adapter / existing tracker
      │
      ▼
annotated output
```

Consolidate duplicated demos only after the working behavior has been recovered.

---

## 13. Notebooks

Current public repository content includes `notebooks/demo.ipynb`.

### Target notebook policy

Use notebooks for orchestration and visualization, not project implementation.

A future `notebooks/colab_train.ipynb` should contain:

1. GPU/environment check;
2. clone this GitHub repository;
3. dependency installation;
4. download the selected archive with `gdown`;
5. extract to a fresh directory under `/content`, preserving all source media;
6. path/config selection;
7. call to data preparation;
8. call to training;
9. validation;
10. video inference on the preserved videos;
11. result display;
12. optional Google Drive checkpoint persistence and resume example.

Do not embed:

- dataset parsing algorithms;
- detector training loops;
- NMS implementation;
- benchmark metric implementation;
- large unversioned hyperparameter dictionaries when config files are appropriate.

Use `/content` for active high-throughput work and Drive for persistent source archives
and important outputs.

---

## 14. Tests and quality tooling

The repository already has a useful tooling direction in `pyproject.toml`.

The modernization should **shrink legacy exceptions over time**.

Today, major areas are excluded from Pyright. The end-state should be:

```text
new detector code        checked
new data changes         checked
new integration adapters checked
legacy-only folders      isolated or deleted
```

Do not attempt to make the entire historical repository perfectly typed in one task.

Tests should increasingly encode behavior we care about:

### Data

- annotation parsing;
- bbox conversion;
- absent target handling;
- sample rate;
- train/val outputs;
- generated dataset YAML;
- source-to-generated manifest if added.

### Detector

- framework result -> `Detection`;
- empty detections;
- confidence/class conversion;
- CLI/config smoke checks.

### Integration

- one image inference smoke test;
- one short video smoke test when practical;
- tracker adapter tests later.

### Evaluation

- hand-computable presence/absence cases.

---

## 15. Model and data provenance

Model and dataset provenance should be documented even during prototyping.

At minimum record:

```text
model/checkpoint source
training framework/version
training dataset/source
experiment config
code revision
artifact purpose: prototype vs production candidate
```

Historical weights should not be silently treated as commercially cleared production
artifacts.

Similarly, the availability of a public Anti-UAV dataset does not by itself answer all
commercial training-rights questions. Commercial rights should be reviewed separately
before a production release.

---

## 16. Desired repository shape after detector modernization

This is a directional shape, not a command to reorganize everything immediately.

```text
anti-uav410/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODERNIZATION.md
│   └── TRAINING.md
├── configs/
├── scripts/
│   ├── prepare_data.py          # preserved/adapted
│   ├── train.py                 # thin modern wrapper
│   ├── infer.py                 # thin modern wrapper
│   ├── evaluate.py              # detector validation wrapper
│   └── export.py                # supported export wrapper
├── src/
│   ├── detection/
│   │   ├── results.py
│   │   └── ultralytics_adapter.py
│   ├── tracking/                # temporarily preserved
│   └── evaluation/
│       └── toolkit/             # temporarily preserved
├── Codes/
│   └── tracking_wrapper/        # temporary until tracker decision
├── notebooks/
│   ├── demo.ipynb
│   └── colab_train.ipynb
└── tests/
```

The exact filenames may change as Codex audits the local checkout.

The architectural requirement is the **responsibility boundary**, not cosmetic tree
perfection.

---

## 17. Success criteria for the architecture

The detector modernization is structurally successful when:

- data preparation retains Anti-UAV-specific semantics;
- new training no longer imports copied `models.*` / `utils.*`;
- new inference no longer uses `sys.path` to find detector internals;
- detector validation uses a maintained public API;
- detector output can be normalized into a project-owned structure;
- a fresh Colab can train and run video inference;
- modern detector output can be integrated with the existing tracker;
- legacy detector code is deleted only after parity;
- tracking benchmark semantics have not been silently changed;
- the repository is smaller and easier to reason about than before.

---

## 18. Decision rule

When evaluating an old component, ask:

> Is this Anti-UAV-specific knowledge or generic ML framework machinery?

If it is Anti-UAV-specific and useful, preserve or adapt it.

If it is generic detector framework machinery now supplied by a maintained library,
replace it.

If its value is uncertain, freeze it until a test or benchmark can answer the question.
