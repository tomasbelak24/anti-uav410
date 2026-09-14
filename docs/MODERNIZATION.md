# Anti-UAV410 Modernization Plan

**Status:** Active migration plan
**Target:** Existing `pgryko/anti-uav410` repository, modified in place
**Primary near-term outcome:** Reliable modern detector training and demo without
discarding useful Anti-UAV-specific code
**Codex policy:** Work one milestone at a time; do not advance automatically

---

## 1. Objective

Modernize the existing repository rather than replacing it.

The immediate target is:

```text
clone/install existing repository
          │
          ▼
prepare Anti-UAV data
          │
          ▼
train modern detector in Colab
          │
          ▼
validate
          │
          ▼
run on unseen video
          │
          ▼
integrate with existing tracker
          │
          ▼
remove superseded legacy detector framework
```

This plan deliberately defers:

- tracker replacement;
- full benchmark-evaluator rewrite;
- production deployment optimization;
- YOLOX implementation;
- final commercial detector/licensing choice.

---

## 2. Modernization principles

### 2.1 Preserve behavior before implementation style

The migration must preserve useful behavior before optimizing architecture.

Especially preserve:

- Anti-UAV annotation semantics;
- target `exist` / absent-target behavior;
- useful frame sampling;
- detector-training dataset generation;
- existing tracking behavior until benchmarked;
- Anti-UAV tracking evaluation semantics;
- historical artifacts required for comparison.

### 2.2 Generic framework code is replaceable

Do not spend time repairing old copies of generic YOLO internals if maintained public
APIs provide the same responsibility.

### 2.3 One subsystem at a time

Detector modernization is allowed to touch detector/data integration.

It is not permission to rewrite tracking.

### 2.4 Delete after proof

A legacy implementation is removed only when its replacement has passed the milestone's
delete gate.

### 2.5 Small boundaries, not architecture for architecture's sake

Use a small project-owned detection representation so application code is not tightly
coupled to one detector result object.

Do not build a broad detector-plugin framework in the initial migration.

---

## 3. Milestone overview

| Milestone | Outcome |
|---|---|
| M0 | Baseline and repository assumptions verified |
| M1 | Data preparation verified and protected by tests |
| M2 | Modern detector dependency + small detector boundary introduced |
| M3 | Legacy detector training replaced |
| M4 | Legacy detector inference replaced |
| M5 | Detector validation/export path replaced |
| M6 | Clean Google Colab workflow works end-to-end |
| M7 | Real detector trained, evaluated, and demonstrated |
| M8 | Modern detector integrated with existing tracker |
| M9 | Superseded legacy detector code removed |
| M10 | Tracking alternatives benchmarked |
| M11 | Anti-UAV evaluator modernized with parity tests |
| M12 | Production detector/licensing/deployment decision |

Do not combine milestones unless explicitly approved.

---

# M0 — Freeze and verify the baseline

## Goal

Establish a trustworthy map of the local checkout before code changes.

This milestone is primarily an **audit**.

## Inspect

At minimum inspect:

```text
README.md
pyproject.toml
docs/
configs/
scripts/prepare_data.py
scripts/train.py
scripts/infer.py
scripts/evaluate.py
scripts/export.py
scripts/test.py
src/detection/
src/tracking/
src/evaluation/
Codes/detect_wrapper/
Codes/tracking_wrapper/
Codes/demo_detect_track.py
Codes/detect_tracking.py
notebooks/
tests/
weights/
```

Trace:

1. data preparation;
2. detector training;
3. detector inference;
4. detector validation;
5. export;
6. detect+track integration;
7. Anti-UAV benchmark evaluation.

## Do not modify

Do not modify runtime code during the M0 audit.

Documentation corrections are allowed only after discrepancies have been reported.

## Required output

Produce a written audit containing:

- actual repository tree relevant to the migration;
- actual entry points;
- import/path hacks;
- duplicated detector code;
- test coverage by subsystem;
- historical checkpoints and where they are referenced;
- differences between repository reality and `docs/ARCHITECTURE.md`;
- proposed corrections to later milestones if necessary.

## Acceptance criteria

M0 passes when:

- the data flow is traced;
- the detector flow is traced;
- tracker integration is traced;
- benchmark evaluator location is known;
- current test commands are known;
- assumptions in this plan are either confirmed or corrected.

## Verified M0 result

M0 was audited against public `main` commit
`001b79550070d6ec1b3d81bda1d93a85302234f7` on 2026-08-26. The recoverable local
baseline is branch `codex/baseline-m0`; implementation work starts on
`codex/m1-data-preparation`.

Corrections carried into M1-M6:

- the user-verified Colab archive is Google Drive file
  `1F0nGafdWP4PddmqVDLFNpHmukEFRpX8Y`, is publicly downloadable with `gdown`, and
  matches the current RGBT filename/layout contract;
- real-data M1 verification locally audited the 5.6 GB `Anti-UAV-RGBT.zip` associated
  with Google Drive file `1VZFq7g-z5-k0VEkGLKW2i29RzGd47dRD`: it has root-level
  `train`/`val`/`test` splits containing 160/67/91 video-only RGBT sequences;
- the separate legacy Anti-UAV410 benchmark layout uses numbered JPEGs plus
  `IR_label.json`; JPEG compatibility is useful but is not part of the audited RGBT ZIP;
- archives must be extracted into a fresh directory because extraction does not remove
  stale files and can otherwise create a misleading mixed dataset tree;
- all audited IR annotation arrays align, but 179 train, 72 validation, and 43 test
  frames contain the contradictory combination `exist=1` and a zero-size box; M1 keeps
  those distinct from genuine absence through `annotation_status=invalid_bbox`;
- source videos must remain available after preparation for later video inference;
- an empty YOLO label cannot distinguish source absence from invalid/missing metadata,
  so the M1 source manifest is required;
- the Anti-UAV benchmark implementation is
  `anti_uav_jittor/anti_uav410_jit/experiments/anti_uav.py`, while
  `src/evaluation/toolkit/` is a separate generic toolkit;
- there are three copied detector stacks: `src/detection`, `Codes/detect_wrapper`, and
  `anti_uav_jittor/anti_uav_edtc_jit/yolov5`; `demo_ui.py` imports the third;
- `tests/test_training_smoke.py` exercises `Codes/detect_wrapper`, not the primary
  `scripts/train.py` + `src/detection` path;
- detector/tracker integration expects an absent
  `Codes/detect_wrapper/weights/best.pt`; root `demo_ui.py` instead expects
  `weights/drone_detector.pt`.

## Baseline safety

Before destructive later work, create a recoverable Git point, for example a branch or
tag under the user's chosen workflow.

Codex must not push or rewrite remote history unless explicitly asked.

---

# M1 — Verify and protect dataset preparation

## Goal

Keep `scripts/prepare_data.py` as the working data-preparation base while making its
critical behavior explicit and testable.

Do **not** redesign the data layer.

## Primary file

```text
scripts/prepare_data.py
```

## Inspect also

```text
tests/
README.md
docs/TRAINING.md
any existing dataset fixtures
any config/YAML files produced by the script
```

## Behavior to preserve

At minimum:

- parse Anti-UAV JSON;
- `exist` flags;
- `gt_rect`;
- bbox conversion;
- IR processing;
- benchmark JPEG + `IR_label.json` processing;
- existing RGB/both capability if currently supported;
- train/validation split handling;
- `sample_rate=5` workflow;
- source frame numbering;
- empty/absent-target label behavior;
- image/label generation;
- dataset YAML generation.

## Recommended changes

Only make changes that improve confidence or future integration.

Candidates:

1. validate `sample_rate > 0`;
2. clearer errors for unreadable videos/annotation mismatches;
3. explicit checks for impossible boxes;
4. deterministic output behavior;
5. summary counts for frames, present targets, absent targets;
6. a small JSON Lines source manifest that preserves split, sequence, source frame,
   modality, source presence, source box, generated paths, and annotation status.
7. automatic per-sequence selection between complete IR video and JPEG layouts;
8. separate discovered, processed, skipped, video-source, and JPEG-source counts.
9. root-level/singly-wrapped ZIP detection and refusal to merge with unrelated files.

The manifest is required because empty detector labels alone lose the distinction
between true target absence and invalid or missing annotations.

## Tests to add or verify

Use tiny synthetic fixtures.

At minimum test:

```text
annotation parser reads exist + gt_rect
bbox conversion produces correct normalized values
invalid/empty bbox is handled
sample_rate is respected
JPEG frame order remains aligned with IR_label.json
target-present frame creates label
target-absent frame does not create a false positive label
train/val output locations are distinct
dataset YAML points to expected paths
```

Do not require the full Anti-UAV dataset in unit tests.

## Do not touch

```text
scripts/train.py
scripts/infer.py
scripts/evaluate.py
Codes/tracking_wrapper/
src/tracking/
src/evaluation/toolkit/
```

except for a tiny test/import change proven necessary for M1.

## Acceptance criteria

M1 passes when:

- existing intended data-preparation behavior is still available;
- the core behavior is covered by lightweight tests;
- a small local fixture can be prepared;
- the verified root-level ZIP layout resolves without a manual path search;
- reruns cannot silently mix stale derived outputs;
- source videos and JPEGs remain untouched by preparation and explicit overwrite handling;
- no detector/tracker subsystem was rewritten.

## Delete gate

Nothing significant is deleted in M1.

---

# M2 — Introduce the modern detector boundary

## Goal

Introduce the smallest practical boundary between the project and a modern detector
library.

Ultralytics is the initial R&D detector implementation.

M2 does **not** replace training yet.

## Inspect

```text
pyproject.toml
src/detection/
scripts/infer.py
Codes/detect_wrapper/
Codes/demo_detect_track.py
demo_ui.py
anti_uav_jittor/anti_uav_edtc_jit/yolov5/
tests/
```

## Required design

Create or identify a project-owned detection representation conceptually equivalent to:

```python
@dataclass(slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
```

Provide a small Ultralytics adapter capable of:

```text
load model
predict image/frame/source
convert results -> Detection[]
```

The exact filenames should follow the local repository structure discovered in M0.
Do not reorganize the entire package merely to match this document.

## Dependency change

Add a constrained Ultralytics dependency suitable for the prototype workflow.

Do not aggressively remove the existing Torch/OpenCV/etc. dependencies yet because
frozen tracking code may still rely on them.

Record the resolved Ultralytics version in experiment output once real training begins.

## Required tests

At minimum:

- construct `Detection`;
- convert a mocked framework result;
- empty result -> empty list;
- confidence/class/bbox fields survive conversion;
- adapter import does not require `sys.path` mutation.

A small real-model CPU inference smoke test is useful if download/network behavior can
be made optional and does not destabilize normal tests.

## Do not touch

Do not rewrite:

```text
scripts/train.py
Codes/tracking_wrapper/
src/tracking/
src/evaluation/toolkit/
```

Do not delete old detector code yet.

## Acceptance criteria

M2 passes when:

- the modern detector adapter can be imported normally;
- one image/frame can conceptually flow to `Detection[]`;
- new detector code does not import copied `models.*` / `utils.*`;
- old detector remains available for still-unmigrated legacy paths during the transition.

## Implemented M2 boundary

The modern boundary lives in:

```text
src/detection/results.py
src/detection/ultralytics_adapter.py
```

`Detection` contains only absolute `xyxy` coordinates, confidence, and class ID.
`UltralyticsDetector` loads an Ultralytics model and accepts one image or video frame per
call, returning `Detection[]`. Keeping prediction frame-based avoids losing frame identity
when later video inference is implemented in M4.

Ultralytics is constrained to `>=8.4,<9` for the prototype path. It remains an R&D
dependency rather than a final commercial licensing decision. The adapter imports
Ultralytics only when a model is constructed, so importing the project-owned detection
shape does not initialize the framework or load a model.

The module names deliberately avoid `types.py` and `ultralytics.py`: legacy entry points
still prepend `src/detection` to `sys.path`, so those generic names could shadow Python's
standard `types` module or the third-party `ultralytics` package before M4 removes the path
mutation.

## Delete gate

No legacy detector deletion yet.

---

# M3 — Replace legacy detector training

## Goal

Replace the old project-owned YOLO training loop with a thin modern training entry
point.

This is the highest-value code reduction in the migration.

## Replace

```text
scripts/train.py
```

The old implementation may be preserved temporarily through Git history or a clearly
isolated legacy path if M0 proves that old checkpoint reproducibility requires it.

Do not maintain two normal supported training paths long-term.

## Responsibilities of the new training entry point

It may own:

- argument/config parsing;
- model/checkpoint selection;
- dataset YAML selection;
- image size;
- epochs;
- batch;
- device;
- workers;
- experiment/run name;
- output directory;
- selected supported augmentations;
- resume request;
- logging of the effective run configuration.

It should delegate:

- model construction;
- training loop;
- loss;
- optimizer implementation;
- NMS;
- internal checkpoint format;
- detector validation internals

to the maintained library.

## Configuration

Prefer a small version-controlled experiment config.

Do not translate every historical YOLOv5 flag.

Keep parameters that are actually useful to the UAV experiments.

A baseline config should be sufficient to reproduce the first Colab run.

## Resume

The training path must support resuming from the library-supported checkpoint/state
without manually reconstructing every historical argument.

## Tests

Unit/smoke tests should verify:

- config/CLI parsing;
- correct forwarding of key arguments;
- output path/run naming;
- resume path forwarding;
- no import from copied `models.*` / `utils.*`.

Mock the heavy training call in normal unit tests.

## Manual/GPU acceptance

A later Colab smoke test must demonstrate:

```text
prepared dataset
    -> 1 epoch
    -> best/last checkpoint or equivalent expected artifacts
```

M3 can be code-complete before the GPU test, but it is not **integration-complete**
until M6.

## Do not touch

Do not modernize:

```text
tracking
Anti-UAV benchmark evaluator
detect+track demo
```

Do not delete `Codes/detect_wrapper` yet.

## Acceptance criteria

M3 passes when:

- new training no longer imports old `models` or `utils`;
- a normal help/config command works;
- lightweight tests pass;
- the training call is thin and uses public library APIs;
- resume is represented;
- the old training loop is no longer the supported default.

## Delete gate

Do not delete the full old detector framework merely because training was replaced.
Inference and integration still need to migrate.

---

# M4 — Replace legacy detector inference

## Goal

Replace `scripts/infer.py` legacy YOLO internals while preserving useful external
inference behavior.

## Useful behavior to preserve

Where still relevant:

- model/weights selection;
- image input;
- video input;
- directory input;
- confidence threshold;
- IoU threshold;
- device;
- annotated output;
- predictable output path.

Do not preserve obscure legacy flags without a use case.

## Integration requirement

Use the M2 detector boundary for application-facing output.

The script may allow library-native saving for simple standalone inference, but
detector+tracker integration must have access to normalized `Detection[]`.

The root `demo_ui.py` is a separate inference entry point importing
`anti_uav_jittor/anti_uav_edtc_jit/yolov5`. For the smallest M4, the Colab workflow may
use the modern `scripts/infer.py` path and leave this UI explicitly legacy. It must not
be mistaken for a migrated path or silently used by M6.

## Tests

At minimum:

- CLI/config parsing;
- mocked image prediction;
- empty prediction behavior;
- output path behavior;
- threshold forwarding;
- no import from copied legacy detector modules.

If practical, run one short local video/image smoke test.

## Do not touch

Tracking remains frozen.

Do not modify benchmark metrics.

## Acceptance criteria

M4 passes when:

- normal inference does not mutate `sys.path`;
- normal inference does not import copied `models.*` / `utils.*`;
- an image/video can be processed through the modern detector path;
- annotated output can be produced;
- integration code can receive `Detection[]`.

## Delete gate

Still do not delete all old detector code. Detector validation/export and tracker
integration are not yet complete.

---

# M5 — Replace detector validation and export

## Goal

Remove the remaining need to maintain YOLO-specific detector validation/export
internals.

## Detector validation

Modernize detector-focused:

```text
scripts/evaluate.py
```

or replace it with a better-named thin entry point if M0 shows the existing name is
misleading.

Use the maintained detector validation API.

Required output should include useful detector metrics such as:

- precision;
- recall;
- mAP;
- location of detailed result artifacts.

## Important boundary

Do **not** replace:

```text
anti_uav_jittor/anti_uav410_jit/experiments/anti_uav.py
src/evaluation/toolkit/
```

The first path is the verified Anti-UAV410 benchmark implementation; the second is a
separate generic tracking toolkit. Neither belongs to detector validation M5.

Detector evaluation and Anti-UAV tracking evaluation are separate systems.

## Export

Replace old graph-surgery/export logic with supported library export where a concrete
format is required.

Do not add every possible export dependency by default.

Support only formats justified by current prototype needs.

ONNX may be useful later; TensorRT is a deployment milestone, not a prerequisite for
the investor prototype.

## `scripts/test.py`

If M0 confirms this file only exists to support the legacy detector training/validation
path, mark it for deletion after M5 checks pass.

## Tests

- validation command/config forwarding;
- metrics/result-path handling;
- export command forwarding if export is retained in M5;
- no legacy `models.*` / `utils.*` imports in normal detector evaluation.

## Acceptance criteria

M5 passes when:

- detector validation uses the modern public API;
- detector metrics can be generated;
- supported export uses a modern public API;
- normal detector workflows no longer need old `scripts/test.py`.

## Delete gate

After M5, deletion of some detector-only scripts is allowed **only if** M3–M5 are
independently working.

Do not delete the detector wrapper required by the old detect+track demo until M8.

---

# M6 — Create the clean Google Colab workflow

## Goal

Provide the primary practical workflow for training experiments.

The notebook is an orchestrator, not a second implementation.

## File

Prefer:

```text
notebooks/colab_train.ipynb
```

The existing `demo.ipynb` can remain unless it is clearly superseded.

## Notebook flow

Keep the notebook concise.

Recommended sequence:

```text
1. GPU/environment check
2. clone the GitHub repository
3. install project/dependencies in the active Colab kernel
4. install/use `gdown`
5. download the selected source archive with `gdown`
6. extract the archive to a fresh directory under `/content`
7. define dataset + output paths
8. run dataset preparation without removing source videos
9. run a 1-epoch training smoke test
10. validate
11. run video inference on a preserved source/sample video
12. display output video
13. optionally persist important checkpoints/results to Drive
14. show resume after runtime restart
```

## Environment rule

Use one Python environment: the Colab notebook kernel.

Avoid mixing:

```text
uv-created private venv
another interpreter path
notebook kernel imports
```

in one workflow.

## Storage rule

Prefer:

```text
Public Google Drive file via gdown
    -> source archive
/content
    -> extracted source videos / active prepared dataset / training IO
Mounted personal Google Drive (optional)
    <- selected important checkpoints and outputs
```

Do not train directly against a large dataset on mounted Drive when local `/content`
storage is practical.

## Source acquisition

The supported M6 workflow clones the GitHub repository. Dataset acquisition is separate:
use `gdown` with an explicitly selected verified file ID, rather than embedding Drive download
logic inside the repository's data converter.

## Required smoke test

A **fresh Colab GPU runtime** should demonstrate:

- project/dependencies install;
- public archive download succeeds using the selected recorded `gdown` file ID;
- data path resolves;
- source videos remain available after preparation;
- prepared data is usable;
- 1 epoch completes;
- checkpoint artifacts are created;
- validation runs;
- video inference runs;
- annotated video is viewable;
- important outputs can persist to Drive.

Then restart the runtime and verify resume from the persisted checkpoint.

## Do not include yet

- hyperparameter tuning;
- tracker replacement;
- YOLOX;
- TensorRT optimization;
- large production infrastructure.

## Acceptance criteria

M6 passes only after the fresh-runtime smoke test has actually been performed.

---

# M7 — Train the first serious detector

## Goal

Produce a technically credible prototype model and reproducible experiment.

Infrastructure work is no longer the main focus.

## Experiment requirements

Record at minimum:

```text
model/checkpoint
framework version
dataset/preparation configuration
image size
epochs
batch
augmentation settings
seed if supported
GPU
run/output name
code revision
```

## Metrics

Start with detector metrics:

- recall;
- precision;
- mAP.

Also begin recording UAV-relevant analysis when practical:

- recall by target pixel size;
- false positives on representative videos;
- qualitative failures.

Do not make investor-facing performance claims that cannot be reproduced.

## Output

Preserve:

```text
effective config
best checkpoint
last checkpoint
metrics/results
selected prediction video
```

## Acceptance criteria

M7 passes when a model can be retrained from recorded configuration and demonstrates
useful UAV detection on held-out/unseen video.

---

# M8 — Integrate the modern detector with the existing tracker

## Goal

Recover the combined detect+track demo without replacing the tracker.

## Inspect

```text
Codes/demo_detect_track.py
Codes/detect_tracking.py
Codes/tracking_wrapper/
src/tracking/
legacy detector wrapper interface
```

Determine what exact detection representation the tracker bridge expects.

## Strategy

Add the smallest adapter needed:

```text
modern detector
      │
      ▼
Detection[]
      │
      ▼
legacy integration shape
      │
      ▼
existing tracker
```

Do not rewrite tracking algorithms in M8.

## Tests

Use a short deterministic or representative video.

Verify:

- detector initializes;
- detection is handed to tracker correctly;
- tracker initializes when expected;
- output frame/video is produced;
- target loss does not crash the pipeline.

## Acceptance criteria

M8 passes when the modern detector can drive the existing tracking demo end-to-end.

## Delete gate

Only after M8 passes may the old detector wrapper required solely by the legacy
detect+track integration be removed.

---

# M9 — Remove superseded legacy detector code

## Goal

Physically reduce the repository after all normal detector flows have migrated.

## Candidate deletion

Subject to M0 findings and M3–M8 parity:

```text
Codes/detect_wrapper/
old copied detector `models/`
old copied detector `utils/`
obsolete legacy training/eval helper scripts
old detector-only path hacks
old detector-specific test helper
```

## Before deletion

Search the repository for imports/references.

No supported path may still depend on the files.

Preserve old behavior through Git history rather than keeping dead copies inside the
normal code tree.

## After deletion

Run:

- relevant tests;
- detector training smoke/config checks;
- inference smoke test;
- detector validation smoke test;
- detect+track integration smoke test.

## Acceptance criteria

M9 passes when normal supported workflows no longer reference the deleted detector
framework and the repository becomes materially smaller/simpler.

---

# M10 — Benchmark tracking alternatives

## Goal

Choose tracking based on measured Anti-UAV behavior, not library popularity.

## Compare

At minimum consider:

```text
existing tracker
ByteTrack
another justified modern alternative
```

A dedicated single-object tracker may be more appropriate than MOT for some benchmark
semantics.

## Required evaluation dimensions

- target-present overlap/tracking quality;
- target-loss behavior;
- false-positive track persistence;
- re-acquisition;
- track continuity;
- runtime/latency;
- integration complexity.

Do not assume detector-driven tracking is equivalent to ground-truth-initialized SOT.

## Output

Produce a short decision record with:

- tested trackers;
- config;
- data split;
- metrics;
- qualitative failure cases;
- recommendation.

---

# M11 — Modernize Anti-UAV benchmark evaluation

## Goal

Make benchmark evaluation testable and maintainable without changing its meaning.

## Process

1. identify exact current formulas and file formats;
2. preserve a small set of known legacy outputs;
3. write hand-computable metric tests;
4. implement/refactor project-owned evaluation types;
5. compare new output to legacy output;
6. document any intentional discrepancy;
7. remove superseded evaluator code only after parity.

## Required edge cases

```text
target present + correct prediction
target present + partial overlap
target present + missing prediction
target absent + no prediction
target absent + false prediction
```

## Acceptance criteria

M11 passes when evaluation is deterministic, independently testable, and compatible
with the selected Anti-UAV benchmark definition.

---

# M12 — Production detector and licensing decision

## Goal

Choose the actual production detector stack only after the technical prototype has
evidence.

## Compare paths

### Path A — Ultralytics commercial path

Use if technical advantage justifies commercial licensing and the business terms are
acceptable.

### Path B — permissive implementation

Train a fresh production candidate through an implementation with licensing acceptable
to the company, using commercially cleared training data and documented model
provenance.

Possible candidates must be evaluated at decision time; do not hard-code today's
alternative into the architecture prematurely.

## Required production provenance

Before calling a model production-ready, document:

```text
code/framework license
pretrained weight provenance
training-data rights
annotation rights
training framework/version
model lineage
third-party notices
company ownership/assignment of internal IP
```

This is a business/legal review gate, not a reason to block the R&D prototype.

---

## 4. What must not happen during M0–M9

The following are anti-goals:

- rewriting the entire repository;
- replacing working data preparation with a new framework for aesthetics;
- replacing detector and tracker in one task;
- deleting benchmark semantics before tests exist;
- adding YOLOX merely to prove replaceability;
- implementing custom NMS/loss/export code that a maintained library already provides;
- preserving every old YOLO CLI flag;
- making broad unrelated formatting changes in migration patches;
- expanding Ruff/Pyright ignores for new code;
- deleting historical weights without checking whether they are needed for comparison;
- making production licensing claims from prototype artifacts.

---

## 5. Codex task protocol

For milestones larger than a trivial edit, use two passes.

### Pass A — plan/audit

Prompt shape:

```text
Goal:
<one milestone outcome>

Context:
Read AGENTS.md, docs/ARCHITECTURE.md, and the active milestone in
docs/MODERNIZATION.md.

Inspect:
<specific files/directories>

Constraints:
<do-not-touch boundaries>

Done when:
Provide a repo-specific implementation plan with:
- files to modify/add,
- behavior to preserve,
- tests/checks,
- legacy code that remains,
- uncertainties/discrepancies.

Do not modify code yet.
```

Review the plan before implementation.

### Pass B — implement

Prompt shape:

```text
Implement the approved plan for milestone <Mx> only.

Stay inside the milestone scope.

Run the acceptance checks defined in docs/MODERNIZATION.md.

If repository reality conflicts with the plan, make the least destructive choice and
report the discrepancy rather than silently expanding scope.

Do not proceed to the next milestone.

At completion report:
1. files changed,
2. behavior preserved,
3. behavior intentionally changed,
4. tests/checks run and results,
5. remaining legacy dependencies,
6. manual/GPU validation still required,
7. unrelated issues discovered but not changed.
```

---

## 6. Recommended first Codex prompt

Use this before any modernization code change:

```text
Audit M0 only.

Read:
- AGENTS.md
- docs/ARCHITECTURE.md
- docs/MODERNIZATION.md

Do not modify runtime code.

Verify the documentation against the local checkout.

Trace:
1. dataset preparation,
2. detector training,
3. detector inference,
4. detector validation,
5. detector export,
6. detector + tracker integration,
7. Anti-UAV benchmark evaluation.

For each, identify:
- actual entry point,
- files/modules used,
- sys.path/path assumptions,
- framework coupling,
- tests,
- expected model/checkpoint artifacts.

Also identify:
- duplicated detector framework code,
- commands in pyproject.toml that do not match real callables,
- dependencies that appear to exist only for legacy areas,
- anything in ARCHITECTURE.md or MODERNIZATION.md that is incorrect.

Done when:
Return a concrete audit and proposed documentation corrections for M1-M6.

Do not implement M1 or any later milestone.
```

---

## 7. Review checklist after every Codex milestone

Before accepting a patch, ask:

### Scope

- Did it change only the intended subsystem?
- Did it perform unrelated cleanup?

### Behavior

- What behavior was preserved?
- What behavior changed intentionally?
- Is any useful legacy behavior now inaccessible?

### Coupling

- Did the patch reduce dependence on copied framework internals?
- Did it introduce a new unnecessary abstraction or dependency?

### Verification

- Which tests ran?
- Are failures pre-existing or new?
- Does this milestone require a manual Colab/GPU check?

### Deletion

- Was anything deleted?
- Did its delete gate actually pass?
- Is Git history sufficient for historical recovery?

### Next step

- Is the milestone genuinely complete?
- What evidence is still missing before proceeding?

---

## 8. Definition of success for the modernization program

The program has achieved its near-term purpose when all of the following are true:

```text
data preparation remains reliable
modern detector trains in Colab
training can resume
detector validation works
video inference works
modern detector integrates with existing tracker
old detector framework is no longer required by normal workflows
superseded detector code has been removed
Anti-UAV tracking semantics remain intact
experiments are reproducible enough to support technical claims
```

The final production detector, commercial licensing path, and tracker choice may still
change later. The architecture should make those decisions possible without repeating
the repository cleanup.
