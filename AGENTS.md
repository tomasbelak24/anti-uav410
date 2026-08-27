# AGENTS.md

## Mission

Modernize the existing `pgryko/anti-uav410` repository **in place**.

Do not rebuild the project from scratch. Preserve useful Anti-UAV-specific behavior
while replacing obsolete or duplicated generic ML framework code with maintained
libraries.

Near-term goal:

```text
prepare Anti-UAV data
→ train a modern detector in Colab
→ validate
→ run on unseen video
→ integrate with the existing tracker
→ remove superseded legacy detector code
```

## Source of truth

Use this file as a map, not an encyclopedia.

- `docs/ARCHITECTURE.md` — current repository responsibilities and boundaries.
- `docs/MODERNIZATION.md` — milestone plan, acceptance criteria, and delete gates.

For a scoped task, read the relevant architecture sections and the **active milestone**.
Do not load or act on later milestones unless needed for a dependency decision.

## Golden rules

1. **Preserve working Anti-UAV domain behavior.**
2. **Do not vendor another detector framework into this repo.**
3. **Replace one subsystem at a time.**
4. **Do not delete legacy code before its replacement passes its delete gate.**
5. **Reduce framework coupling rather than spreading library-specific result types.**
6. **Prefer adapting working project-specific code over rewriting it for aesthetics.**

## Repository boundaries

### Preserve / adapt

- `scripts/prepare_data.py`
- target `exist` / absent-target semantics
- useful dataset conversion behavior
- useful demos and sample videos
- tests that encode real behavior
- Anti-UAV tracking/evaluation semantics

### Replace incrementally

- `scripts/train.py`
- `scripts/infer.py`
- detector-focused `scripts/evaluate.py`
- `scripts/export.py`
- legacy detector internals under `src/detection`
- `Codes/detect_wrapper`

### Freeze unless the active milestone says otherwise

- `Codes/tracking_wrapper`
- `src/tracking`
- `src/evaluation/toolkit`
- Jittor-specific code
- historical weights/checkpoints

## Working method

For non-trivial changes:

1. inspect the relevant code;
2. read the active milestone;
3. identify files likely to change;
4. identify behavior that must survive;
5. identify explicit non-goals;
6. implement only that milestone;
7. run relevant checks;
8. report unrelated issues instead of fixing them opportunistically.

If repository reality conflicts with the docs, report the discrepancy and choose the
least destructive path until the plan is corrected.

Do not turn a migration task into broad cleanup.

## Git safety

Unless explicitly instructed otherwise:

- do not push;
- do not alter remotes;
- do not force-push or rewrite history;
- do not amend existing commits;
- do not delete untracked user files;
- do not remove historical weights or legacy code before its delete gate passes.

## Dependencies

Add dependencies only for a clear repository responsibility.

Prefer maintained public APIs. Do not remove existing dependencies simply because they
look unused if frozen tracking/evaluation code may still require them.

Ultralytics is acceptable for the **prototype detector path**. It is not a final
commercial licensing decision.

## Detector boundary

Keep the boundary small. Application code should consume a project-owned detection
shape rather than framework-specific result objects, conceptually:

```python
@dataclass(slots=True)
class Detection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
```

Do not build a generalized plugin framework now.

## Data rules

Generated YOLO labels are a **derived training representation**.

When changing preparation, preserve enough information to recover:

- sequence;
- source frame index;
- modality;
- target presence/absence;
- bounding box;
- split.

Never silently discard absent-target semantics.

## Tracking and evaluation rules

Do not modernize tracking during detector migration.

Keep detector validation separate from Anti-UAV tracking evaluation. Anti-UAV includes
target-absence behavior and must not be reduced to generic MOT or mAP semantics.

## Validation

Use repository-configured tools where relevant:

```bash
pytest
ruff check .
pyright
```

Run focused tests for the changed behavior and broader checks when feasible.

The repository has known legacy lint/type exclusions. Do **not** expand exclusions to
make new code pass. If a broad check fails only because of existing legacy code, report
that fact instead of repairing unrelated areas.

GPU-heavy validation belongs to milestones that explicitly require Colab/GPU testing.

## Completion report

At the end of a migration task report:

1. files changed;
2. behavior preserved;
3. intentional behavior changes;
4. checks run and results;
5. remaining legacy dependencies;
6. manual/GPU validation still required;
7. unrelated issues discovered but intentionally not changed.

Do not proceed to the next modernization milestone unless explicitly asked.
