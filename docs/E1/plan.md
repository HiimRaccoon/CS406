# E1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL for future authorized execution: use `executing-plans` for native execution, or `subagent-driven-development` if that method is explicitly selected. Execute one reviewed task at a time; checkbox steps track completion. This document does not select an execution method or authorize implementation.

**Goal:** Build the official E1 full-image baseline with reproducible four-LR tuning and three disjoint final seeds, following the reviewed eight-epoch protocol.

**Architecture:** Keep official E1 behavior in a small `src/e1/` package. Reuse frozen `src/preprocessing/` directly; deliberately adapt tested pilot mechanics into E1-owned modules with explicit official identities, rather than treating the pilot runner as the official implementation. Separate single-run execution, campaign declarations/selection, and selected-checkpoint reporting.

**Tech Stack:** Python 3.11, PyTorch/TorchVision, Pillow, NumPy, pytest, JSON/CSV, and single-GPU UIT/Slurm. Actual official package/runtime adoption remains R8 in the specification; do not change dependencies or introduce a framework before that review.

**Spec:** [specs.md](specs.md), reviewed E1 decisions recorded through commit `2d784446424005f727dd01a86d97d06775ad7d0d`.

**Status:** DRAFT FOR REVIEW, documentation only. The request to write this plan does not approve the whole specification, resolve its numeric review register, create runnable files, or authorize training. [tasks.md](tasks.md) is the dependency/completion ledger; this plan owns interfaces and verification details.

## Global Constraints

- E1 only: ConvNeXt-Tiny, one `896 x 384` global view, `[N,124]` logits, full fine-tuning from epoch 1.
- Reuse exact fixed membership and label IDs: 34997 train, 7497 val, 7502 test; split seed 406 never regenerates membership.
- Frozen preprocessing v1 and all four split artifacts remain byte-for-byte unchanged. Import the existing preprocessing implementation; do not duplicate its geometry or change its backend.
- Pin `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, filename `convnext_tiny-983f1562.pth`, full SHA-256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.
- Natural distribution, standard unweighted CE, `label_smoothing=0`, no class weighting/balancing; head LR multiplier 1, no LLRD.
- Maximum stochastic depth 0.1 with the selected TorchVision block-linear schedule; brightness/contrast amplitudes 0.10/0.10.
- AdamW decay 0.05 for Conv/Linear weights; zero decay for biases, LayerNorm, and ConvNeXt layer-scale parameters. Every trainable parameter appears exactly once.
- Successful tuning/final runs complete exactly eight training epochs and eight full validations. Early stopping is OFF; validation performance never shortens or extends a run.
- Supported validation Macro-F1 selects checkpoints using `tau=1e-6`, tie-break `earlier`; all reported metrics come from that checkpoint.
- Warmup then cosine uses the full eight-epoch optimizer-update horizon. Official schedule details remain unresolved; never reuse the three-epoch P1 horizon.
- Exactly four LR candidates with one tuning seed, then the selected LR with three final seeds. All four training seed IDs are distinct; the tuning ID is absent from the final set.
- Pilot seed 3406 and LR 1e-4 are diagnostic references, not official defaults. No P1-v2, E2/E3 implementation, geometry search, or extra ablation.
- No test-image access during implementation, smoke, tuning, training, or validation. A final test evaluator is outside this implementation plan.
- Resume preserves the original eight-epoch budget, selected-checkpoint state, scheduler/RNG state, and identity. Pilot checkpoints cannot be official resume inputs.

## Review Focus

These failure modes need observable tests at the interfaces named below:

1. A malformed path, changed artifact, or test split must fail before image opening/model allocation: T02 and T07.
2. Tail accumulation groups must receive exact sample weighting without losing the final image: T05.
3. Flat/decreasing validation curves or an earlier best checkpoint must still produce eight completed epochs and the original scheduler horizon: T05 and T07.
4. Interruption/restart, including persistent workers and a partial epoch, must preserve order/jitter, state, and budget: T02, T06, T07, and T10.
5. Seed overlap, a failed/short LR trial, cross-bucket false positives, or a report from the wrong checkpoint must not contaminate selection/aggregation: T01, T04, T08, and T09.

## Approval Gates and Unresolved Inputs

| Gate | Required before proceeding | Current state |
|---|---|---|
| G0 - design/plan approval | User reviews the specification, this plan, and the task ledger; separately authorizes implementation and chooses its execution method | Open; writing documents is the only authorized work in this task |
| G1 - numeric contract | Resolve spec R1-R4 and R6-R8; record R9 conventions needed by the implemented reporting/selection behavior | Open; no LR values, numeric seed IDs, batch adoption, schedule, optimizer details, RNG/init binding, or runtime adoption chosen here |
| G2 - implementation acceptance | T01-T10 pass their relevant CPU/target checks; user reviews readiness and concrete run declarations | Not started |
| G3 - official execution | Explicit authorization for the reviewed tuning/final configurations and UIT execution | Not granted by this plan |
| Final test access | Separate final-evaluation specification/authorization after locked checkpoints/protocol | Outside scope |

R5 is already FIXED: eight epochs, full validation every epoch, no early stopping, checkpoint tau=1e-6/earlier. Layer-scale no-decay and seed disjointness are also fixed. Do not reopen those choices or add stopping-related numeric requirements.

**T00 is the first execution prerequisite.** If an unresolved choice affects an interface, numerical oracle, or fixture, record the user's decision and update that part of this draft before implementing it. Do not let a task author choose a research value to make the plan runnable. Synthetic test values are local fixtures, never proposals for the official LR grid or seed set.

## Proposed File Structure

The following files are planned, not created by this documentation task. The package is E1-scoped, without experiment registries or a premature shared E1/E2/E3 framework.

| Planned file | Responsibility | Task |
|---|---|---|
| `src/e1/__init__.py` | Package marker only | T01 |
| `src/e1/config.py` | Complete reviewed-recipe validation, immutable run resolution, fingerprints | T01 |
| `src/e1/data.py` | Exact manifest integrity, train/val isolation, natural epoch sampler, reviewed RNG/loaders | T02 |
| `src/e1/model.py` | E1 backbone/head construction, strict weight loading, fixed optimizer groups | T03 |
| `src/e1/metrics.py` | Full-split 124-output metrics and frequency-bucket accounting | T04 |
| `src/e1/engine.py` | Sample-weighted epoch training, validation, reviewed update schedule, checkpoint comparator | T05 |
| `src/e1/checkpoint.py` | Atomic official checkpoints, identity validation, state/RNG restore | T06 |
| `src/e1/train.py` | Preflight/runtime records, one eight-epoch run, lock/signals, summaries/history, CLI | T07 |
| `src/e1/campaign.py` | Exactly four tuning declarations, complete-trial ledger, locked LR selection, three final declarations | T08 |
| `src/e1/report.py` | Selected-checkpoint validation analysis and three-final-seed aggregation | T09 |
| `tests/e1/conftest.py`, `tests/e1/test_*.py` | E1 behavior tests; fixtures remain synthetic and isolated | T01-T10 |
| `scripts/slurm/e1_train.sbatch`, `docs/E1/runbook.md` | One declared run/job and reviewed deployment instructions | T10 |

Future reviewed numeric files belong under `configs/training/` with a versioned ID/checksum chosen during G1; actual run artifacts belong under a separate E1 run root, never `runs/pilot/`. Campaign-specific resolved declarations and `selection.json` are execution artifacts, not additional research recipes. Do not create any of these files now.

Existing `src/pilot/`, pilot tests/scripts, frozen preprocessing/configs, and split artifacts remain unchanged. Adapt only the needed mechanics into E1-owned code and verify them through E1 tests. Direct imports from `src.pilot` must not become an unreviewed official runtime dependency. No new general-purpose abstractions or third-party dependencies are planned.

## Shared Interface Contracts

Use dictionaries for serialized contracts rather than a generic configuration framework. Each task below owns the listed interface; private helpers are implementation choices.

- **Recipe:** a complete user-reviewed versioned contract with seeds, four LR candidates, batch, runtime, optimizer, initialization/RNG, scheduler, fixed budget/checkpoint rules, reporting conventions, source-evidence references, and per-value rationale. No unresolved required fields or implicit defaults. Explicitly inactive fields, such as scaler settings when scaling is disabled, are distinct from unresolved required values. Preserve the spec's common-protocol constraints for future method comparisons.
- **Run configuration:** resolved recipe/hash plus `stage` (`tuning` or `final`), `run_id`, LR, training seed/role, data/weight/output paths, and, for final runs, a locked LR-selection reference. Operational resume refers to the same configuration, not a newly generated run.
- **Identity:** fixed-artifact/preprocessing/pretrained hashes, official source/config/recipe identity, label mapping, and the reviewed environment/runtime fingerprint. Pilot source hashes are provenance references and are not equalities imposed on official code.
- **Progress:** completed epoch, optimizer updates, full history, and selected checkpoint metadata. There are no validation-driven stopping counters.
- **Sample/batch:** the dataset yields `(global_tensor, label_id, relative_path)`; loaders collate `[N,3,384,896]`, targets, and paths. Training consumes targets; validation/reporting retains paths for traceability.
- **Metrics result:** overall/per-class/bucket metrics and counts, plus confusion data; optional prediction rows are requested only when exporting selected-checkpoint analysis.

## Task T00: Review and Freeze the Inputs

**Dependencies:** none. **Files:** review [specs.md](specs.md), [pilot report](../../reports/pilot/pilot_results.md), and this plan; future recipe/checksum files only after user review.

- [ ] Obtain G0 review and separate implementation authorization; leave all implementation tasks unchecked until that occurs.
- [ ] Record exact R1 LR candidates and LR-trial tie/failure rule; exact R2 four distinct numeric seed IDs; R3 batch; R4 schedule details; R6 betas/eps/clipping; R7 initialization/RNG binding; and R8 environment/runtime/allocation adoption.
- [ ] Record the applicable R9 reporting/std convention and final-evaluation handoff without unlocking test access. R5 requires no new decision.
- [ ] Review/freeze the complete numeric contract and checksum; do not auto-export a partial recipe or assume the pilot collector accepts a changed official contract.
- [ ] Reconcile any resulting interface/test-oracle changes in this plan. Synthetic expected LR endpoints must be written from the reviewed formula, not inferred by calling the production scheduler.
- [ ] Record approval/provenance and commit only the reviewed documents/recipe. A recipe hash identifies content; it is not evidence of user approval by itself.

**Acceptance:** every required official value has an explicit reviewed source; no pilot numeric promotion; G1 is recorded. If it remains incomplete, stop dependent execution and report the unresolved fields rather than choosing them.

## Task T01: Official Configuration and Synthetic Fixtures

**Dependencies:** T00. **Create:** `src/e1/__init__.py`, `src/e1/config.py`, `tests/e1/conftest.py`, `tests/e1/test_config.py`.

**Interface:** `load_recipe(path: Path) -> dict`, `validate_recipe(recipe: dict) -> dict`, and `resolve_run(recipe: dict, *, stage: str, lr: float, seed: int, paths: dict) -> dict`. Recipe loading checks the reviewed version/checksum; run resolution checks seed role, LR membership, fixed invariants, and explicit paths without model/GPU allocation.

- [ ] Write failing tests for missing/invalid required fields, nonfinite/duplicate LR values, seed overlap, non-eight-epoch budgets, enabled early stopping, altered checkpoint tau/tie, changed decay rules, class-weight/balanced-sampling overrides, and missing/invalid full weight hashes. Define `synthetic_recipe` and `synthetic_run` fixtures here for later tests; their values are test-only and cannot be exported into runtime configs.

```python
def test_seed_overlap_is_rejected(synthetic_recipe):
    from src.e1.config import validate_recipe
    synthetic_recipe["seeds"]["final"][0] = synthetic_recipe["seeds"]["tuning"]
    with pytest.raises(ValueError):
        validate_recipe(synthetic_recipe)
```

- [ ] Run `python -m pytest tests/e1/test_config.py -q`; confirm failure is caused by the missing/incorrect E1 contract, not an unrelated import or fixture error.
- [ ] Implement the interfaces and fail-before-allocation validation. Require four distinct training IDs, exactly four positive distinct LRs, the fixed eight-epoch/checkpoint/decay settings, and the full pinned SHA. Do not accept pilot-purpose configurations or a test-split override.
- [ ] Run the same test file; require all tests to pass. Check that changing only operational path resolution does not silently change scientific configuration, and that unresolved fields cannot acquire defaults.
- [ ] Commit the scoped module and tests.

## Task T02: Exact Data and Frozen Preprocessing Integration

**Dependencies:** T01. **Create:** `src/e1/data.py`, `tests/e1/test_data.py`.

**Interface:** `FixedManifest(artifacts_dir: Path)` exposes `fingerprints`, `labels`, `n_train`, and `records(split: str)` for train/val only; `E1Dataset(manifest, data_root, split, seed, augmentation)` yields the sample contract above; `make_loaders(manifest, data_root: Path, config: dict) -> dict` returns `train` and `val` DataLoaders. The train sampler supports `set_epoch(epoch: int)`; loader generators are available for checkpointing.

- [ ] Write tests that verify four artifact digests, 34997/7497/7502 membership counts, 124 label IDs, train/val coverage, path containment, and rejection of test access before any image-open call.
- [ ] Add synthetic image tests for both raw layouts: shape `[3,384,896]`, known padding, unchanged mean-filled mask/padding under jitter, deterministic validation, no local views, retained final short batch, and natural no-replacement epoch coverage.
- [ ] Pin the reviewed R7 order/jitter behavior across epoch changes, worker counts, and persistent-worker restart. Use synthetic images or manifest metadata; do not open actual test images.
- [ ] Run `python -m pytest tests/e1/test_data.py tests/test_preprocessing.py -q` and observe the new E1 tests fail before implementation.
- [ ] Implement the E1 data interface while importing the existing `Preprocessor`/frozen config unchanged. Deliberately adapt manifest/seed/sampler mechanics from the pilot after review; fail on corrupt/missing samples instead of skipping them.
- [ ] Rerun the same checks and commit. The manifest remains the membership authority; a filesystem directory scan cannot replace it.

## Task T03: E1 Model and Fixed Optimizer Groups

**Dependencies:** T01. **Create:** `src/e1/model.py`, `tests/e1/test_model.py`.

**Interface:** `build_e1(weights_path: Path, expected_sha256: str, initialization: dict) -> tuple[nn.Module, dict]`; `E1Model.encode(images: Tensor) -> Tensor` returns `[N,768]`, and `forward(images: Tensor) -> Tensor` returns `[N,124]`; `build_optimizer(model: nn.Module, config: dict) -> tuple[Optimizer, dict]` returns the optimizer and enumerated groups.

- [ ] Write failing tests for exact filename/full-digest enforcement, strict state loading, retained pretrained pooling/final LayerNorm, a new biased 768-to-124 classifier, all parameters trainable, explicit 0.1 depth schedule, and evaluation-mode behavior. Use tiny synthetic backbones/state files for CPU mechanics; they are not pretrained evidence.
- [ ] Add an optimizer test with Conv/Linear weights, biases, LayerNorm, and a trainable ConvNeXt-style `layer_scale`: only Conv/Linear weights decay at 0.05; every other listed category has zero decay; every trainable tensor appears exactly once and all groups share the same LR.
- [ ] Run `python -m pytest tests/e1/test_model.py -q` and confirm the new behavior is not implemented yet.
- [ ] Implement only E1, the pinned offline pretrained loader, reviewed initialization binding, and fixed groups. Do not add architecture switches, head multipliers, separate encoders, or automatic downloads.
- [ ] Run the tests; require pass, record actual group names/init rule/version, and commit. Real pretrained/full-resolution execution belongs to T10 after target authorization.

## Task T04: Full-Split Metrics and Bucket Accounting

**Dependencies:** T01, T02. **Create:** `src/e1/metrics.py`, `tests/e1/test_metrics.py`.

**Interface:** `Metrics124.update(logits: Tensor, targets: Tensor, loss_sum: float) -> None`; `Metrics124.compute(n_train: dict, labels: list[str]) -> dict`. Retain full-split confusion counts and GT-conditioned Top-k counts so bucket metrics do not lose cross-bucket errors.

- [ ] Write tests for 124-output Top-1/3/5, supported-only Macro-F1, GT-support-weighted F1, sample-mean NLL, unsupported F1=null, empty-evaluation failure, and all four frequency buckets plus the 1-2 subset.

```text
Cross-bucket oracle: two GT samples, labels 0 and 1; both predict 0.
Class 0 belongs to the tail, class 1 to the head.
Class 0 F1 must be 2/3, including the false positive from class 1.
Filtering to tail GT rows first would incorrectly produce F1=1.
```

- [ ] Assert Top-k ranking still includes unsupported output labels and bucket coverage derives only from unique n_train. Use distinct synthetic scores for ranking tests; record any reviewed ranking/report convention from R9.
- [ ] Run `python -m pytest tests/e1/test_metrics.py -q` and inspect the failure.
- [ ] Implement the metric interface and schema; do not compute Macro-F1 separately per minibatch or recompute bucket F1 from filtered confusion matrices.
- [ ] Rerun tests, compare fixed metadata bucket counts against the spec, and commit.

## Task T05: Epoch Mechanics, Update Scheduler, and Checkpoint Comparator

**Dependencies:** T01, T03, T04. **Create:** `src/e1/engine.py`, `tests/e1/test_engine.py`.

**Interfaces:** `UpdateScheduler(optimizer, schedule: dict, total_updates: int)` exposes `step()`, `state_dict()`, and `load_state_dict(state)`; `train_epoch(model, loader, optimizer, scheduler, scaler, *, device, runtime: dict, accumulation: int, clipping: dict) -> dict`; `evaluate(model, loader, *, device, runtime: dict, n_train: dict, labels: list[str], collect_predictions: bool = False) -> dict`; `select_checkpoint(current: dict, selected: dict | None) -> bool`.

- [ ] Write an accumulation test comparing gradients/updates with a direct sample-mean calculation for unequal microbatches and a short final group. Assert every sample is counted once and scheduler steps match optimizer updates, not microbatches or validations.
- [ ] Add exact checkpoint tests: within 1e-6 retain the earlier epoch even when NLL improves; an increase above 1e-6 replaces it; lower F1 does not replace it.

```python
def test_checkpoint_tie_does_not_use_nll():
    from src.e1.engine import select_checkpoint
    selected = {"epoch": 1, "macro_f1_supported": 0.5, "nll": 1.0}
    current = {"epoch": 2, "macro_f1_supported": 0.5, "nll": 0.1}
    assert select_checkpoint(current, selected) is False
```

- [ ] After R4 review, write independent expected LR endpoint/update fixtures for the adopted formula. Assert an eight-epoch horizon, unchanged by an earlier selected epoch, and rejection of changed scheduler configuration on restore. These are schedule tests, not an LR search.
- [ ] Run `python -m pytest tests/e1/test_engine.py -q`, implement the interfaces, then rerun to pass. Preserve sample weighting, unscaled-gradient checks, explicit numerical failure, and reviewed clipping/runtime behavior; no skipped-update budget extension.
- [ ] Commit the engine and tests. Evaluation uses model.eval() without augmentation, never advances the optimizer/scheduler, and can return validation prediction rows when explicitly requested.

## Task T06: Official Checkpoints and Exact Resume

**Dependencies:** T01, T02, T03, T05. **Create:** `src/e1/checkpoint.py`, `tests/e1/test_checkpoint.py`.

**Interfaces:** `save_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, config: dict, progress: dict, identity: dict, loaders: dict) -> None`; `read_checkpoint(path: Path, *, config: dict, identity: dict) -> dict`; `restore_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, config: dict, identity: dict, loaders: dict) -> dict` returns restored progress. Use a distinct official checkpoint kind and atomic replacement; internal RNG/state helpers remain private.

- [ ] Write a tiny CPU/dropout uninterrupted-versus-resumed test comparing final model/optimizer/scheduler/RNG, selected epoch, and history metrics. Check boundary-zero recovery and replay of an unfinished epoch from the previous committed boundary.
- [ ] Reject pilot checkpoint kinds, changed seed/LR/batch/runtime/identity, `max_epochs != 8`, altered schedule horizons, invalid completed-update counts, and attempts to resume from best.pt as if it were last.pt. `read_checkpoint` may validate/read best.pt for T09 reporting; only the restore/resume operation rejects that role mismatch.
- [ ] Test atomic-save failure and load rejection for incomplete checkpoint contents; a failed replacement must leave the previous last checkpoint readable.
- [ ] Run `python -m pytest tests/e1/test_checkpoint.py -q`; implement the three interfaces, then rerun to pass. Restore all Python/NumPy/CPU/CUDA and loader-generator state required by the adopted runtime.
- [ ] Commit. Saved progress contains no validation-driven stopping state, and resume never promotes a pilot checkpoint or changes the original budget.

## Task T07: One Official Eight-Epoch Runner

**Dependencies:** T01-T06. **Create:** `src/e1/train.py`, `tests/e1/test_train.py`.

**Interfaces:** `preflight(config: dict) -> dict` returns verified official identity without model allocation; `run(config: dict, run_dir: Path, *, resume: Path | None = None, stop_requested: Callable[[], bool]) -> dict`; module `main()` resolves one declared tuning/final run or performs a no-allocation configuration check. Reuse the interfaces above; do not add a generic experiment runner.

- [ ] Write an eight-epoch synthetic CPU integration test with decreasing/flat validation F1. Require eight complete training calls, eight full validations, full planned update counts, an earlier selected checkpoint, and no performance-based stop.
- [ ] Test preflight failure before model/image allocation for missing review/hash/conformance, wrong paths/packages, invalid seed roles, and final runs lacking a locked selection reference. Keep official source identity separate from pilot source evidence.
- [ ] Test interruption after a completed epoch, mid-epoch failure/replay, epoch-zero recovery, resume to exactly epoch eight, single-writer locking, and immutable existing run IDs. Compare scientific state rather than wall-clock timings.
- [ ] Run `python -m pytest tests/e1/test_train.py -q`; implement one-run orchestration, explicit runtime application/records, full validation/history, atomic best/last checkpoints, and complete/failure/interrupted summaries. Signals request operational boundary interruption only, never validation-driven termination.
- [ ] Rerun the test file and the E1 suite; require pass and commit. Tests may replace external CUDA/I/O edges with small CPU fixtures, but no production bypass flag or synthetic-evidence acceptance path is permitted.

## Task T08: Four-Trial Ledger and Disjoint Final Declarations

**Dependencies:** T01, T07. **Create:** `src/e1/campaign.py`, `tests/e1/test_campaign.py`.

**Interfaces:** `plan_tuning(recipe: dict, paths: dict) -> list[dict]`; `select_lr(recipe: dict, trial_records: list[dict]) -> dict`; `plan_final(recipe: dict, selection: dict, paths: dict) -> list[dict]`. Declare exactly four tuning runs and then three fresh final runs; declaration/selection does not submit jobs. Persist a versioned ledger and locked `selection.json` with recipe/trial-summary/checkpoint references and hashes.

- [ ] Test that four reviewed LR values produce exactly four declarations with the same tuning seed and eight-epoch budget. Test that three final declarations use three distinct seeds excluding the tuning seed and only the locked selected LR.
- [ ] Reject duplicate/extra/missing candidates, incomplete or failed trial sets presented as complete, wrong seed/recipe/checkpoint identity, records marked as synthetic/test fixtures, metric values from a different epoch, and a modified selection file.
- [ ] Pin the reviewed R1 LR-trial tie rule independently of the earlier-epoch checkpoint tie-break. Do not infer that lower LR, earlier trial, or lower NLL wins an LR tie unless explicitly reviewed.
- [ ] Run `python -m pytest tests/e1/test_campaign.py -q`; implement the interfaces and explicit failure/retry ledger without automatic new candidates, batch changes, or job submission; rerun to pass.
- [ ] Commit. Final training initializes from ImageNet; the selected tuning checkpoint is provenance for LR selection, not a warm-start source.

## Task T09: Selected-Checkpoint Validation Reports and Final-Seed Aggregation

**Dependencies:** T02-T08. **Create:** `src/e1/report.py`, `tests/e1/test_report.py`.

**Interfaces:** `evaluate_selected(config: dict, run_dir: Path) -> dict`; `aggregate_final(run_dirs: list[Path], recipe: dict, selection: dict) -> dict`. Require complete eight-epoch run records and load the selected official checkpoint; evaluate validation only with the same data/preprocessing/identity contract. Use T04/T05 rather than a second metric implementation.

- [ ] Test selection of an earlier best.pt while the full run completed eight epochs; every exported metric/prediction must match that checkpoint, not last.pt or the best epoch for another metric.
- [ ] Test overall/per-class/bucket CSV/JSON, confusion/most-confused pairs, path/label-ID Top-k predictions, unsupported classes, and full learning-curve/history preservation.
- [ ] Reject aggregation unless there are exactly three complete distinct final seeds, disjoint from tuning, sharing the locked recipe/selection. Test the reviewed std convention with a hand-calculated synthetic oracle; do not replace scores with an ensemble.
- [ ] Run `python -m pytest tests/e1/test_report.py -q`; implement the interfaces, rerun to pass, and commit. Do not implement or expose a test-split evaluator in this task.

## Task T10: UIT Readiness and Authorized Smoke

**Dependencies:** T01-T09; target checks additionally require explicit UIT authorization. **Create after review:** `scripts/slurm/e1_train.sbatch`, `docs/E1/runbook.md`, `tests/e1/test_entrypoints.py`, `tests/e1/test_gpu_smoke.py`.

- [ ] Write CPU CLI tests proving configuration-only checks allocate no model/GPU, missing fields fail clearly, one invocation names one run, and test access/search overrides are rejected. Mark GPU tests opt-in under the existing `gpu` marker; require real reviewed paths/hash/configuration.
- [ ] Write the E1-specific single-run Slurm wrapper and runbook using reviewed allocation/runtime/environment variables. Preserve the eight-epoch configuration and boundary-resume contract; do not turn the wrapper into an architecture/LR/batch search loop or reuse pilot run IDs.
- [ ] Run `python -m pytest tests/e1 -q -m "not gpu"`, the existing CPU regression suite, `python -m compileall -q src/e1 tests/e1`, and a shell syntax check of the new wrapper. Require all applicable checks to pass; no dependency upgrade or pilot/preprocessing modification is part of acceptance.
- [ ] After target authorization, verify actual environment/weight/dataset/conformance identity and run a bounded synthetic/train-only smoke for real model loading, full-size forward/backward, finite gradients, memory, and interrupted/resumed behavior. Record the observed results as smoke/acceptance evidence, not an official trial or a P1-v2 study.
- [ ] Review readiness and concrete run declarations with the user; commit the implementation/runbook and acceptance record through the approved publishing workflow. G2 remains open until target checks are actually evidenced, not merely listed here.

## Task T11: Separately Authorized Official Execution and Handoff

**Dependencies:** T00-T10, G2 PASS, and G3 authorization. **Files:** execution artifacts under the reviewed E1 run root, campaign ledger/selection, and final E1 validation report; no new architecture or code branch is implied.

- [ ] Review the four resolved tuning declarations, shared numeric contract, walltime/resume plan, and seed roles before submitting any job. Execute only the four declared LRs with the single tuning seed; retain failures/retries and require full eight-epoch completion before equal-budget selection.
- [ ] Lock the selected LR using R1 and the selected-checkpoint supported validation Macro-F1, preserving the full four-trial ledger. Freeze the selected configuration without copying tuning model/optimizer state into final runs.
- [ ] Review and execute the three final declarations under their distinct, disjoint final seeds. Require eight training epochs/eight validations per successful run and unchanged full scheduler horizons on resume.
- [ ] Generate selected-checkpoint validation analysis and mean/std reports using T09. Preserve per-seed checkpoints and lineage for possible later E3 work; do not create feature caches.
- [ ] Review results/limitations and archive the artifact inventory. Test remains locked: final test evaluation requires its own later authorization, not automatic continuation of this task.

## Validation and Handoff Rules

During future implementation, use the smallest meaningful red/green check for each task, then run the relevant regression suite before committing. A missing-file/module failure establishes only that work is unimplemented; verify the intended assertion once the interface exists. A passing CPU fixture is not UIT evidence. Never download weights or open dataset test images to make a unit test pass.

Do not mark an implementation task complete because this plan names its files or tests. Each task needs its observable acceptance evidence and scoped diff. `tasks.md` tracks the same IDs and gate state; update it only when the corresponding work is genuinely complete.

**Self-review performed for this draft:** spec Sections 3-5 map to T01-T03; Sections 6-8 to T01/T03/T05-T08; Sections 9-10 to T04/T06-T10; pilot-evidence limitations and open decisions to T00/T10; acceptance to the tests above and G2. Eight-epoch execution, fixed checkpoint ties, layer-scale no-decay, seed disjointness, test isolation, and full-budget resume have explicit checks. Numeric schedule/RNG/reporting oracles are conditional on the user's unresolved decisions, never agent-invented defaults.

The next step is user review of [specs.md](specs.md), this plan, and [tasks.md](tasks.md), followed by selection of an execution method and separate authorization if implementation is desired. No task below the approval gates runs automatically.
