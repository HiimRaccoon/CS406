# E1 Task Ledger

**DRAFT FOR REVIEW - planning only; implementation and official execution have not started.**

Date: **2026-10-31**. Authority: [specs.md](specs.md), with execution interfaces and red/green verification defined in [plan.md](plan.md). This ledger contains the same task IDs and dependencies as the plan; it does not create another recipe or resolve any NEEDS REVIEW value.

Only `docs/E1/plan.md` and `docs/E1/tasks.md` are deliverables of the current request. All checkboxes below intentionally remain unchecked. Planned modules, tests, numeric contracts, scripts, runbooks, and run artifacts do not exist merely because they appear in this ledger.

## 1. Fixed Protocol Guardrails

- E1 is ConvNeXt-Tiny with one 896 x 384 global view, the exact fixed split/124-label mapping, frozen preprocessing v1, and the pinned ImageNet checkpoint/full hash.
- Natural unweighted CE, full fine-tuning from epoch 1, head LR multiplier 1, maximum stochastic depth 0.1, brightness/contrast 0.10/0.10.
- Conv/Linear weights decay at 0.05; biases, LayerNorm, and ConvNeXt layer-scale parameters have zero decay.
- Successful tuning/final runs complete eight training epochs and eight full validations; early stopping OFF. Checkpoint selection uses supported validation Macro-F1, tau=1e-6, and earlier ties; it cannot shorten the scheduler/run budget.
- Exactly four LR candidates with one tuning seed, then selected LR with three distinct final seeds; all four training IDs are distinct. Pilot 3406 is not automatically reused; split seed 406 never changes membership.
- No test-image access, P1-v2, E1-HR/E2/E3 implementation, geometry/augmentation search, extra candidates, or automatic pilot-to-official promotion.
- Do not create a generic permanent training framework or rewrite frozen/pilot files as an incidental part of E1.

## 2. Approval Gates

- [ ] **G0:** User reviews the E1 spec/plan/task ledger, authorizes implementation separately, and selects its execution method.
- [ ] **G1:** Applicable unresolved official values are reviewed and captured in a complete versioned numeric contract with source evidence/checksum; R5 and the fixed protocol guardrails are not reopened.
- [ ] **G2:** Required CPU/target implementation acceptance is evidenced and the user reviews readiness plus concrete run declarations.
- [ ] **G3:** User authorizes the reviewed official UIT tuning/final execution.
- [ ] **Later final-evaluation gate:** Separate authorization after locked protocol/checkpoints; no task here unlocks test access.

## 3. NEEDS REVIEW Inputs

| Spec register | Required resolution | Owner of planned consumption |
|---|---|---|
| R1 | Exact four LR values; LR-trial tie/failure accounting | T00, T01, T08 |
| R2 | Exact numeric tuning/final IDs; their four-way disjointness is already fixed | T00, T01, T08 |
| R3 | Common effective batch and E1 microbatch/accumulation | T00, T01, T02, T05 |
| R4 | Warmup duration/start factor and cosine formula/floor details; full eight-epoch horizon is fixed | T00, T01, T05 |
| R5 | No new choice: eight epochs, no early stopping, tau=1e-6/earlier | Compliance tests T01, T05-T08 |
| R6 | AdamW betas/eps and gradient clipping; layer-scale no-decay is already fixed | T00, T01, T03, T05 |
| R7 | Official initialization and RNG/worker/resume binding | T00, T02, T03, T06 |
| R8 | Runtime/environment, allocation, and output adoption | T00, T01, T07, T10 |
| R9 | Reporting/std convention and final-evaluation handoff | T00, T09; test access remains a separate gate |

No numeric IDs/grid or other adoption values are chosen by this ledger. Synthetic test fixtures are not official configurations. If a decision changes a planned interface or numerical oracle, revise that portion of the plan before dependent implementation.

## 4. Dependency-Ordered Tasks

| Task | Depends on | Planned deliverable | Completion evidence |
|---|---|---|---|
| T00 | None | Reviewed inputs/approval record; later complete numeric contract | User decisions and provenance; no required official defaults or unresolved fields |
| T01 | T00 | E1 configuration interface and synthetic fixtures | Configuration tests reject invalid invariants, missing values, seed overlap, and pilot configs |
| T02 | T01 | Exact manifest, train/val data, frozen preprocessing/RNG/loaders | Data tests prove isolation, full sample coverage, both layouts, and correct epoch/jitter behavior |
| T03 | T01 | E1 model, strict weights, fixed optimizer groups | Model tests verify backbone/head/init/depth behavior and every decay/no-decay category |
| T04 | T01, T02 | 124-output full-split and bucket metrics | Metric tests retain cross-bucket false positives and supported-only averaging |
| T05 | T01, T03, T04 | Epoch mechanics, reviewed scheduler, fixed comparator | Engine tests prove sample-weighted tails, update accounting, full horizon, and tau/earlier ties |
| T06 | T01, T02, T03, T05 | Official atomic checkpoint/resume interface | CPU state equivalence and identity/budget/pilot-checkpoint rejection tests |
| T07 | T01-T06 | One-run official runner with preflight/history/runtime/failure handling | Eight-epoch synthetic integration and interruption/lock/zero-allocation checks |
| T08 | T01, T07 | Four-trial declarations/ledger, locked LR selection, three final declarations | Campaign tests reject overlap, short/failed trial substitution, wrong identity, and extra candidates |
| T09 | T02-T08 | Selected-checkpoint validation analysis and three-seed aggregation | Report tests enforce checkpoint lineage, distinct final seeds, and reviewed std convention |
| T10 | T01-T09; authorization for target work | E1-specific deployment instructions, one-run Slurm wrapper, acceptance evidence | CPU regression/syntax checks plus authorized real-weight/full-size/resume smoke |
| T11 | T00-T10, G2 PASS, G3 | Official tuning/final execution artifacts and validation report | Four complete tuning trials, locked LR, three complete final runs, reproducible per-seed reports |

Dependencies define execution order; they do not request parallel agents or authorize any task automatically. Native task-by-task execution and delegated execution both require the user's later method selection.

### T00 - Reviewed Inputs

- [ ] Record G0 and resolve the applicable R1-R4/R6-R9 fields with the user; do not choose official values on the user's behalf.
- [ ] Review/freeze the complete numeric contract/checksum and reconcile any resulting plan/test-oracle changes.
- [ ] Record G1 and commit the approved input artifacts through the reviewed workflow.

### T01 - Configuration

- [ ] Write failing `tests/e1/test_config.py` tests and define isolated synthetic fixtures in `tests/e1/conftest.py`.
- [ ] Implement `src/e1/config.py` and the package marker; run the T01 verification command to pass.
- [ ] Verify all fixed rules and unresolved-field failures, inspect the scoped diff, and commit.

### T02 - Data

- [ ] Write failing `tests/e1/test_data.py` tests for artifact/path/test-access isolation, both input layouts, natural tails, and reviewed epoch/RNG behavior.
- [ ] Implement `src/e1/data.py`, importing frozen preprocessing unchanged; pass data and existing preprocessing tests.
- [ ] Confirm that no split/preprocessing bytes changed and commit.

### T03 - Model and Optimizer

- [ ] Write failing `tests/e1/test_model.py` tests, including mandatory full weight digest and no-decay layer-scale parameters.
- [ ] Implement `src/e1/model.py` for E1 only; pass the T03 tests with synthetic CPU mechanics.
- [ ] Record group/init/depth metadata expectations and commit; real target execution stays in T10.

### T04 - Metrics

- [ ] Write failing `tests/e1/test_metrics.py` tests, including the hand-calculated cross-bucket F1 oracle from the plan.
- [ ] Implement `src/e1/metrics.py`; pass full-split/support/bucket/Top-k checks.
- [ ] Verify metadata bucket coverage and commit.

### T05 - Epoch Engine

- [ ] Write failing `tests/e1/test_engine.py` tests for short-tail weighting, optimizer updates, the reviewed schedule oracle, and fixed checkpoint ties.
- [ ] Implement `src/e1/engine.py`; require tau=1e-6/earlier and a full eight-epoch schedule regardless of selected epoch.
- [ ] Pass tests, verify nonfinite behavior, and commit without adding validation-based stopping.

### T06 - Checkpoints

- [ ] Write failing `tests/e1/test_checkpoint.py` tests for full-state equivalence, atomic-save recovery, and wrong-identity/budget/pilot-kind rejection.
- [ ] Implement `src/e1/checkpoint.py`; restore the original budget, RNG/loaders, optimizer/scheduler, history, and selected-checkpoint state.
- [ ] Pass boundary-zero/partial-epoch replay checks and commit.

### T07 - Single-Run Execution

- [ ] Write failing `tests/e1/test_train.py` integration tests proving eight epochs/eight validations even with flat or decreasing validation F1.
- [ ] Implement `src/e1/train.py` with explicit preflight/runtime records, single-writer locking, operational interruption, and atomic checkpoint/history handling.
- [ ] Pass the E1 CPU suite, verify no production bypass flags or test-image path, and commit.

### T08 - Tuning and Final Declarations

- [ ] Write failing `tests/e1/test_campaign.py` tests for exactly four tuning declarations, four-way seed disjointness, valid full-budget evidence, locked selection, and three fresh final declarations.
- [ ] Implement `src/e1/campaign.py` with the reviewed LR-trial tie rule; declaration/selection must not submit jobs.
- [ ] Pass all campaign guards and commit; final runs must not warm-start from tuning checkpoints.

### T09 - Reports

- [ ] Write failing `tests/e1/test_report.py` tests for selected-checkpoint lineage, validation exports, and the reviewed three-seed std convention.
- [ ] Implement `src/e1/report.py` using the shared E1 metrics/evaluation interface; reject incomplete, overlapping, or inconsistent final-seed records.
- [ ] Pass report checks and commit; no test-split evaluator or E3 caching is added.

### T10 - Deployment and Acceptance

- [ ] Write/pass E1 entrypoint CPU tests and opt-in target smoke tests as specified in the plan.
- [ ] Prepare the reviewed one-run Slurm wrapper and E1 runbook; run CPU regression, compilation, and shell syntax checks.
- [ ] Only after target authorization, record actual environment/integrity/full-size forward-backward/resume smoke evidence; do not label it a trial or P1-v2.
- [ ] Review G2 readiness and concrete run declarations with the user; commit the scoped deliverables.

### T11 - Official Runs, Only After G3

- [ ] Execute the four reviewed LR trials with the single tuning seed; successful trials each complete eight epochs and eight validations.
- [ ] Lock LR selection from the selected-checkpoint validation F1 and the reviewed trial tie rule, retaining the complete ledger/failure history.
- [ ] Execute the selected configuration with the three distinct final seeds, all disjoint from tuning; each successful run completes eight epochs.
- [ ] Generate selected-checkpoint validation analysis and per-seed/mean/std reports; archive identities/checkpoints and review limitations.
- [ ] Hand off for separately authorized final evaluation. Keep test images locked and do not start E1-HR/E2/E3 automatically.

## 5. Completion and Evidence Rules

Follow each task's red/green/check/commit steps in [plan.md](plan.md). A task is complete only when its tests and artifact evidence exist, not when its description is written. Keep failed checks, operational interruptions, and retries traceable rather than marking them as completed runs.

Test fixtures may exercise eight epochs with tiny CPU models; they are not UIT measurements or official experiment records. Runtime policy, source identity, and artifact validation must prevent synthetic evidence from becoming official campaign evidence.

Only the two planning documents are complete as deliverables of the current request. G0-G3, T00-T11, official code, jobs, and evaluation remain unexecuted. The next action is user review, not automatic implementation.
