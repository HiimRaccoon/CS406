# E1 Task Ledger

**G2 PASS / G3a PASS - T11a COMPLETE. Selected LR: 1e-4. STOP for explicit G3b review; final seeds NOT STARTED; test access LOCKED.**

Updated: **2026-11-02**. Authority: [specs.md](specs.md), the [T00 user decision record](t00_review.md), and the complete [E1 numeric contract v1](e1_numeric_contract_v1.json) with its [checksum](e1_numeric_contract_v1.sha256), explicitly frozen at G1. Technical interfaces remain in [plan.md](plan.md); this ledger owns current gate/completion state. G0 and R1-R4/R6-R9 input decisions are recorded; G1 is PASS; see [g1_approval.json](g1_approval.json), [implementation evidence](implementation_log.md), and [independent review corrections](implementation_review.md).

Current official-stage status is recorded in [t11a_handoff.md](t11a_handoff.md): the user confirms G2 PASS, G3a PASS and T11a complete; the supplied tuning CSV contains four eight-epoch trials with seed 1001. Selected LR is `1e-4`, selected epoch 7, supported validation Macro-F1 `0.45248836792097563`, following the frozen rule. G3b is OPEN / NOT APPROVED; final seeds 2001/3001/4001 have NOT started; test access remains LOCKED. Earlier T10 preparation evidence remains historical. The frozen contract, scientific code, split/preprocessing bytes, pilot files, reviewed numeric decisions, task order and gate dependencies remain unchanged.

## 1. Fixed Protocol Guardrails

- E1 is ConvNeXt-Tiny with one 896 x 384 global view, the exact fixed split/124-label mapping, frozen preprocessing v1, and the pinned ImageNet checkpoint/full hash.
- Natural unweighted CE, full fine-tuning from epoch 1, head LR multiplier 1, maximum stochastic depth 0.1, brightness/contrast 0.10/0.10.
- Conv/Linear weights decay at 0.05; biases, LayerNorm, and ConvNeXt layer-scale parameters have zero decay.
- Successful tuning/final runs complete eight training epochs and eight full validations; early stopping OFF. Checkpoint selection uses supported validation Macro-F1, tau=1e-6, and earlier ties; it cannot shorten the scheduler/run budget.
- Exactly four LR candidates with one tuning seed, then selected LR with three distinct final seeds; all four training IDs are distinct. Pilot 3406 is not automatically reused; split seed 406 never changes membership.
- No test-image access, P1-v2, E1-HR/E2/E3 implementation, geometry/augmentation search, extra candidates, or automatic pilot-to-official promotion.
- Do not create a generic permanent training framework or rewrite frozen/pilot files as an incidental part of E1.

## 2. Approval Gates

- [x] **G0 - design and planning approval:** User explicitly approved the E1 spec, implementation plan, and task ledger for T00 only on 2026-10-31; see [verbatim decision record](t00_review.md). This does not authorize implementation.
- [x] **G1 - numeric-contract freeze and implementation authorization:** After T00 prepares the complete versioned official numeric contract, source evidence/checksum, and reconciled dependent plan/test-oracle changes, the user reviews/freezes that contract and explicitly authorizes implementation to begin. No production E1 implementation task T01+ may start before G1 PASS. R5 and the fixed protocol guardrails are not reopened.
- [x] **G2 - implementation acceptance: PASS.** Recorded on 2026-11-01 from the user's explicit completed-stage confirmation; see [t11a_handoff.md](t11a_handoff.md). No new target measurements are inferred from this documentation update. G2 alone does not authorize official training.
- [x] **G3a - official LR tuning authorization: PASS.** Recorded from the user's confirmation for the completed four-trial T11a stage. This covers tuning only; final training remains unauthorized.
- [ ] **G3b - final-training authorization:** After T11a stops and hands off its completed tuning ledger and selected-LR evidence, the user explicitly reviews them and authorizes the selected configuration for the three official final seeds through T11b. G3a or automatic LR selection cannot satisfy G3b.
- [ ] **Later final-evaluation gate:** Separate authorization after locked protocol/checkpoints; no task here unlocks test access.

Gate flow: **G0 -> T00 -> G1 -> T01-T09 -> T10 -> G2 -> G3a -> T11a -> STOP for user review -> G3b -> T11b -> separate final-evaluation gate**. T10 target GPU smoke requires its own explicit target-work authorization; implementation approval alone does not authorize GPU jobs. All official GPU work uses the reviewed UIT target environment and Slurm execution path.

## 3. Reviewed T00 Inputs - Frozen at G1

| Spec register | Required resolution | Owner of planned consumption |
|---|---|---|
| R1 | Exact four LR values; LR-trial tie/failure accounting | T00, T01, T08, T11a |
| R2 | Exact numeric tuning/final IDs; their four-way disjointness is already fixed | T00, T01, T08, T11a, T11b |
| R3 | Common effective batch and E1 microbatch/accumulation | T00, T01, T02, T05 |
| R4 | Warmup duration/start factor and cosine formula/floor details; full eight-epoch horizon is fixed | T00, T01, T05 |
| R5 | No new choice: eight epochs, no early stopping, tau=1e-6/earlier | Compliance tests T01, T05-T08 |
| R6 | AdamW betas/eps and gradient clipping; layer-scale no-decay is already fixed | T00, T01, T03, T05 |
| R7 | Official initialization and RNG/worker/resume binding | T00, T02, T03, T06 |
| R8 | Runtime/environment, allocation, and output adoption | T00, T01, T07, T10, T11a, T11b |
| R9 | Reporting/std convention and final-evaluation handoff | T00, T09, T11b; test access remains a separate gate |

The frozen contract records the user's reviewed values for all rows above: grid `[2.5e-5,5e-5,1e-4,2e-4]`; tuning `1001`; finals `[2001,3001,4001]`; effective/microbatch/accumulation `2/2/1`; 139992-update eight-epoch schedule with 1000-update warmup/start 0.1/floor 1e-6; AdamW `[0.9,0.999]`/eps 1e-8, clipping OFF; deterministic initialization/RNG/FP32 target lock; account `uit`/QoS `mig2-35`, 4 CPUs/16 GiB/09:00:00; sample std ddof=1 and ascending-label-ID exact-logit ties. G1 approval binds exact JSON SHA-256 `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`. See the contract for complete definitions, provenance, and runtime paths. Synthetic plan oracles remain test-only values.

## 4. Dependency-Ordered Tasks

| Task | Depends on | Planned deliverable | Completion evidence |
|---|---|---|---|
| T00 | G0 PASS | Complete versioned numeric contract/checksum and reconciled dependent plan/test-oracle changes, submitted for G1 | User decisions and provenance; no required official defaults or unresolved fields; user freeze and implementation authorization recorded at G1 |
| T01 | T00, G1 PASS | E1 configuration interface and synthetic fixtures | Configuration tests reject invalid invariants, missing values, seed overlap, and pilot configs |
| T02 | T01, G1 PASS | Exact manifest, train/val data, frozen preprocessing/RNG/loaders | Data tests prove isolation, full sample coverage, both layouts, and correct epoch/jitter behavior |
| T03 | T01, G1 PASS | E1 model, strict weights, fixed optimizer groups | Model tests verify backbone/head/init/depth behavior and every decay/no-decay category |
| T04 | T01, T02, G1 PASS | 124-output full-split and bucket metrics | Metric tests retain cross-bucket false positives and supported-only averaging |
| T05 | T01, T03, T04, G1 PASS | Epoch mechanics, reviewed scheduler, fixed comparator | Engine tests prove sample-weighted tails, update accounting, full horizon, and tau/earlier ties |
| T06 | T01, T02, T03, T05, G1 PASS | Official atomic checkpoint/resume interface | CPU state equivalence and identity/budget/pilot-checkpoint rejection tests |
| T07 | T01-T06, G1 PASS | One-run official runner with preflight/history/runtime/failure handling | Eight-epoch synthetic integration and interruption/lock/zero-allocation checks |
| T08 | T01, T07, G1 PASS | Four-trial declarations/ledger, locked LR selection, three final declarations; preparation does not authorize execution | Campaign tests reject overlap, short/failed trial substitution, wrong identity, and extra candidates |
| T09 | T02-T08, G1 PASS | Selected-checkpoint validation analysis and three-seed aggregation | Report tests enforce checkpoint lineage, distinct final seeds, and reviewed std convention |
| T10 | T01-T09, G1 PASS; explicit authorization before target GPU work | E1-specific UIT deployment instructions, reviewed one-run Slurm wrapper/runbook, acceptance evidence | CPU regression/syntax checks plus authorized real-weight/full-size forward-backward/resume smoke on the UIT target |
| T11a | T00-T10, G1 PASS, G2 PASS, G3a PASS | Official UIT GPU/Slurm tuning ledger and selected-LR evidence; stop for user review | Four complete eight-epoch/eight-validation tuning trials, traceable failures/retries, frozen-rule LR selection, no final seeds launched |
| T11b | T11a complete with tuning handoff, G1 PASS, G2 PASS, G3b PASS | Selected-configuration official UIT GPU/Slurm final-training artifacts and validation report | Three complete eight-epoch/eight-validation final runs with distinct seeds disjoint from tuning; reproducible per-seed/mean/std reports; test remains locked |

Dependencies define execution order; they do not request parallel agents or authorize any task automatically. Implementation method selection accompanies the later G1 authorization. T01-T09 build and validate the E1 implementation locally/CPU as applicable; T10 owns authorized UIT deployment/target acceptance. T11a and T11b require their separate stage authorizations even when all implementation tasks are complete.

### T00 - Reviewed Inputs

- [x] Record G0 approval for T00 only and the user's complete R1-R4/R6-R9 decisions in [t00_review.md](t00_review.md). Preserve R5; retain lowercase `src/e1/` and `tests/e1/` as expressly instructed. No implementation authorization is inferred.
- [x] Prepare/inspect the complete versioned numeric contract/checksum with source evidence and reconcile dependent plan/test oracles. Submit the complete artifact for separate G1 review; all required official numeric inputs are recorded. Validation is limited to documentation/contract checks, not implementation tests or target evidence.
- [x] Record the user's G1 review/freeze of the complete contract, explicit implementation authorization, and selected execution method; commit the approved input artifacts through the reviewed workflow. Keep T01+ blocked until G1 PASS.

### T01 - Configuration

- [x] Write failing `tests/e1/test_config.py` tests and define isolated synthetic fixtures in `tests/e1/conftest.py`.
- [x] Implement `src/e1/config.py` and the package marker; run the T01 verification command to pass.
- [x] Verify all fixed rules and unresolved-field failures, inspect the scoped diff, and commit.

### T02 - Data

- [x] Write failing `tests/e1/test_data.py` tests for artifact/path/test-access isolation, both input layouts, natural tails, and reviewed epoch/RNG behavior.
- [x] Implement `src/e1/data.py`, importing frozen preprocessing unchanged; pass data and existing preprocessing tests.
- [x] Confirm that no split/preprocessing bytes changed and commit.

### T03 - Model and Optimizer

- [x] Write failing `tests/e1/test_model.py` tests, including mandatory full weight digest and no-decay layer-scale parameters.
- [x] Implement `src/e1/model.py` for E1 only; pass the T03 tests with synthetic CPU mechanics.
- [x] Record group/init/depth metadata expectations and commit; real target execution stays in T10.

### T04 - Metrics

- [x] Write failing `tests/e1/test_metrics.py` tests, including the hand-calculated cross-bucket F1 oracle from the plan.
- [x] Implement `src/e1/metrics.py`; pass full-split/support/bucket/Top-k checks.
- [x] Verify metadata bucket coverage and commit.

### T05 - Epoch Engine

- [x] Write failing `tests/e1/test_engine.py` tests for short-tail weighting, optimizer updates, the reviewed schedule oracle, and fixed checkpoint ties.
- [x] Implement `src/e1/engine.py`; require tau=1e-6/earlier and a full eight-epoch schedule regardless of selected epoch.
- [x] Pass tests, verify nonfinite behavior, and commit without adding validation-based stopping.

### T06 - Checkpoints

- [x] Write failing `tests/e1/test_checkpoint.py` tests for full-state equivalence, atomic-save recovery, and wrong-identity/budget/pilot-kind rejection.
- [x] Implement `src/e1/checkpoint.py`; restore the original budget, RNG/loaders, optimizer/scheduler, history, and selected-checkpoint state.
- [x] Pass boundary-zero/partial-epoch replay checks and commit.

### T07 - Single-Run Execution

- [x] Write failing `tests/e1/test_train.py` integration tests proving eight epochs/eight validations even with flat or decreasing validation F1.
- [x] Implement `src/e1/train.py` with explicit preflight/runtime records, single-writer locking, operational interruption, and atomic checkpoint/history handling.
- [x] Pass the E1 CPU suite, verify no production bypass flags or test-image path, and commit.

### T08 - Tuning and Final Declarations

- [x] Write failing `tests/e1/test_campaign.py` tests for exactly four tuning declarations, four-way seed disjointness, valid full-budget evidence, locked selection, and three fresh final declarations.
- [x] Implement `src/e1/campaign.py` with the reviewed LR-trial tie rule; declaration/selection must not submit jobs. Preparing or validating declarations is not execution authorization: T11a stops after selection, and final seeds require G3b.
- [x] Pass all campaign guards, including rejection of a tuning-to-final automatic launch, and commit; final runs must not warm-start from tuning checkpoints.

### T09 - Reports

- [x] Write failing `tests/e1/test_report.py` tests for selected-checkpoint lineage, validation exports, and the reviewed three-seed std convention.
- [x] Implement `src/e1/report.py` using the shared E1 metrics/evaluation interface; reject incomplete, overlapping, or inconsistent final-seed records.
- [x] Pass report checks and commit; no test-split evaluator or E3 caching is added.

### T10 - UIT Deployment and Acceptance

**T10 COMPLETE.** The user confirms the completed UIT GPU/resume smoke in Slurm **job 8198**: pytest **1 passed**, `evidence.json` status **PASS**, current train/val hash audit passed, real H200 MIG environment verified, forward/backward finite, and resume replay model/optimizer/scheduler/RNG/loader state **bitwise equal**. **No test images opened.** This is user-executed evidence, recorded on 2026-11-02; the agent did not execute or rerun the smoke. G2 remains PASS. The historical `t10_acceptance.json` preparation record and all run evidence remain unchanged.

- [x] Write E1 entrypoint CPU tests and the opt-in target smoke test in `tests/e1/test_entrypoints.py` and `tests/e1/test_gpu_smoke.py`; CPU checks pass, GPU smoke remains unexecuted by the agent.
- [x] Prepare `scripts/slurm/e1_train.sbatch` and [runbook.md](runbook.md); run local CPU regression, compilation, and shell syntax checks. Exact allocation and seed/declaration/last.pt guards are tested without job submission; see [implementation_log.md](implementation_log.md).
- [x] Record the user's current manually passed UIT CPU regression, H200 MIG environment, real-weight synthetic full-size forward/backward, and real train/frozen-preprocessing forward/backward as user attestations. The agent does not rerun or independently re-verify those checks; no unreported observations are inferred.
- [x] Record the completed user-executed final opt-in UIT GPU/resume smoke: job 8198, pytest 1 passed, evidence.json PASS, current train/val hash audit passed and real H200 MIG environment verified. The bounded checkpoint fixture is not an official completed epoch or tuning outcome.
- [x] Record the user's UIT full-size forward/backward and last.pt resume acceptance: finite forward/backward and bitwise-equal replay model/optimizer/scheduler/RNG/loader state; no test images opened. The user executed the smoke through Slurm; the agent did not execute or rerun it. This is implementation acceptance, not a tuning trial, P1-v2 or extra optimization experiment.
- [x] Record the user's G2 PASS and separate G3a PASS confirmation with the completed T11a handoff; deployment deliverables were committed previously. Neither gate authorizes final training; G3b remains OPEN / NOT APPROVED.

### T11a - Official UIT LR Tuning, Only After G3a

**COMPLETE**, as confirmed by the user and summarized in [t11a_handoff.md](t11a_handoff.md).

- [x] Record the user's G2 PASS and G3a PASS confirmation for completed official tuning.
- [x] Record all four frozen LR candidates `[2.5e-5, 5e-5, 1e-4, 2e-4]`, seed 1001, each with eight completed epochs/eight validation records and 139992 optimizer updates. No successful shortened trial is substituted.
- [x] Preserve existing run evidence, failures/retries and selection artifacts without modification. The supplied CSV contains completed trial histories; it does not enumerate failures/retries or checkpoint lineage, which this documentation update does not reconstruct.
- [x] Record selected LR **`1e-4`**, selected epoch **7**, supported validation Macro-F1 **`0.45248836792097563`**. Recalculation follows tau=1e-6/earlier checkpoint ties and the global-maximum-anchored/lower-LR trial rule; see the handoff table. The existing selection artifact is unchanged.
- [x] **STOP and hand off completed tuning for explicit G3b review. No final seeds launched.** G3b is **OPEN / NOT APPROVED**; seeds **2001/3001/4001 NOT STARTED**. Test images remain **LOCKED**; no E1-HR/E2/E3 work is started.

### T11b - Official UIT Final Training, Only After G3b

**NOT STARTED. G3b OPEN / NOT APPROVED. Final seeds 2001/3001/4001 have not started. Test access LOCKED.**

- [ ] Record the user's explicit G3b review of the completed T11a tuning ledger and selected LR, and authorization of that selected configuration for the three official final seeds. No final seed may run before G3b PASS.
- [ ] Execute selected LR x three distinct official final seeds on the reviewed UIT GPU/Slurm environment, all disjoint from the tuning seed. Start each run freshly from the pinned ImageNet checkpoint; do not warm-start from a tuning checkpoint.
- [ ] Ensure every successful final run completes all eight training epochs and eight full validations, with early stopping OFF and the original scheduler/resume budget unchanged.
- [ ] Generate selected-checkpoint per-seed validation analysis and mean/std reports using the reviewed reporting convention; archive identities/checkpoints, retain failures/retries, and review limitations. Do not open test images.
- [ ] Hand off for separately authorized final evaluation. Keep test images locked and do not start E1-HR/E2/E3 automatically.

## 5. Completion and Evidence Rules

For future authorized implementation, follow each task's technical red/green/check/commit steps in [plan.md](plan.md), subject to this ledger's revised gate/dependency precedence. Older plan wording cannot authorize T01+ before G1 or final seeds before G3b. A task is complete only when its tests and artifact evidence exist, not when its description is written. Keep failed checks, operational interruptions, and retries traceable rather than marking them as completed runs.

Test fixtures may exercise eight epochs with tiny CPU models; they are not UIT measurements or official experiment records. Runtime policy, source identity, and artifact validation must prevent synthetic evidence from becoming official campaign evidence.

G1 remains bound to the exact frozen contract SHA-256 in [g1_approval.json](g1_approval.json). G2 and G3a are PASS and T11a is COMPLETE per the user's current instruction and the supplied four-trial tuning CSV. See [t11a_handoff.md](t11a_handoff.md) for provenance and the unchanged selection rule. **STOP for explicit G3b review**: G3b is OPEN / NOT APPROVED, T11b/final seeds 2001/3001/4001 are NOT STARTED, and test access is LOCKED. Earlier preparation/implementation records retain their historical meaning. This documentation-only update changes no scientific code, numeric contract, selection artifact, run evidence or gate dependency.
