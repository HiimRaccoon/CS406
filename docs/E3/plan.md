# E3 Implementation Plan

> **For agentic workers:** At separately authorized execution, use `executing-plans` for native execution, or `subagent-driven-development` only if the user explicitly selects that method at E3-G1. Implement task by task with checkbox tracking. This plan selects no implementation method and authorizes no implementation, parallel agents or jobs.

**Goal:** Implement and verify the six-arm frozen-representation classifier study, then perform separately authorized cache extraction, a bounded pilot selecting one common head protocol, twelve final head runs and validation closeout.

**Architecture:** E3 owns source-checkpoint verification, deterministic cache indices, head-only sampling/training/recovery, stage gates and reports. Preserve the completed E1/E2 encoders and their feature boundaries, and narrowly reuse unchanged preprocessing and metric mechanics. Standalone cached-feature head training cannot update or reactivate a source encoder.

**Tech stack:** Python/Torch/TorchVision and deterministic FP32 reference in spec Section 10; pytest with nonofficial CPU fixtures; separately authorized target acceptance. E3 optimizer, batch, horizon, pilot envelope and execution-resource choices remain PENDING until their required freezes.

**Spec:** [specs.md](specs.md). **Current status:** The user approved proceeding to plan/task preparation and publication on 2026-11-11. E3-G0 is PASS for design preparation only, as recorded in [tasks.md](tasks.md). E3-G1 and all execution gates remain unapproved. The spec's draft gate table is the earlier review snapshot; this ledger records the later scoped G0 decision without changing scientific spec bytes. Publishing documentation does not freeze a numeric contract.

## Global constraints

- Main report arms are E1 original, E3-E1-N, E3-E1-cRT, E2 original, E3-E2-N and E3-E2-cRT. Source/root seeds are exactly `[2001,3001,4001]`: six source references, twelve new final head runs.
- PRIMARY: cRT minus N within family/source seed/cache/initialization. SECONDARY: N minus original. EXPLORATORY: compare the E1 and E2 balancing effects by matching seed. E3 is not a new controlled E1-versus-E2 architecture experiment.
- ONE common official head protocol across all four retraining arms and all three source seeds: same optimizer family/configuration, LR/schedule, weight-decay policy, batch/accumulation, presentation/update budget, validation cadence, checkpoint comparator and numerics. Only dimension-derived head shape/count and fan-in initialization bounds may differ.
- N/cRT within each family/seed must additionally share exact source/cache bytes and actual initial weight/bias tensors. Only sampling changes; use fresh optimizer/scheduler states in each arm.
- E1 feature is the exact 768-d output of `E1Model.encode`. E2 is `[G;mean(L)]`, 1536-d, with per-view final LayerNorm, one encoder, global-first then batch-major flattened locals. Preserve source final normalization/layer-scale; no added normalization, projection, PCA or learned fusion.
- E1 global WH `[896,384]`; E2 global WH `[896,384]` plus four locals WH `[320,320]`, ordered `left,middle_left,middle_right,right_raised`, directly from frozen raw preprocessing. Both train and val cache extraction have jitter OFF and encoder eval/stochastic-depth OFF.
- Fixed split/labels/preprocessing: 34997 train, 7497 val, 124 outputs and 106 val-supported labels. Verify the exact artifact/config/module hashes and six selected checkpoint references in spec Sections 3–5; do not regenerate split seed 406.
- Six immutable source-cache bundles, twelve split payloads, one feature per original image/source checkpoint. Preserve extracted FP32 values losslessly; on-disk format/dtype/chunk/batch choices are PENDING. Bulk caching requires its own gate.
- Natural sampling is shuffled image permutations without replacement within a pass. cRT draws a class with `1/124`, then an image uniformly within that class, with replacement across draws. Match actual presentations/updates/tails, never nominal epochs alone.
- Only new biased linear weights/biases are trainable: E1 95,356 parameters, E2 190,588; representation trainable count zero. Standard unweighted CE, label smoothing 0, raw 124 logits, true group sample-mean loss.
- Fixed-budget termination, early stopping OFF. Head checkpoint selection uses supported val Macro-F1, chronological improvement `>1e-6`, earlier ties; cadence/horizon PENDING. Every metric comes from that one selected head.
- Top-1/3/5 rank all 124 unchanged logits; exact ties use ascending label ID. Preserve supported-only F1, evaluation-support Weighted-F1, diagnostic sample-mean NLL and full-split cross-bucket false positives.
- Buckets are `1-19`, `20-99`, `100-499`, `>=500`; `1-2` is an overlapping diagnostic subset, with only two val-supported labels/two val images. Resampling never changes unique support/buckets.
- Aggregate all three seeds arithmetically with sample std `ddof=1`, no pre-rounding or ensembles; classification display uses four-decimal percentages and differences use percentage points.
- No E1-HR main arm, StreetCLIP, full-model training, source checkpoint reselection, feature transformation, balancing loss or extra seed/search branch. Completed E1/E1-HR/E2, master, frozen inputs and original evidence remain unchanged.
- Final test remains LOCKED / NOT AUTHORIZED at every stage here. Reject test images/features/previews/predictions before opening, including image-byte hash audits. Metadata access grants no evaluation permission.

## Review focus

| Failure mode | Required behavior / owner |
|---|---|
| A 1536-d cache has swapped halves, pooled locals before LayerNorm, or mixed source-seed chunks | Marker-feature/call-order and cache-index checks reject it; original-head replay checks exact prediction lineage — T02/T03 |
| A family-specific LR/batch/budget is hidden in a declaration although paired N/cRT values match | Compare one common protocol fingerprint across all twelve final runs; allow only computed dimension consequences — T01/T05/T10/T11 |
| A cRT retry consumes different class/image draws or a short final accumulation group changes loss scaling | Restore actual sampler cursor/RNG and all committed state; replay uncommitted work with true presentation denominators — T04/T07/T08/T09 |
| A float-tolerance pass masks changed Top-5 predictions or equal scores hide reordered cache keys | Enforce exact membership/ranking/counts separately from reviewed logits/NLL tolerances — T02/T03/T11 |
| Rehashed analysis files omit predictions, use GT-filtered bucket F1, or pool seed scores incorrectly | Reconstruct discrete metrics from full ordered predictions, validate CSV content/history, aggregate paired deltas before sample std — T06/T11 |

## Authorization sequence and PENDING handling

```text
E3-G0 (PASS: documentation preparation only)
  -> T00b source/cache-pilot contract candidates and reviewed numeric inputs
  -> E3-G1 exact cache/pilot contract freeze + method + named implementation scope
  -> T01-T12 implementation/CPU/deployment preparation
  -> E3-TARGET-AUTH -> T13 bounded source/extraction/runtime/recovery acceptance
  -> E3-G2 readiness review
  -> E3-CACHE-AUTH -> T14 bulk train/val extraction and original-head replay
  -> E3-CACHE-ACCEPT
  -> E3-PILOT-AUTH -> T15 bounded pilot + common-protocol candidate -> STOP
  -> E3-HEAD-FREEZE -> T16 approved final protocol/12 declarations -> STOP
  -> E3-FINAL-AUTH -> T17 twelve head runs/selected analyses
  -> T18 evidence review -> E3-CLOSEOUT
  -> separate study-wide final evaluation, still LOCKED
```

These are exactly the spec's gates; task numbers add no permission. TARGET-AUTH/G2 concern bounded selected-source loading, feature extraction/runtime and deterministic recovery. Optional fixture backward passes update a standalone head only; they neither reopen E1/E2 nor train an encoder. Passing acceptance emits evidence/candidates, never self-approved permissions.

T00b must resolve every input needed to implement/accept the cache and execute each finite pilot candidate. The final selected optimizer/LR/init/budget may remain PENDING until T15/T16, but a pilot candidate must itself be complete. Official-head declarations reject unresolved fields and are unavailable before HEAD-FREEZE. Do not serialize an executable configuration containing `PENDING`, null, implicit defaults or source-recipe fallbacks.

| Spec pending IDs | Required preparation / later outcome |
|---|---|
| P01 | T00b prepares explicit actual source paths/evidence bindings; T02/T13 verify actual binaries and six complete lineages before G2/bulk cache |
| P02/P03 | T00b resolves lossless cache storage, extraction/worker/recovery policy and explicit float replay tolerances before G1/acceptance; T03/T13 prove them |
| P04/P05 | T00b prepares actual pilot source/seed/coverage, finite candidates, caps and common-protocol selection/convergence criteria; PILOT-AUTH permits execution only after accepted caches |
| P06/P07/P08 | T00b supplies complete candidate initializer/optimizer/LR/decay/schedule/batch/draw/update/validation values; T15 chooses ONE common profile and HEAD-FREEZE locks its final values |
| P09/P10 | T00b fixes cRT RNG API/keys/state and applicable pilot runtime/resource/commit boundary before implementation; final bindings remain common across all arms |
| P11 | This plan proposes file/interface ownership; T00b freezes schemas/paths/approval bindings; T12/T13 record actual consumed-source/device evidence rather than invented hashes |

No current task writes the numeric candidates or resolves these scientific choices. Their separate design preparation is T00b. Existing source tuning ID 1001, source LRs and source eight-epoch accounting are history, not E3 pilot/head defaults.

## File ownership and reuse boundary

Paths below are **future deliverables**, except the three E3 documents. No source/test package is created by preparing this plan.

| Proposed files | Responsibility / owner |
|---|---|
| `docs/E3/specs.md`, `plan.md`, `tasks.md` | Scientific spec, implementation design and live gate/task ledger — T00a |
| `docs/E3/e3_source_manifest_v1.json`, `e3_cache_pilot_contract_v1.json` | Reviewed source registry and complete cache/acceptance/pilot input candidates — T00b |
| Contract sidecars and scoped approval receipts | Only after explicit approval of exact bytes at the corresponding existing gate; never generated as permission by a harness |
| `src/e3/__init__.py`, `config.py`; `tests/e3/__init__.py`, `conftest.py`, `test_config.py` | E3 schema/kind/common-protocol/gate guards and private nonofficial fixtures — T01 |
| `src/e3/data.py`, `sources.py`; `tests/e3/test_sources.py` | Exact train/val metadata/preprocessing and strict six-source loading/freeze/replay adapters — T02 |
| `src/e3/cache.py`; `tests/e3/test_cache.py` | Complete immutable cache writing/validation/reads and original-head replay — T03 |
| `src/e3/sampling.py`; `tests/e3/test_sampling.py` | Natural/cRT streams, class indices and resumable sampler state — T04 |
| `src/e3/model.py`; `tests/e3/test_model.py` | Paired head construction/hash, approved optimizer groups and common policy — T05 |
| `src/e3/metrics.py`; `tests/e3/test_metrics.py` | Narrow unchanged metric exports/independent semantic oracles — T06 |
| `src/e3/engine.py`; `tests/e3/test_engine.py` | Head-only presentation accounting, approved schedules, validation/selection — T07 |
| `src/e3/checkpoint.py`; `tests/e3/test_checkpoint.py` | State/atomic authoritative checkpoints and deterministic boundary replay — T08 |
| `src/e3/runtime.py`, `train.py`; `tests/e3/test_runtime.py`, `test_train.py` | Stage-specific inspection/preflight, scientific identity, one-run/attempt lifecycle — T09 |
| `src/e3/campaign.py`; `tests/e3/test_campaign.py` | Bounded pilot ledger/selection and exactly twelve final declarations/complete-run validation — T10 |
| `src/e3/report.py`; `tests/e3/test_report.py` | Selected exports, six-arm aggregation and ranked comparison hierarchy — T11 |
| `tests/e3/test_entrypoints.py`, `test_gpu_smoke.py`; E3-specific approved execution wrappers; `docs/E3/runbook.md` | CPU deployment checks and separately opted-in acceptance harness — T12/T13 |
| `docs/E3/implementation_log.md`, readiness/acceptance/cache/pilot/final/closeout evidence indices | Actual scoped evidence, produced only when each authorized task occurs — T01-T18 |
| `docs/E3/e3_head_protocol_v1.json` and separate approved checksum/receipt | ONE common final head protocol, candidate after T15 and exact-byte freeze at HEAD-FREEZE/T16 |

Reuse unchanged `src/preprocessing/geoguessr_v1.py` and the source model `encode` semantics. Reuse `src/e1/metrics.py` for `Metrics124`/`rank_logits`; use the completed checkpoint comparator unchanged where its input interface fits. Do not reuse full-model train loops, full-model optimizer builders, fixed 139992-update scheduler validation or E1/E2 stage receipts as E3 validators. Implement only the selected head optimizer/schedule algorithms authorized in the candidate contract, not a generic search framework. Source adapters own E3 verification of historical schemas; never monkeypatch source globals, alter completed contracts or import pilot harness code at runtime.

Actual imports of source model classes may transitively consume their package/config modules. Enumerate every actual consumed repository path explicitly at T12/T13, including relevant dependency files and launch wrappers; do not invent a closure from intended direct imports or hash a wildcard. Reporting-only code/harnesses have separate audited inventories unless actually consumed by training. Keep clean commit/job IDs separate from scientific equality, while preserving full immutable configuration/receipt/cache/source integrity checks.

## Record and interface contracts

Use repository-style modules/plain dictionaries, not a new configuration framework. Mandatory normalized records:

- Source entry: family, source seed/run ID, selected epoch, actual binary path/SHA-256, source contract/report/selection bindings and original selected-analysis references. Registry keys are exactly `(E1|E2,2001|3001|4001)`.
- Cache index: source identity; split/feature dimension; ordered `(relative_path,label_id)` keys; full row coverage; extraction/runtime/storage identity; chunk paths/shapes/bytes/hashes; complete-index digest. Only train/val accepted indices are loadable.
- Common head protocol: one initializer rule/seed-binding rule, optimizer/groups, LR trajectory, presentation/update/tail/validation budget, comparator and numerical policy. Family/seed/source/cache/actual initialized tensors are run bindings outside that shared profile; dimensions are derived from family, not tunable overrides. Sampling arm selects one of the two fixed distributions outside the common optimization profile.
- Head declaration: E3 stage `pilot|final`, family, source/root seed, arm `natural|crt`, source/cache/init/protocol fingerprints, original planned budget and explicit applicable receipt paths/bindings. Final declarations reference one HEAD-FREEZE artifact and one common protocol fingerprint across all twelve runs.
- Progress: committed presentation/update/validation counters, optional reviewed pass/window index, full history/selected state and selected head weights. The concrete commit boundary follows the approved contract, never an implicit epoch assumption.
- Proposed distinct kinds: `e3_cache_request_v1`, `e3_cache_index_v1`, `e3_acceptance_request_v1`, `e3_pilot_run_v1`, `e3_final_head_run_v1`, `e3_head_checkpoint_v1`, `e3_selected_validation_analysis_v1`, `e3_study_validation_report_v1`. Final validation rejects acceptance/pilot/fixture artifacts as completed final runs.

Canonical object fingerprints inherit `json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)` with default separators. File-byte hashes are SHA-256. Initial tensor-state identity includes ordered tensor names, dtype/shape and contiguous value bytes, rather than relying on incidental checkpoint serialization metadata. Full schema serialization and exact-byte gate receipts belong to T00b/G1 and T16, not this plan's preparation.

## Task workflow

After the named authorization, each implementation task follows **behavioral RED → minimal GREEN → scoped regression → protected-byte review → scoped commit**. Imports/scaffolding alone are not behavioral RED. Use test-owned nonofficial fixtures/private doubles; no production approval-bypass flag, alternate official dimension, fake runtime receipt or reduced-budget official run. Task checklists below remain unchecked until actual evidence exists.

### E3-T00a — Prepare plan and ledger

**Files:** this plan and [tasks.md](tasks.md). **Inputs:** approved revised spec and the user's documentation/publish request. **Outputs:** reviewable design/task artifacts only.

- [x] Define module ownership, exact interfaces, common-profile constraints, task dependencies and the existing gate sequence.
- [x] Keep P01-P11 explicit; distinguish complete pilot-candidate values from later selected official values; create no numeric contract/defaults.
- [x] Retain the scientific spec bytes; record the later scoped G0 decision in the task ledger.

### E3-T00b — Prepare source/cache-pilot contracts for G1

**Depends:** E3-G0. **Files:** proposed source manifest/cache-pilot contract; related ledger review entries. **Consumes:** spec Sections 3/13/14. **Produces:** complete review candidates and actual candidate digests, no approved sidecars/receipts before explicit G1.

- [ ] Collect P01-P11 decisions/materialized inputs needed by cache/acceptance/pilot; require actual source paths or explicitly blocked source handoff, never fabricated observations.
- [ ] Serialize fixed values and bounded complete pilot candidate protocols; require one common-profile selection rule, reviewed float tolerances and exact runtime/commit/RNG/storage policies. Final selected values may remain explicitly unresolved outside executable pilot candidates.
- [ ] Check JSON/links/hashes/schema/arithmetic without model/image execution, and present exact bytes/digests plus this plan/ledger for G1 scope/method review. Stop before implementation.

### E3-T01 — Contract, common-profile and permission guards

**Files:** config/package/CPU fixtures in ownership table. **Depends:** explicit E3-G1. **Interfaces:** `load_contract(path: Path, *, stage: str) -> dict`; `validate_common_protocol(protocol: dict) -> dict`; `resolve_cache_request(contract: dict, source: dict, paths: dict) -> dict`; `resolve_head_run(contract: dict, protocol: dict, *, stage: str, family: str, seed: int, arm: str, bindings: dict) -> dict`; `require_approval(path: Path, *, gate: str, expected: dict) -> dict`; `fingerprint(value) -> str`.

- [ ] Write `test_common_protocol_has_no_family_override`, `test_final_rejects_pending_or_unfrozen_profile`, `test_wrong_stage_receipt_fails_before_allocation`: four arms share one profile; changed family LR/budget/numerics fails; source IDs/dimensions are fixed; no fixture/pilot approval can authorize final work.
- [ ] Run `python -m pytest tests/e3/test_config.py -q`; record behavioral RED.
- [ ] Implement strict approved-contract parsing, canonical identities and stage guards; retain unresolved final fields only in nonexecutable design/outcome records.
- [ ] Rerun that command plus unchanged E1/E2 config regressions; require PASS with no model/GPU allocation or image reads.
- [ ] Review protected bytes and commit only T01 files plus observed ledger/log evidence.

### E3-T02 — Exact inputs and strict frozen source adapters

**Files:** `data.py`, `sources.py`, `test_sources.py`. **Depends:** T01. **Interfaces:** `FixedManifest(artifacts_dir: Path)` with `records(split: str) -> tuple[dict,...]`; `audit_train_val(manifest, data_root: Path) -> dict`; `verify_source(entry: dict, contract: dict) -> dict`; `load_frozen_source(entry: dict, *, contract: dict, device) -> tuple[Module,dict]`; `make_source_loader(manifest, *, source: dict, request: dict)`.

- [ ] Write source-kind/seed/epoch/hash/strict-state tests and both-layout encoder spies. Assert E1 selected epochs 8/7/8, E2 8/8/8, dimensions 768/1536; E2 calls global then flattened locals and normalizes each view before mean. Sentinels reject test/corrupt/root-escape inputs before opening; all representation parameters frozen, original head preserved and eval mode enforced.
- [ ] Run `python -m pytest tests/e3/test_sources.py -q`; record behavioral RED using tiny nonofficial source fixtures, not copied real binaries.
- [ ] Implement metadata/source-envelope adapters and frozen preprocess/model boundary without changing source builders/validators. Hash and strict-load actual binaries only in the authorized execution path.
- [ ] Rerun plus `tests/test_preprocessing.py`, `tests/e1/test_model.py` and `tests/e2/test_model.py`; require PASS. Record synthetic scope and exact transitive imports.
- [ ] Review protected files and commit scoped T02 changes/evidence.

### E3-T03 — Immutable feature caches and original-head replay

**Files:** `cache.py`, `test_cache.py`. **Depends:** T01/T02. **Interfaces:** `extract_cache(request: dict, *, source_model, loader, output: Path) -> dict`; `validate_cache(index_path: Path, *, expected: dict) -> dict`; `CacheDataset(index_path: Path, *, expected: dict)` returning `(feature,label_id,relative_path)`; `verify_original_replay(source_entry: dict, cache_index: dict, evaluation: dict, policy: dict) -> dict`.

- [ ] Write `test_cache_requires_complete_ordered_manifest`, `test_seed_or_fusion_swap_rejected`, `test_partial_index_cannot_publish`, `test_replay_tolerance_never_waives_prediction_change`. Prove exact train/val row/key/label coverage, finite correct dimensions, lossless values, immutable chunks and exact source-head ranking/count equivalence separate from float tolerances.
- [ ] Run `python -m pytest tests/e3/test_cache.py -q`; record behavioral RED on private small cache fixtures.
- [ ] Implement only the T00b-reviewed storage/chunk/recovery policy, atomic completion index and replay. Reuse one accepted cache by both arms; no jittered bank or feature transform.
- [ ] Rerun with source/preprocessing regressions; require PASS for corruption/reordered labels/mixed source/seed/test-cache rejection and atomic-write failure recovery.
- [ ] Review source/protected hashes and commit scoped T03 files/evidence. Bulk extraction stays gated.

### E3-T04 — Deterministic natural/cRT samplers

**Files:** `sampling.py`, `test_sampling.py`. **Depends:** T01/T03. **Interfaces:** `make_sampler(cache_index: dict, *, policy: str, seed: int, rng_contract: dict)`; sampler `set_window(index: int)`, `state_dict() -> dict`, `load_state_dict(state: dict) -> None`, and iteration of cache row indices for the declared draw budget.

- [ ] Write tests proving natural full-pass permutation, class-first then within-class-uniform cRT mapping, replacement/repeats, all 124 class supports and immutable manifest-ordered class indices. A mocked draw stream pins exact class/image index behavior; test probabilities/algorithm rather than asserting exact empirical class quotas.
- [ ] Run `python -m pytest tests/e3/test_sampling.py -q`; record behavioral RED.
- [ ] Implement the approved API/derived-seed keys and serialized class/image draw/cursor state. Keep init/loader/sampler RNG independent and bucket metadata unchanged.
- [ ] Rerun and compare uninterrupted versus restored suffixes, including window changes and repeated extreme-tail rows; require PASS.
- [ ] Review and commit scoped sampler changes/evidence.

### E3-T05 — Paired linear heads and approved common optimizer

**Files:** `model.py`, `test_model.py`. **Depends:** T01/T03. **Interfaces:** `make_paired_heads(*, family: str, seed: int, protocol: dict) -> tuple[Module,Module,dict]`; `build_head_optimizer(head, *, protocol: dict) -> tuple[Optimizer,dict]`; `head_state_sha256(head) -> str`.

- [ ] Write tests requiring new biased heads, exact paired weight/bias equality and hash, counts 95356/190588 and only head tensors in optimizer once each. Across families, assert common optimizer/LR/decay/schedule/batch/numerics and one initializer rule, allowing only computed fan-in/shape effects.
- [ ] Run `python -m pytest tests/e3/test_model.py -q`; record behavioral RED with isolated CPU RNG/outer-state restoration tests.
- [ ] Implement only approved pilot-candidate initialization/optimizer algorithms, cloning one initial state per pair before training. Reject source-head warm-start and silent optimizer/initializer defaults.
- [ ] Rerun with config/model regressions; require PASS, unchanged cache/source bytes and zero representation gradients/trainables.
- [ ] Review and commit scoped T05 files/evidence.

### E3-T06 — Exact metrics and bucket semantics

**Files:** `metrics.py`, `test_metrics.py`. **Depends:** T01/T02. **Interfaces:** unchanged `Metrics124.update/compute` and `rank_logits` narrow exports.

- [ ] Write independent oracles: all-equal logits rank `[0,1,2,3,4]`; GT `[0,1]`/predictions `[0,0]` yield class-0 F1 `2/3`, supported Macro-F1 `1/3`; cross-bucket false positives retained; unsupported F1 null with completed numeric precision/recall denominators. Verify spec bucket coverage from metadata and overlapping `1-2` membership.
- [ ] Run `python -m pytest tests/e3/test_metrics.py -q`; record behavioral RED.
- [ ] Add narrow unchanged metric reuse; no duplicate or GT-filtered F1 implementation.
- [ ] Rerun with E1/E2 metric regressions; require PASS for all-124 ranking, sample-mean NLL and null no-support buckets.
- [ ] Review and commit scoped T06 files/evidence.

### E3-T07 — Head-only engine, matched budgets and selection

**Files:** `engine.py`, `test_engine.py`. **Depends:** T03/T04/T05/T06. **Interfaces:** `make_head_scheduler(optimizer, *, protocol: dict)`; `train_window(head, loader, optimizer, scheduler, *, protocol: dict, device) -> dict`; `evaluate_head(head, loader, *, manifest, device, collect_predictions: bool) -> dict`; `select_head(current: dict, selected: dict|None) -> bool`.

- [ ] Write a private accounting oracle with **fixture-only** draws=5, microbatch=2, accumulation=2: group denominators 4 then 1, two updates, five presentations. Natural/cRT use equal budgets/validation points/LR trajectories; differing nominal epochs cannot substitute. Pin selected F1 `.5` retaining `.5000005` and replacing at `.500002`, independent of NLL.
- [ ] Run `python -m pytest tests/e3/test_engine.py -q`; record behavioral RED against direct sample-mean optimizer updates and independent approved-schedule endpoints/interior values.
- [ ] Implement cached-head loops and approved schedules/comparator. Keep full fixed budget with no early stopping, finite checks and explicit update/presentation counters; no encoder forward/backward in head training.
- [ ] Rerun plus unchanged comparator/metric regression; require PASS for short tails, flat/decreasing validation curves, scheduler exhaustion and nonfinite failure without skipped credit.
- [ ] Review and commit scoped T07 files/evidence. Fixture values never become official head parameters.

### E3-T08 — Atomic head state and deterministic replay

**Files:** `checkpoint.py`, `test_checkpoint.py`. **Depends:** T01/T04/T05/T07. **Interfaces:** `capture_head_state(*, head, optimizer, scheduler, loaders, sampler, progress: dict) -> dict`; `restore_head_state(state: dict, *, head, optimizer, scheduler, loaders, sampler) -> dict`; `save_head_checkpoint(path: Path, *, state: dict, config: dict, identity: dict, role: str) -> None`; `read_head_checkpoint(path: Path, *, config: dict, identity: dict) -> dict`.

- [ ] Write boundary-zero/discard/reconstruct/replay and atomic-failure tests covering both samplers, all RNG/generators/cursors, scheduler/histories/selected weights and tail-group rollback. Wrong family/seed/arm/cache/init/common profile/role or original source checkpoint must fail.
- [ ] Run `python -m pytest tests/e3/test_checkpoint.py -q`; record behavioral RED.
- [ ] Implement the approved boundary units with authoritative last state and derived selected/history publication repair. Reject best-as-resume and extension after a complete budget.
- [ ] Rerun CPU state-equivalence and publication-failure tests; require exact logical state equality excluding clocks/audit IDs. CUDA state proof remains T13 if applicable.
- [ ] Review and commit scoped T08 files/evidence.

### E3-T09 — Stage preflight, identity and one-run lifecycle

**Files:** `runtime.py`, `train.py`, `test_runtime.py`, `test_train.py`. **Depends:** T01-T08. **Interfaces:** `inspect_inputs(request: dict) -> dict`; `preflight(request: dict, *, operation: str) -> dict`; `run_head(config: dict, run_dir: Path, *, resume: Path|None=None, stop_requested) -> dict`.

- [ ] Write allocation/open sentinels for wrong gates, changed scientific inputs, test access and pilot/final confusion. Test acceptance inspection without circular future G2 requirements; official heads require applicable accepted caches, HEAD-FREEZE and FINAL-AUTH.
- [ ] Run `python -m pytest tests/e3/test_runtime.py tests/e3/test_train.py -q`; record behavioral RED for lifecycle/recovery and immutable attempt records.
- [ ] Implement stage-specific entrypoints, single-writer/attempt records, runtime scientific projection and same-run recovery. Check-config allocates no model/GPU and does not claim execution readiness.
- [ ] Rerun with unchanged checkpoint/runtime regressions; require full declared fixture budget despite worsening F1, no extra updates after terminal-publication failure, docs-only audit-commit eligibility and rejection of consumed-source changes.
- [ ] Review and commit scoped T09 files/evidence; final campaign parsing is wired at T10 before official execution can pass.

### E3-T10 — Bounded pilot and twelve-run campaign guards

**Files:** `campaign.py`, `test_campaign.py`, final validation wiring in `train.py`. **Depends:** T01/T09. **Interfaces:** `validate_completed_head_run(run_dir: Path) -> dict`; `select_common_protocol(pilot_records: list[dict], envelope: dict) -> dict`; `plan_final(protocol: dict, source_manifest: dict, cache_indices: list[dict], paths: dict) -> list[dict]`; `write_declarations(output: Path, declarations: list[dict]) -> None`.

- [ ] Write tests for exact finite pilot envelope/caps/predeclared selection and one resulting profile; reject best-per-family/arm selection, rehashed incomplete runs, extra/missing/cherry-picked final IDs, pilot-as-final and common-profile mismatches.
- [ ] Run `python -m pytest tests/e3/test_campaign.py -q`; record behavioral RED.
- [ ] Implement full committed-run/cache/source/init/history/selected-state parsing and exactly `2 families * 3 seeds * 2 sampling arms = 12` final declarations. Prepare paired initial-state bindings without reusing trained pilot weights; no function submits jobs or self-approves a freeze.
- [ ] Rerun campaign/final-preflight tests; require all twelve share the same common profile and each pair share source/cache/init hashes; pilot completion returns a mandatory review handoff.
- [ ] Review and commit scoped T10 files/evidence.

### E3-T11 — Selected exports and six-arm comparison report

**Files:** `report.py`, `test_report.py`. **Depends:** T02/T03/T06/T10. **Interfaces:** `evaluate_selected(config: dict, run_dir: Path) -> dict`; `validate_analysis(run_dir: Path) -> dict`; `aggregate_study(head_run_dirs: list[Path], source_manifest: dict, cache_indices: list[dict], protocol: dict) -> dict`; `render_study(report: dict) -> str`.

- [ ] Write tests for selected earlier-than-last heads, original-source schema differences and complete ordered predictions/CSV contents despite recomputed hashes. Synthetic per-seed values `[.1,.2,.3]` must yield mean `.2`, sample std `.1`; paired-delta std comes from differences, not subtracted stds.
- [ ] Run `python -m pytest tests/e3/test_report.py -q`; record behavioral RED, including hierarchy labels and complete 12-head/6-original coverage.
- [ ] Implement all eight selected export artifacts and complete content/lineage validation, with unchanged metric semantics, diagnostic NLL evidence, one common profile and explicit PRIMARY/SECONDARY/EXPLORATORY contrasts. Preserve exact source original references; no new architecture claim.
- [ ] Rerun report/metric and unchanged E1/E2 reporting regressions; require rejection of truncated/reordered predictions, false source equality, missing seeds, GT-filtered bucket F1 and ensemble substitution.
- [ ] Review and commit scoped T11 files/evidence; publish analysis index last after all exports succeed.

### E3-T12 — CPU readiness and bounded deployment preparation

**Files:** entrypoint/acceptance tests, approved E3 wrappers, runbook, readiness/log. **Depends:** T01-T11. **Interfaces:** strict CLI operations for check-config, source acceptance, extract-cache, run-head and report; one request/declaration per invocation, with no scientific override/search/test flags.

- [ ] Write CLI/wrapper tests for actual reviewed environment/seed export, stage-specific receipt rejection, same-run last only and no allocation during config checks; record behavioral RED.
- [ ] Implement only the T00b-reviewed execution profile and an opt-in train-only source/extraction/standalone-head/recovery acceptance harness. No GPU opt-in means an explicitly unexecuted check, not acceptance. With opt-in, missing authorization is failure, not skip.
- [ ] Run future verification commands below and record actual outcomes/CPU-versus-target scope; review consumed-source closure and separate harness/reporting inventories.

```bash
python -m pytest tests/e3 -q -m "not gpu"
python -m pytest -q -m "not gpu"
python -m compileall -q src/e3 tests/e3
# For each approved shell wrapper only:
bash -n scripts/slurm/e3_extract.sbatch
bash -n scripts/slurm/e3_head.sbatch
```

- [ ] Review protected bytes and commit scoped preparation artifacts. **STOP before target work until E3-TARGET-AUTH.** CPU tests and skipped target checks do not satisfy G2.

### E3-T13 — Authorized bounded source/extraction/runtime/recovery acceptance

**Depends:** T12 and E3-TARGET-AUTH. **Produces:** actual target evidence/source handoff and unapproved G2 review candidate.

- [ ] Verify real selected-source bytes, six-source registry/input bindings, actual software/device/runtime/allocation and current train/val integrity without reading test bytes. Record clean commit and explicit consumed-source/harness inventories separately.
- [ ] Execute the approved bounded train-only fixture on both source feature families/layouts. Confirm freezing/eval, exact encoder/view/fusion/head dimensions, original-head equivalence and finite standalone-head gradients; no backbone optimizer or source-training continuation.
- [ ] Prove uninterrupted versus discard/reconstruct/replay state equality under the accepted stack, including cRT draws and applicable CUDA state; record float tolerances separately from exact discrete prediction/count requirements.
- [ ] Preserve attempts/failures/measurement scopes and present source/cache declarations plus evidence for E3-G2. Create no self-approved receipt and perform no bulk extraction or pilot selection.

### E3-T14 — Authorized bulk caches and original-head validation replay

**Depends:** T13, E3-G2, E3-CACHE-AUTH. **Produces:** six complete source bundles/twelve train/val split payloads and cache-acceptance package.

- [ ] Execute only approved extraction declarations; require 34997 train/7497 val rows per source, one finite deterministic feature each, correct keys/labels/source identity and lossless stored values.
- [ ] Reuse actual selected original heads for complete val replay against materialized original selected analyses, preserving ordered rankings/counts and reviewed float NLL/logit tolerances. Investigate any mismatch instead of replacing reference checkpoints/results.
- [ ] Validate all chunks/indices/hashes/lineage and atomic completion records. Present the complete package for E3-CACHE-ACCEPT; do not start a pilot automatically.

### E3-T15 — Authorized small pilot and one common protocol candidate

**Depends:** T14, E3-CACHE-ACCEPT and E3-PILOT-AUTH. **Produces:** bounded pilot outcomes and a complete common-head protocol candidate.

- [ ] Execute only the approved candidate envelope/seed/source/cache scope and caps, preserving all failures/retries/histories. Pilot fixtures/runs remain distinct from official final records.
- [ ] Apply the predeclared selection/convergence-evidence rule to choose ONE common protocol across the four retraining arms; no family-specific LR/init/budget or extra candidates after the cap.
- [ ] Present full pilot evidence, common profile and exact proposed final protocol bytes/digest for E3-HEAD-FREEZE. **STOP**; neither candidate production nor a winning score authorizes final runs.

### E3-T16 — Record approved head freeze and prepare final declarations

**Depends:** T15 and explicit E3-HEAD-FREEZE. **Produces:** exact approved final protocol/sidecar/receipt and twelve concrete declarations, not jobs.

- [ ] Verify the exact user-approved bytes and create their external checksum/scoped freeze record. Resolve every official numeric field and validate the unchanged common-profile/dimensional exceptions.
- [ ] Prepare exactly twelve final declarations, one source seed per family/arm, paired identical initial head-state hashes and accepted source/cache bindings. No trained pilot state may initialize finals.
- [ ] Present declarations and evidence for E3-FINAL-AUTH; stop before official head training.

### E3-T17 — Authorized twelve final head runs and selected analyses

**Depends:** T16 and E3-FINAL-AUTH, with applicable accepted identities/caches. **Produces:** twelve complete new head runs and their selected val analyses.

- [ ] Execute only approved declarations using fresh paired initial heads and optimizer/scheduler states. Preserve ONE common profile and exact matched presentation/update/tail/validation accounting across arms and seeds.
- [ ] Complete the full frozen head budget over reviewed same-run attempts; retain last/best/histories/RNG/sampler progress and failure/replay evidence. Do not train source representations or add seeds.
- [ ] Export and verify selected-head analyses for every run; retain six unchanged original-head references and hand off for T18 without any test access.

### E3-T18 — Validation closeout and study handoff

**Depends:** T17. **Produces:** six-arm aggregate/comparison, exact scoped inventory and E3-CLOSEOUT review package.

- [ ] Verify all source/cache/init/declaration/protocol/history/selected-head/prediction/export bindings and complete budgets. State actual binary-byte versus checksum-reference verification scope precisely.
- [ ] Verify three-seed ddof=1 metrics/buckets, paired differences and hierarchy, dimensional caveats, geographic/extreme-tail/seed uncertainty and measured efficiency scopes. All twelve new runs share the one locked common protocol.
- [ ] Present the concrete validation package for explicit E3-CLOSEOUT. Closing E3 grants no final-test permission; final evaluation requires separate study-wide approval after all intended protocols/checkpoints are locked.

## Coverage and current handoff

| Spec sections | Plan owners |
|---|---|
| 1–2: scope, hierarchy, six-arm/common-profile design | T00b/T01/T05/T10/T11/T18 |
| 3–6: source lineage, fixed inputs, exact features/caches/replay | T02/T03/T12/T13/T14 |
| 7–9: paired heads, sampling/accounting, bounded common-profile pilot | T04/T05/T07/T10/T15/T16/T17 |
| 10: runtime/RNG/identity/recovery | T01/T04/T08/T09/T12/T13 |
| 11–12: comparator, exact metrics/buckets, reports/content integrity | T06/T07/T11/T17/T18 |
| 13–15: gates/PENDING/acceptance | T00b/T01/T09/T12-T18 and every execution entrypoint |

The current deliverable is this plan and the task ledger only. E3-G0 permits design preparation; candidate inputs, implementation method, E3-G1 freeze and later stages require their own review. No source implementation, contract, cache, model execution, training or test evaluator is delivered by this document. Next review concerns the plan/ledger and remaining T00b inputs, not an automatic implementation launch.
