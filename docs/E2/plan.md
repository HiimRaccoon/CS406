# E2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL at authorized execution: use `executing-plans` for native execution or `subagent-driven-development` only if that method is explicitly selected at E2-G1. Implement task-by-task and track the steps below. Preparing this document selects no execution method and authorizes no implementation or parallel-agent work.

**Goal:** Implement and verify the specified E2 five-view experiment, then execute its separately approved acceptance, four-trial tuning, three final seeds, and validation closeout without changing the common scientific controls.

**Architecture:** E2 owns its configuration, multi-view data/model/epoch loops, runtime, checkpoint envelopes, campaigns, and reports. Reuse the unchanged E1 scheduler, checkpoint comparator, and metric mechanics where their interfaces fit; both view transfers and fused-loss accounting live in E2. Preserve frozen preprocessing and all completed E1/E1-HR scientific files and evidence.

**Tech stack:** Python 3.11.17; Torch 2.8.0+cu128; TorchVision 0.23.0+cu128; deterministic FP32; the remaining software lock in spec Section 8; pytest CPU fixtures and separately opted-in UIT Slurm target acceptance.

**Spec:** [specs.md](specs.md). **Current status:** E2 is CLOSED per the user's 2026-11-10 server-completion confirmation; T11-T14 and G2/G3/closeout are recorded in [the task ledger](tasks.md) and [server closeout record](server_closeout_2026-11-10.md). Server-side receipts and lineage are not mirrored in this checkout. E3-specific execution and final evaluation/test remain separately gated.

## Global constraints

- Authority is the E2 spec, then the unchanged E1/E1-HR numeric references for inherited mechanics. This plan does not reopen their settled controls. Conflicting implementation conveniences must not become new defaults.
- Global WH `[896,384]`; four local WH `[320,320]`; one shared ConvNeXt-Tiny representation; per-view pretrained final LayerNorm; fused `[G;mean(L)]` dimension 1536; `Linear(1536,124,bias=True)`.
- Execute global first, then one encoder call over batch-major flattened `B*4` locals. Preserve config local order. No per-crop loop/chunking substitution, detached path, auxiliary classifier/loss, or fusion redesign.
- Direct frozen raw preprocessing and one brightness/contrast pair per original image, amplitudes 0.10/0.10. Fixed split and 124-label order; train 34997, val 7497; test metadata only.
- Effective batch/microbatch/accumulation **2/2/1**, `drop_last=False`; **17499 updates/epoch**, **8 epochs/8 full validations/139992 updates**; early stopping OFF. Views are not samples or optimizer updates.
- Natural unweighted fused CE, true original-image denominator including the one-image tail; full fine-tuning from epoch 1; AdamW `[0.9,0.999]`, eps `1e-8`, decay 0.05 only Conv/Linear weights, zero for biases/LayerNorm/layer-scale; clipping OFF, no LLRD, head LR multiplier 1.
- Head initialization `linear_default`, `kaiming_uniform_(a=sqrt(5))`, bias bound `1/sqrt(1536)`, isolated CPU RNG bound to the official seed and outer state restored.
- Inherited update schedule: `N=139992`, `W=1000`, start factor 0.1, absolute floor `1e-6`, exact formula/endpoints in spec Section 6; no warmup reset or horizon/budget extension.
- Independent grid `[2.5e-5,5e-5,1e-4,2e-4]`, tuning seed 1001, final seeds `[2001,3001,4001]`; fresh pinned ImageNet initialization per run. Checkpoint improvement `>1e-6`/earlier; LR eligibility relative to global maximum `M`, `M-score<=1e-6`, lowest eligible LR.
- Deterministic FP32/RNG/worker/software policy exactly as spec Section 8; mean/sample std `ddof=1`, full precision before four-decimal percentage display.
- **09:00:00 per attempt**, one declared run per attempt, possibly multiple attempts per run. Commit initial boundary zero and completed epochs after full validation; replay an uncommitted epoch from its beginning using that run's previous `last.pt`.
- Only a material target-acceptance failure preventing correct protocol execution can trigger explicit user review of inherited controls. Minor operational failures do not reopen controls; consumed-source fixes still require renewed target acceptance.
- Final test remains LOCKED through closeout and until study-wide closure, including E3 if retained, and separate explicit evaluation authorization. This plan starts no E3 caching/training.

### Comparator bindings and parameter-count oracles

**E1 is the primary architectural comparator**, sharing E2's global input at 896×384. **E1-HR is a contextual/complementary comparator**. Reporting and closeout must label that hierarchy; E2-versus-E1-HR is not controlled evidence isolating local views or resolution.

Use reviewed official aggregate artifacts materialized/provided in the execution environment and supplied through explicit actual paths. Do not presume a repository-relative report exists on branch `e2`. The required exact file-byte hashes are:

| Comparator | Role | Required SHA-256 |
|---|---|---|
| Official E1 aggregate | Primary architectural comparator | `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e` |
| Official E1-HR aggregate | Contextual/complementary comparator | `9cb44e8aa98d18065402ea3be750586187aeb4b0e4ff23cfa6badfcc484426cf` |

Verify the supplied bytes against these pins and record actual paths/hashes in reporting provenance. Missing artifacts or mismatches fail comparison/closeout; no repository-path default, substituted artifact, or additional unsupported path/hash is introduced. T00b serializes roles and both exact hashes as required scientific/reporting bindings; T09/T14 enforce them.

The classifier `Linear(1536,124,bias=True)` has a **190,588-parameter** design oracle (`1536*124+124`); expected total E2 parameters are **28,010,716**. These are design/acceptance oracles, not official future observations/results. T00b serializes both counts, T03 checks the actual specified architecture, and T11 verifies the real target model; mismatches fail verification rather than changing the oracle.

## Review focus

| Failure condition | Required behavior and owning task |
|---|---|
| Local tensor has correct dimensions but images/crops are silently exchanged | Marker-feature and call-order oracles prove batch-major reconstruction, local mean, concatenation, and shared parameter gradients — E2-T02/T03/T05 |
| A walltime kill occurs during validation or terminal report publication | Discard uncommitted epoch work; recover selected state atomically; after eight committed epochs publish completion without further training — E2-T06/T07 |
| A clean docs/test-only commit differs from the acceptance commit | Compare scientific/runtime projection; accept unchanged consumed source/runtime while retaining the new audit commit; reject every consumed-source change — E2-T07/T08 |
| A syntactically rehashed report contains incomplete coverage or wrong selected weights | Verify complete lineage and eight-epoch budgets rather than caller scores or superficial hash consistency — E2-T08/T09 |
| E1 comparator lacks newer HR/E2 top-level fields | Validate its actual reviewed schema/contract and derive seeds from `per_seed`; do not require invented fields or demand equal cross-architecture runtime identity — E2-T09 |

## Approval sequence and scope

```text
User-requested corrected spec/plan/ledger preparation (complete)
  -> E2-G0 explicit user review (PASS)
  -> E2-T00b numeric-contract candidate and complete review package (prepared; STOP)
  -> E2-G1 exact-byte freeze + selected method + named implementation permission
  -> E2-T01 ... E2-T10 implementation / CPU / deployment preparation
  -> STOP: separate E2-TARGET-AUTH
  -> E2-T11 actual target acceptance
  -> E2-G2 explicit readiness/identity/declaration review
  -> E2-G3a explicit tuning authorization
  -> E2-T12 four complete trials + locked selection
  -> STOP: E2-G3b explicit final-seed authorization
  -> E2-T13 three final runs / selected validation analyses
  -> E2-T14 / E2-CLOSEOUT review
  -> E3's own specification and gates; final evaluation remains separate
```

The user separately approved E2-G1 at the exact reviewed commit/digest, selected native task-by-task execution and authorized T01-T10 only. That historical scope is recorded by the sidecar and G1 receipt. On 2026-11-10, the user confirmed all later E2 server work and gates complete and instructed E2 closeout; see [server closeout record](server_closeout_2026-11-10.md).

E2-G2/G3 and closeout are now recorded complete per the user's server-completion confirmation; their server receipts and run artifacts are not mirrored in this checkout. E3-specific execution and study-wide final evaluation/test remain separate gates.

## File ownership and reuse boundary

All paths below are planned unless the file already exists. Do not fabricate hashes/status for future files.

| Files | Responsibility | Owner task |
|---|---|---|
| `docs/E2/specs.md`, `plan.md`, `tasks.md` | Scientific requirements, implementation design, live ledger | E2-T00a |
| `docs/E2/e2_numeric_contract_v1.json` | Complete candidate serialization, exact controls, source/provenance and gate semantics | E2-T00b |
| `docs/E2/e2_numeric_contract_v1.sha256`, `e2_g1_approval.json` | Exact-byte freeze and explicit digest-bound approval; created only when G1 is approved | E2-G1 |
| `src/e2/__init__.py`, `config.py` | E2 identity/kinds, recipe/run validation, canonical fingerprints, permission loading | E2-T01 |
| `src/e2/data.py` | Exact manifest, five-view dataset, deterministic order/jitter/loaders and transfer validation | E2-T02 |
| `src/e2/model.py` | Strict pinned builder, shared view encoder/fusion, seeded fused head and exhaustive optimizer groups | E2-T03 |
| `src/e2/metrics.py` | Narrow unchanged `Metrics124`/`rank_logits` exports | E2-T04 |
| `src/e2/engine.py` | E2 multi-view train/evaluate loops; narrow scheduler/comparator reuse | E2-T05 |
| `src/e2/checkpoint.py` | State capture/restore primitives and strict atomic E2 checkpoint envelopes | E2-T06 |
| `src/e2/runtime.py`, `train.py` | Current-input/runtime identity, strict preflight, one-run lifecycle and attempt records | E2-T07 |
| `src/e2/campaign.py` | Declarations, complete-run evidence validation, locked selection, final declaration preparation | E2-T08 |
| `src/e2/report.py` | Selected validation exports, aggregate, primary E1 and contextual/complementary E1-HR comparisons | E2-T09 |
| `tests/e2/__init__.py`, `conftest.py`, `test_config.py`, `test_data.py`, `test_model.py`, `test_metrics.py`, `test_engine.py`, `test_checkpoint.py`, `test_train.py`, `test_recovery.py`, `test_campaign.py`, `test_artifacts.py`, `test_report.py` | Synthetic CPU behavioral evidence, isolated from official evidence | E2-T01–T09 |
| `tests/e2/test_entrypoints.py`, `test_gpu_smoke.py`, `scripts/slurm/e2_train.sbatch`, `docs/E2/runbook.md` | CLI/wrapper checks, single-declaration deployment, separately opted-in acceptance harness | E2-T10/T11 |
| `docs/E2/implementation_log.md`, `cpu_readiness.json`, `target_acceptance.md`, `tuning_handoff.md`, `closeout_review.md` | Observed task/stage evidence, created when that work occurs | Respective later stages |

Reuse unchanged `src/e1/engine.py` only for `UpdateScheduler`, `select_checkpoint`, and `NumericalFailure`, and unchanged `src/e1/metrics.py` for metrics/ranking. E2 implements its own dictionary-view epoch loops; do not re-export E1 `train_epoch`/`evaluate`. Do not call E1/HR config, recipe, manifest constructor, builder, runtime, checkpoint, campaign, or report validators as E2 validators; their identities and shapes belong to completed experiments. Do not monkeypatch production globals to bypass those contracts or import `src/pilot/` at runtime.

The official model always uses the pinned architecture/weights and exact input contract. CPU tests may directly construct tiny stand-ins or inject private test doubles; no production fixture switch, alternate weight digest, lower official budget, or small-resolution CLI is introduced. Mark every synthetic evidence envelope nonofficial and make campaign validation reject it.

### Explicit scientific/runtime closure

The planned consumed repository paths are:

```text
src/__init__.py
src/e2/__init__.py
src/e2/config.py
src/e2/data.py
src/e2/model.py
src/e2/metrics.py
src/e2/engine.py
src/e2/checkpoint.py
src/e2/runtime.py
src/e2/train.py
src/e2/campaign.py
src/e1/__init__.py
src/e1/engine.py
src/e1/metrics.py
src/preprocessing/__init__.py
src/preprocessing/geoguessr_v1.py
configs/preprocessing/geoguessr_v1.json
configs/preprocessing/geoguessr_v1.sha256
requirements.txt
pyproject.toml
scripts/slurm/e2_train.sbatch
```

Normal imports consume the package markers; final preflight consumes the E2 campaign selection validator. Review actual transitive consumption at implementation/acceptance, add any necessary extra path explicitly through review, and reject missing/omitted consumed files. Do not hash broad wildcards. `src/e2/report.py` is reporting audit tooling unless actually imported on the training path; keep that import out of the intended runtime design.

Hash each actual consumed file, then hash the map with the established canonical serialization: UTF-8 `json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False)` using its default separators. Exact file-byte SHA-256 and canonical object fingerprints have distinct meanings. The later contract records this policy/planned closure; acceptance records actual file hashes after implementation.

Maintain a separate exact-hash audit inventory for all actual `tests/e2/` harness/helper files, optional test package markers, acceptance outputs, and reporting tooling. Record path/bytes/SHA-256/role and retain immutable snapshots. Test-only proof corrections require evidence review when they affect proof validity; unrelated test edits do not invalidate runtime source equality.

`identity.scientific_runtime` contains the reviewed consumed-source map/fingerprint, E2 contract, software/effective runtime, device/driver and reviewed allocation policy, current data root, fixed inputs/weights, and conformance binding. `identity.audit` records the clean Git commit/cleanliness, harness inventory, and per-attempt operational identifiers. Compare the scientific projection across acceptance/run/resume/campaign, not the entire audit dictionary or Git SHA. A docs/test-only clean commit is eligible only when all scientific/runtime bindings remain equal; any consumed-file change requires renewed target acceptance. Full config, receipts, evidence and checkpoint lineage checks remain independent and mandatory.

## Interfaces and serialized records

Use plain dictionaries and repository-style modules, not a generic configuration framework. The following names are the task-to-task interface contract.

| Module | Public interface |
|---|---|
| `config` | `load_recipe(path: Path = CONTRACT_PATH) -> dict`; `validate_recipe(recipe: dict) -> dict`; `resolve_run(recipe: dict, *, stage: str, lr: float, seed: int, paths: dict) -> dict`; `validate_run(config: dict) -> dict`; `validate_acceptance_request(request: dict) -> dict`; `fingerprint(value) -> str`; `scientific_run_payload(config: dict) -> dict`; `runtime_identity(identity: dict) -> dict`; `require_approval(path: Path, *, gate: str, contract_sha256: str, bindings: dict) -> dict` |
| `data` | `FixedManifest(artifacts_dir: Path)` with labels/support/fingerprints, `records(split: str) -> tuple`, `verify_images(data_root: Path, splits=("train","val")) -> dict`; `E2Dataset(manifest,data_root,split,seed,augmentation)`; `EpochSampler.set_epoch(epoch: int)`; `E2Loader.prepare_resume(completed_epoch: int)`; `make_loaders(manifest,data_root: Path,config: dict) -> dict`; `move_views(views: dict, *, device, batch_size: int) -> dict` |
| `model` | `E2Model(backbone,initialization: dict)`; `encode_view(images) -> Tensor[B,768]`; `encode(views: dict) -> Tensor[B,1536]`; `forward(views: dict) -> Tensor[B,124]`; `build_e2(weights_path: Path,expected_sha256: str,initialization: dict) -> tuple[Module,dict]`; `build_optimizer(model,config: dict) -> tuple[Optimizer,dict]` |
| `metrics` | Unchanged `Metrics124.update(logits,targets,loss_sum)`, `.compute(n_train,labels) -> dict`, `rank_logits(logits)` |
| `engine` | Reused `UpdateScheduler(optimizer,schedule,total_updates)` and `select_checkpoint(current: dict,selected: dict|None) -> bool`; E2-owned `train_epoch(model,loader,optimizer,scheduler,scaler,*,device,runtime: dict,accumulation: int,clipping: dict) -> dict`; `evaluate(model,loader,*,device,runtime: dict,n_train: dict,labels: list[str],collect_predictions: bool=False) -> dict` |
| `checkpoint` | `capture_state(*,model,optimizer,scheduler,scaler,loaders,progress: dict) -> dict`; `restore_state(state: dict,*,model,optimizer,scheduler,scaler,loaders) -> dict`; `save_checkpoint(path: Path,*,model,optimizer,scheduler,scaler,config,progress,identity,loaders) -> None`; `read_checkpoint(path: Path,*,config,identity) -> dict`; `restore_checkpoint(path: Path,*,model,optimizer,scheduler,scaler,config,identity,loaders) -> dict` |
| `runtime` | `inspect_target_inputs(config: dict) -> dict`; `prepare(config: dict,checked: dict) -> dict` |
| `train` | `preflight(config: dict) -> dict`; `run(config: dict,run_dir: Path,*,resume: Path|None=None,stop_requested: Callable[[],bool]) -> dict` |
| `campaign` | `plan_tuning(recipe,paths) -> list[dict]`; `write_declarations(output: Path,declarations: list[dict]) -> None`; `validate_completed_run(run_dir: Path) -> dict`; `select_lr(recipe,trial_records: list[dict]) -> dict`; `write_selection(path: Path,recipe,selection) -> None`; `read_selection(path: Path,recipe) -> dict`; `plan_final(recipe,selection,paths) -> list[dict]` |
| `report` | `evaluate_selected(config,run_dir: Path) -> dict`; `aggregate_final(run_dirs,recipe,selection) -> dict`; `compare_references(e2_report: dict,*,e1_report_path: Path,e1hr_report_path: Path) -> dict` |

Additional interface requirements:

- Dataset items are `(views,label_id,manifest_relative_path)`; `views` has exactly `global` and `local`. Default collation produces `[B,3,384,896]` and `[B,4,3,320,320]`. `move_views` checks keys, FP32 shapes and matching B, transfers both tensors, and returns the same view structure. B is original-image count, never local flattened count.
- Declarations use `kind=official_e2_run_config_v1`, `experiment=E2`, stage `tuning|final`, seed, LR, run ID `e2-{stage}-lr{lr:.8f}-seed{seed}`, full recipe/budget, contract hash, `views`, `representation_dimension=1536`, `fusion=global_then_mean_local`, `view_execution=global_then_flattened_locals_batch_major`, paths, `scientific_sha256` and `config_sha256`. Final declarations add `selection_sha256`.
- `views` explicitly stores global/local WH, local count and the frozen local order. `scientific_run_payload` projects immutable scientific run settings including stage/seed/LR/budget/views/fusion/contract. Compute `scientific_sha256` first; `config_sha256` hashes the full declaration excluding only its own hash. Per-attempt audit metadata is outside the immutable declaration, so an allowed docs/test-only commit does not change its digest.
- Paths explicitly bind data root, weights, run/evidence roots, conformance, acceptance, and stage authorization records; final adds selection. Predetermined receipt paths may precede their creation. Receipts bind declaration digests independently: do not create a declaration↔future-receipt hash cycle. Reject unknown scientific overrides.
- `compare_references` receives explicit `e1_report_path`/`e1hr_report_path` inputs for reviewed official artifacts provided in the execution environment. It verifies the exact comparator hashes/roles from the reviewed reporting bindings above (later serialized in the E2 contract), not caller-selected expected hashes or assumed repository paths. Report provenance records resolved input paths and verified digests; comparator roles are not interchangeable.
- Acceptance requests use a distinct `kind=e2_target_acceptance_request_v1`, `stage=acceptance`, the same frozen recipe/contract/topology, seed1001/baseLR2.5e-5, reviewed paths and target-work approval binding, and the bounded fixture policy in T10. `validate_acceptance_request` accepts no official-run claims. This verification LR is not a selected E2 LR; the harness cannot publish an official run. Official `validate_run` and training CLI reject this request kind.
- Kinds are `official_e2_checkpoint_v1`, `official_e2_run_summary_v1`, `e2_lr_selection_v1`, `e2_selected_validation_analysis_v1`, `e2_three_seed_validation_report_v1`, and `e2_vs_e1_e1hr_validation_comparison_v1`. All carry applicable E2 topology/stage/seed/LR/budget/contract/identity and lineage; aggregates retain all per-seed bindings.
- Boundary progress has `completed_epoch`, committed `updates`, full `history`, `selected`, and `selected_weights`. Official boundaries require `updates=completed_epoch*17499`, complete validation/history for each committed epoch, and no progress beyond 8. Best is reporting-only; last is the full authoritative resume state.
- State primitives capture/restore model, optimizer, full scheduler, inactive scaler policy, Python/NumPy/CPU/all-CUDA RNG, loader states and progress. Strict official envelope validation wraps those primitives; acceptance fixtures use the same primitives inside a distinct nonofficial envelope and cannot pass official checkpoint/campaign validation.
- `inspect_target_inputs` validates the explicitly identified official declaration or acceptance-request schema, then current integrity/source/runtime/allocation without requiring future G2/G3 receipts. `prepare` allocates only after the caller's acceptance or official permissions have been checked. Official `preflight` accepts only official declarations and additionally requires G2 and the relevant G3a/G3b bindings. There is no production CLI approval-bypass flag.
- Training CLI: `python -m src.e2.train --config PATH [--check-config | --resume SAME_RUN/last.pt]`. Check-config validates recipe/declaration/G1 integrity without model/GPU allocation, and identifies missing later gates without claiming execution preflight has passed. One invocation serves one declaration; all recipe overrides/search/test flags fail.

## Task workflow

After the appropriate gate, execute listed tasks in dependency order. Each implementation task records **RED → GREEN → scoped regression → diff/protected-hash review → scoped commit** in `docs/E2/implementation_log.md`. A missing import alone is not a behavioral RED; after minimal scaffolding, demonstrate the intended assertion failure. Record actual commands/results/environment and synthetic-versus-target scope; never prefill pass counts or commit IDs.

CPU tests use temporary output roots and explicitly nonofficial evidence. Do not open actual test images. Run relevant unchanged E1/HR/preprocessing regressions alongside each affected task, then the complete CPU suite at E2-T10. Skipped GPU checks are not target evidence. At execution, use the actual reviewed Git checkout and scoped staging; this plan invents no branch/commit or push instruction.

### E2-T00a — Draft specification, plan, and task ledger

**Status:** COMPLETE — DRAFT DOCUMENTATION ONLY. **Files:** `docs/E2/specs.md`, this plan, `tasks.md`. The spec receives only preparation-status wording/link synchronization; scientific sections remain unchanged.

- [x] Record the requested scope, exact E2 architecture and inherited controls, module interfaces, numerical oracles, dependency order and approval stops.
- [x] During T00a, kept E2-G0 awaiting review and recorded the numeric candidate as later E2-T00b work. The user subsequently approved G0 explicitly; no implementation, target work, tuning, finals, or closeout is complete.
- [x] Perform document/link/ID/interface/number checks and compare protected E1/HR/data/preprocessing hashes. These are documentation checks, not model/test execution.

### E2-T00b — Numeric-contract candidate and G1 review package

**Depends:** explicit E2-G0 approval, supplied by the user on 2026-11-05 for T00b only. **Status:** COMPLETE — CANDIDATE/DOCUMENTATION ONLY. **Files:** [e2_numeric_contract_v1.json](e2_numeric_contract_v1.json) and synchronized design-status entries. Candidate: **86,335 bytes**, exact SHA-256 **`ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72`**. All checks below are document/JSON/hash/arithmetic checks; no CPU/GPU tests ran.

- [x] Serialize every exact spec value, including five-view execution order, fused head fan-in 1536, input fingerprints, batch/update budget, optimizer/groups, scheduler formula, RNG/runtime, selection/reporting, attempts, consumed-source policy, approvals and test prohibition.
- [x] Include required scientific/reporting bindings for **E1 primary architectural / E1-HR contextual-complementary roles**, exact E1 aggregate SHA-256 `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e`, exact E1-HR aggregate SHA-256 `9cb44e8aa98d18065402ea3be750586187aeb4b0e4ff23cfa6badfcc484426cf`, and the parameter-count oracles: fused `Linear(1536,124,bias=True)` classifier **190,588**, total E2 **28,010,716**. Comparator artifacts are supplied/materialized at execution; do not invent their physical paths or other hashes.
- [x] Check candidate completeness explicitly: both comparator roles and exact hashes must be serialized, and the classifier/total parameter oracles must be JSON integers `190588`/`28010716`. Missing or swapped roles, missing or incorrect hashes, or missing or incorrect counts block the E2-G1 review package. Prose references and report-time defaults are insufficient; comparator paths remain explicitly supplied execution-environment inputs without a repository-relative fallback.
- [x] Verify inherited fields against the two frozen numeric references; limit E2 changes to its specified topology/head/view behavior and isolated identity. Provenance records exact preparation snapshot paths/bytes/hashes; later ledger/status changes do not mutate a frozen contract.
- [x] Parse JSON, check arithmetic/formulas/endpoints and reference hashes, resolve all scientific values, and compute exact candidate file SHA-256. Distinguish future observations from unresolved scientific controls; no selected LR/job/result hashes are filled.
- [x] Present the full candidate, plan and ledger for G1 review and method/scope selection. Create no `.sha256` sidecar or approved receipt before explicit approval of the exact digest. Stop before implementation.

### E2-T01 — E2 configuration, identities, and CPU fixture boundary

**Depends:** E2-G1 freeze/implementation permission. **Files:** E2 package/config, `tests/e2/__init__.py`, `conftest.py`, `test_config.py`. **Consumes:** frozen E2 contract/G1 receipt. **Produces:** config interfaces and serialized declaration/kind contracts above.

- [x] RED: add `test_e2_topology_and_common_controls_are_immutable`, `test_other_stage_receipts_are_rejected`, and `test_final_requires_locked_selection`. Assert 4 tuning declarations with seed 1001, separate final seed set, five-view identity, head 1536→124, exact 17499/139992 accounting, G1 digest binding, and rejection of E1/HR/pilot kinds or unknown overrides.
- [x] Run `python -m pytest tests/e2/test_config.py -q` and record behavioral RED.
- [x] GREEN: implement strict recipe/declaration validation, separate nonofficial acceptance-request validation and canonical hashes; test that acceptance requests cannot enter the official training CLI/campaign path. Configuration checks allocate no model/GPU and submit no jobs. Synthetic fixtures are test-owned and cannot alter production approvals or reviewed counts.
- [x] Run that command plus `python -m pytest tests/e1/test_config.py tests/e1hr/test_config.py -q`; require all selected cases PASS.
- [x] Review protected hashes and record evidence; commit only this task's files and ledger/log entries.

### E2-T02 — Fixed manifest, five-view inputs, and deterministic loaders

**Depends:** E2-T01. **Files:** `src/e2/data.py`, `tests/e2/test_data.py`. **Consumes:** validated E2 recipe/declaration and unchanged `Preprocessor`. **Produces:** manifest/dataset/sampler/loader/transfer interfaces above.

- [x] RED: add `test_both_layouts_global_and_locals_match_frozen_preprocessor`, `test_one_jitter_pair_is_shared_by_all_views`, `test_collation_and_transfer_preserve_original_images`, and test-before-open guards. Use synthetic raw RGB canvases of both real layout sizes; independently extract config rectangles from one processed raw tensor and verify locals against direct frozen resize/normalize, not crops from global.
- [x] Pin the structural assertions below, tail B=1, natural permutation/no duplicates, deterministic image-keyed jitter, worker epoch propagation, path containment, corrupt/missing data failure, and test split rejection before an image-open sentinel.

```python
assert views["global"].shape == (2, 3, 384, 896)
assert views["local"].shape == (2, 4, 3, 320, 320)
assert len(targets) == len(paths) == 2
```

- [x] Run `python -m pytest tests/e2/test_data.py -q` to behavioral RED; implement the owned data path and dual-tensor transfer without changing preprocessing.
- [x] GREEN/regression: rerun it plus `python -m pytest tests/test_preprocessing.py tests/e1/test_data.py tests/e1hr/test_data.py -q`; require PASS, unchanged frozen hashes, then log and commit.

### E2-T03 — Pinned shared encoder, exact fusion, and fused-head initialization

**Depends:** E2-T01/T02. **Files:** E2 model and `test_model.py`. **Consumes:** view dictionary and reviewed initialization/optimizer settings. **Produces:** model/builder/optimizer interfaces above, including `encode(views)` for the future E3 feature definition.

- [x] RED: add `test_encoder_calls_global_then_batch_major_locals`, `test_fused_feature_is_global_then_local_mean`, `test_all_view_paths_backpropagate_to_one_encoder`, and `test_head_initialization_uses_fan_in_1536_and_restores_cpu_rng`. Use a tiny CPU backbone/spy and markers distinguishing both images and all crops; verify two calls of sizes B and B*4, exact reshape/mean/concatenation, no duplicated encoder parameters, and finite gradients on both paths.

```python
assert call_batch_sizes == [2, 8]
assert fused.shape == (2, 1536)
assert torch.equal(fused[:, :768], expected_global)
assert torch.equal(fused[:, 768:], expected_locals.mean(1))
assert model.classifier.in_features == 1536
assert model.classifier.out_features == 124
```

- [x] Pin exact seed-bound `nn.Linear(1536,124)` initialization inside an isolated CPU RNG context, full trainability, final LayerNorm before averaging, stochastic-depth endpoints, strict offline filename/full hash/load rejection, and complete nonduplicated decay groups. Test different chunking/order is not silently substituted. CPU random-weight fixtures are never real pretrained evidence.
- [x] Add `test_e2_architecture_parameter_count_matches_design_oracle` on the actual specified ConvNeXt-Tiny architecture, separately from tiny-backbone spy tests. Verify classifier `1536*124+124 = 190,588` and total **28,010,716** parameters; this mandatory design check catches duplicated backbones, a wrong head or extra projection. It is not an observed official result, and tiny stand-ins are not subject to the full-architecture total.
- [x] Run `python -m pytest tests/e2/test_model.py -q` to RED; implement the builder and model without importing pilot/E1/HR recipe validators.
- [x] GREEN/regression: rerun it plus `python -m pytest tests/e1/test_model.py tests/e1hr/test_model.py -q`; review source diff/groups/protected hashes, log and commit. Real-weight full-size target evidence belongs to E2-T11.

### E2-T04 — Metrics reuse with independent numerical oracles

**Depends:** E2-T01/T02. **Files:** E2 metrics and `test_metrics.py`. **Consumes/produces:** unchanged `Metrics124` and ranking interfaces.

- [x] RED: assert all-equal logits rank `[0,1,2,3,4]` without changing logits; target label 4 misses Top-1/3 and hits Top-5. Rank every one of 124 outputs, including unsupported labels.
- [x] Pin cross-bucket GT `[0,1]`, predictions `[0,0]`: class-0/tail F1=2/3 and supported Macro-F1=1/3. Preserve false positives from the head bucket. Unsupported F1 and no-support bucket metrics are null, supported zero-TP F1=0, and NLL is sample-mean CE. Verify fixed bucket coverage using metadata only.
- [x] Run `python -m pytest tests/e2/test_metrics.py -q` to RED; add narrow reuse exports, not a second metric implementation.
- [x] GREEN/regression: run it plus `python -m pytest tests/e1/test_metrics.py tests/e1hr/test_metrics.py -q`; require PASS, log/protected-hash review, commit.

### E2-T05 — Multi-view epoch loops, sample denominators, schedule and comparator

**Depends:** E2-T02/T03/T04. **Files:** E2 engine and `test_engine.py`. **Consumes:** `move_views`, model/metrics, unchanged E1 scheduler/comparator. **Produces:** E2 train/evaluate interfaces above.

- [x] RED: add `test_dictionary_views_transfer_and_original_image_counts`, `test_one_image_tail_matches_direct_sample_mean_steps`, and `test_evaluate_preserves_manifest_paths_and_same_logits`. Compare a tiny fused model with a direct optimizer/CE reference for batch sizes 2 then 1; count three original samples, two updates, and no per-view loss. Verify both tensor transfers, raw logits, sample-weighted loss/NLL, prediction ordering, and finite checks.
- [x] Pin independent scheduler LR oracle `L=1,W=2,N=6,start=.1,f=.01`: `[.1,.55,1,.7525,.2575,.01]`; official u=0/1000/139991 endpoints and exact floor; reject over-budget update/changed restored horizon. Validation advances neither scheduler nor optimizer. Keep eight-epoch control-flow checks in T07.
- [x] Pin selected F1=.5 retaining later .5000005 despite better NLL, replacing at .500002, and retaining equal/lower values. Reject AMP/scaler/clipping and malformed view/label batches; nonfinite logits/loss/gradients or OOM propagate without a skipped update.
- [x] Run `python -m pytest tests/e2/test_engine.py -q` to RED; implement only multi-view loops and narrow numerical reuse. No single-tensor E1 loop re-export.
- [x] GREEN/regression: rerun it plus `python -m pytest tests/e1/test_engine.py tests/e1hr/test_engine.py -q`; require PASS, log/diff/protected-hash review, commit.

### E2-T06 — Atomic official checkpoints and shared recovery primitives

**Depends:** E2-T01/T02/T03/T05. **Files:** E2 checkpoint and `test_checkpoint.py`. **Consumes:** validated config/runtime projection and model/optimizer/scheduler/loaders. **Produces:** state primitive and strict envelope interfaces above.

- [x] RED: add `test_official_e2_boundary_and_topology_are_required`, `test_discard_restore_and_replay_matches_uninterrupted_state`, and `test_atomic_failure_retains_previous_last`. Reject wrong experiment/kind/view execution/head dimension, cross-run seed/LR/config/contract/runtime, best.pt resume, incomplete state, and invalid official epoch/update/history relationships.
- [x] Compare tiny CPU dropout fixtures across save → uncommitted work → reconstruction → restore → replay: exact model/optimizer/scheduler/Python/NumPy/CPU RNG/loader/selected weights/progress equality. Mark fixture envelopes nonofficial; strict production boundary accounting remains unchanged. CUDA RNG is additionally proved at T11.
- [x] Inject failed last.pt replacement and crashes between authoritative last and derived best/history publication. Prior boundary remains readable; selected weights in last permit reconstruction. Boundary zero contains initial full state; completed-run training extension is rejected.
- [x] Run `python -m pytest tests/e2/test_checkpoint.py -q` to RED; implement E2 envelopes around common capture/restore primitives, atomic fsync/replace and single-run identity checks. Primitives expose no CLI permission/budget bypass.
- [x] GREEN/regression: rerun it plus `python -m pytest tests/e1/test_checkpoint.py tests/e1hr/test_checkpoint.py -q`; require PASS, log/protected-hash review, commit.

### E2-T07 — Current-target preflight, runtime and one-run/attempt lifecycle

**Depends:** E2-T01–T06. **Files:** E2 runtime/train, `test_train.py`, `test_recovery.py`. **Consumes:** all prior interfaces; `read_selection` from T08 is wired when that task lands. **Produces:** inspection/preparation/preflight/run and CLI interfaces above.

- [x] RED: sentinel allocation tests reject missing/wrong G1/G2/stage permission, altered topology/source/runtime/inputs, invalid Slurm policy and stale conformance before model allocation. Keep target inspection separate from future G2/G3 receipt checks so authorized acceptance has no circular prerequisite. Full current train/val integrity is required before target model allocation.
- [x] Test actual scientific projection comparisons: a different clean docs/test-only commit with identical consumed source/runtime is accepted; a changed consumed file fails and needs renewed acceptance. Harness edits affect the separate exact audit inventory. Changed config/receipt/evidence hashes still fail. Enumerate actual transitive sources; reject missing/unlisted consumption.
- [x] Run a test-owned eight-epoch CPU integration fixture with flat/decreasing F1, dual-view loaders and private injected setup/preflight doubles. Assert eight train/evaluate calls, full original fixture horizon, earlier selected best and complete histories; every output remains synthetic and fails official campaign validation. Production has no fixture flag or reduced-budget default.
- [x] Cover failure before boundary zero, mid-training, during validation, boundary stop, worker reconstruction, repeated attempts, second-writer rejection, and terminal publication crash after epoch 8. Restore/replay the entire uncommitted epoch; after eight committed epochs recover publication without another update. Preserve failures even when the same unstarted declaration legitimately initializes after no boundary was ever saved.
- [x] Preserve immutable per-attempt request/environment/start-boundary/outcome/event records. A kill may leave unknown partial work; distinguish observed replay work from committed progress. Actual clean commit and job/MIG instance IDs are audit data, not equality keys.
- [x] Run `python -m pytest tests/e2/test_train.py tests/e2/test_recovery.py -q` to RED, implement the one-run path, then GREEN. Regression: `python -m pytest tests/e1/test_entrypoints.py tests/e1/test_recovery.py tests/e1hr/test_train.py tests/e1hr/test_recovery.py -q`; require PASS. Log/diff/protected-hash review and scoped commit. Official final preflight remains fail-closed until T08 selection validation is integrated.

### E2-T08 — Evidence-validated campaign, locked selection and mandatory STOP

**Depends:** E2-T01/T07. **Files:** E2 campaign, `test_campaign.py`, `test_artifacts.py`, final-selection integration in E2 train. **Consumes:** declaration/config/checkpoint/attempt records. **Produces:** campaign APIs above and validated final preflight binding.

- [x] RED: test exclusive creation of exactly four tuning declarations/seed1001 and three fresh final declarations tied to selection. Declaration preparation never submits jobs. Final production preflight now calls `read_selection` and rejects absent/changed selection or wrong final seed/LR.
- [x] Pin grid-order selected scores `[.5000000,.5000006,.5000012,.4900000]` choosing `5e-5` under the global maximum rule, invariant under all record permutations. Reject missing/duplicate/extra/short/failed candidates, caller-supplied score shortcuts, mixed accepted identity, and a fifth/replacement LR.
- [x] Verify full declaration ↔ summary ↔ eight-row history ↔ best/last ↔ selected metrics ↔ attempt inventory lineage, 34997 train/7497 val per epoch, 106 supported labels/124 logits, and 139992 updates. Reject synthetic fixtures and superficial rehashes with wrong selected weights, metric epoch, coverage, or source/runtime projection. Operational audit changes alone remain eligible.
- [x] Run `python -m pytest tests/e2/test_campaign.py tests/e2/test_artifacts.py -q` to RED; implement strict owned parsing/selection. A locked `e2_lr_selection_v1` records `STOP_FOR_USER_E2_G3b_REVIEW` and `final_execution_authorized=false`; no future permission is inferred from a score.
- [x] GREEN/regression: rerun those tests and T07 final-preflight cases, then `python -m pytest tests/e1/test_campaign.py tests/e1/test_artifacts.py tests/e1hr/test_campaign.py tests/e1hr/test_artifacts.py -q`; require PASS. Log/protected-hash review, commit. An uncompletable trial stops for review without changing science.

### E2-T09 — Selected validation exports, aggregate, primary E1 and contextual E1-HR comparisons

**Depends:** E2-T02–T08. **Files:** E2 report and `test_report.py`. **Consumes:** validated complete runs/selection and E2 evaluate/metrics. **Produces:** report APIs/kinds above.

- [x] RED: test best earlier than last; all metrics/predictions/analysis exports must use that selected checkpoint/history, not a separate per-metric optimum. Export overall/per-class/buckets/predictions/confusion/most-confused pairs/curves with an analysis index published last. Bind E2 view/fusion/source/config/selection/checkpoint/history identity and reject stale inputs.
- [x] Require exactly the three complete distinct final seeds under one locked selection. Independent unrounded values `[.1,.2,.3]` give mean=.2, sample std=.1, display `20.0000% +/- 10.0000%`. Preserve null support; reject missing/extra/cherry-picked seeds or ensembles.
- [x] Consume reviewed official aggregate artifacts materialized/provided in the execution environment through explicit paths; do not assume repository-relative reports exist. Require E1 SHA-256 `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e` and E1-HR SHA-256 `9cb44e8aa98d18065402ea3be750586187aeb4b0e4ff23cfa6badfcc484426cf` from the required reporting bindings. Missing inputs, mismatched hashes, or swapped roles fail; record actual paths and verified digests in provenance. Verify contract/benchmark/metric conventions. E1's reviewed schema derives seeds from `per_seed` and does not require newer HR/E2 top-level fields. Obtain fixed-input semantics from each frozen contract where its report schema references them. Compare shared benchmark/metric definitions, not cross-architecture consumed-source equality. Paired deltas join seed IDs and use percentage-point units.
- [x] Add `test_comparator_artifacts_are_explicit_hash_bound_inputs` and `test_report_labels_primary_and_contextual_comparisons`. Reports identify **E1 as primary architectural comparator** because global input is 896×384 and **E1-HR as contextual/complementary comparator**; E2-versus-E1-HR must not claim controlled local-view or resolution evidence.
- [x] Run `python -m pytest tests/e2/test_report.py -q` to RED; implement owned report validation and the stated comparator hierarchy. Record validation-only/geographic/extreme-tail/three-seed limitations and measured efficiency scopes; no pilot/best-seed comparator or test evaluator.
- [x] GREEN/regression: rerun it plus `python -m pytest tests/e1/test_report.py tests/e1hr/test_report.py -q`; require PASS, log/diff/protected-hash review, commit. Preserve the selected 1536-dimensional feature definition for E3, without creating caches.

### E2-T10 — CPU readiness, single-run Slurm wrapper and acceptance preparation

**Depends:** E2-T01–T09. **Files:** wrapper/runbook, `test_entrypoints.py`, `test_gpu_smoke.py`, CPU readiness/log/ledger. **Consumes:** all completed CPU interfaces. **Produces:** reviewable deployment/acceptance harness; no target execution.

- [x] RED: test one declaration only, strict unknown options, check-config without allocation, matching exported seed, own existing last.pt only, and rejection of test/search/LR/batch/view overrides. Wrapper exports `PYTHONHASHSEED`, OMP/MKL=2 and CUBLAS config before its first Python process, then validates and uses `srun` for one E2 invocation.
- [x] Prepare normal/uit/mig2-35/gpu:2g.35gb:1, one task, four CPUs, 16384 MiB, 09:00:00 wrapper and runbook. Explain signal warning/boundary stop versus forced-kill whole-epoch replay; no guarantee an epoch finishes in the warning interval, auto-resubmission or longer walltime.
- [x] Prepare explicit opt-in target acceptance using a separate `e2_target_acceptance_request_v1`/nonofficial evidence root, target-work approval, seed1001/baseLR2.5e-5 and full inherited scheduler horizon. A deterministic three-original-image train-only fixture includes both layouts and a one-image tail; two completed fixture segments allow discard/reconstruct/replay against an uninterrupted reference. These are not official epochs, and their four committed fixture updates do not count toward a run. Capture/restore uses T06's same primitives under a nonofficial fixture envelope; official envelope/count guards are tested separately and never bypassed.
- [x] Without opt-in, GPU tests skip and claim no acceptance. With opt-in, missing target authorization/inputs fails rather than skips. The harness checks actual two-call batching, full-size real-weight finite forward/backward, all-CUDA RNG recovery and memory records; it emits only `approved=false` acceptance/G2 review candidates.
- [x] GREEN and complete CPU readiness checks, interpreted in the actual local environment:

```bash
python -m pytest tests/e2 -q -m "not gpu"
python -m pytest -q -m "not gpu"
python -m compileall -q src/e2 tests/e2
bash -n scripts/slurm/e2_train.sbatch
```

- [x] Require PASS, review source/import closure and protected hashes, record actual CPU environment/limits and skipped GPU scope, then scoped commit. **STOP before E2-T11 until E2-TARGET-AUTH is explicit.**

### E2-T11 — Separately authorized target acceptance

**Depends:** E2-T10 and E2-TARGET-AUTH. **Files:** new per-attempt E2 acceptance evidence, `target_acceptance.md`, unapproved G2 review candidate and four prepared tuning declarations. **Consumes:** reviewed harness/source/contract, actual Slurm allocation and real inputs. **Produces:** target evidence for G2; not approval or tuning.

- [x] Inspect actual source/import closure, effective software/FP32 settings, device/driver/allocation, full train/val audit and preprocessing conformance before model allocation. Capture exact consumed-source map plus separate clean Git/harness/evidence audit inventories. E1/HR/pilot evidence alone is insufficient.
- [x] Only inside the explicitly authorized Slurm allocation, execute the opt-in `tests/e2/test_gpu_smoke.py` harness prepared at T10. Check both raw layouts, global `[2,3,384,896]`, local `[2,4,3,320,320]`, internal local batch8, logits `[2,124]`, tail and shared-encoder gradient paths. Verify the real pinned-weight model has exactly **190,588 classifier parameters and 28,010,716 total E2 parameters** against the design/acceptance oracles; a mismatch fails acceptance and is not a new parameter-count result/default. Record actual memory and timing scope.
- [x] Compare the bounded uninterrupted versus discard/restore/replay fixture exactly for model, optimizer, full-horizon scheduler, Python/NumPy/CPU/all-CUDA RNG, loader/order, selected state and fixture history. Wall-clock/audit fields are excluded. Label bounded state reconstruction correctly; it is not an observed full-natural-epoch or OS-kill target claim and cannot pass official campaign validation.
- [x] Preserve every acceptance attempt/failure. Repair minor operational issues under unchanged controls; renewed target acceptance follows any consumed-source repair. Return for explicit control review only on a material inability to execute the protocol correctly.
- [x] Present CPU/target evidence, the actual scientific/runtime identity and exactly four concrete tuning declarations for E2-G2 review. No self-approved receipt or G3a tuning launch follows a passing smoke.

### E2-T12 — Four full-budget official LR trials and STOP

**Depends:** E2-T11, explicit E2-G2 PASS and E2-G3a authorization. **Files:** E2 tuning declarations/runs/attempt ledgers, locked selection, `tuning_handoff.md`.

- [x] Execute exactly the approved four declarations, one per Slurm attempt invocation and multiple same-run attempts if necessary. Keep 09:00:00 per attempt and all original controls/budgets; retain failures/replay work.
- [x] Require four valid 8-epoch/8-validation/139992-update outcomes and complete parser lineage under the accepted scientific/runtime identity. Publish every selected checkpoint score/epoch and choose the lowest LR eligible under the global maximum rule.
- [x] Lock selection and hand off the full tuning package. **STOP for explicit E2-G3b review.** No selected E2 LR is prefilled here and no final model/job starts automatically.

### E2-T13 — Separately authorized final training and selected validation analysis

**Depends:** complete E2-T12, E2-G3b and still-applicable E2-G2 identity. **Files:** three final declarations/runs/attempt records and selected validation analyses.

- [x] Bind the locked selection and explicitly approved seeds 2001/3001/4001. Initialize each from the pinned ImageNet file with a fresh fused head, optimizer and scheduler; never inherit tuning/E1/HR/pilot state.
- [x] Complete all 8 epochs/8 validations/139992 updates per seed over same-declaration attempts as needed. Preserve best/last, complete histories, environment and attempts/failures, then export selected-checkpoint validation analyses.
- [x] Validate all three full budgets/coverage/selection/identity bindings before closeout. Preserve selected model/feature lineage for a future separately authorized E3 stage; test remains locked.

### E2-T14 — Validation closeout and comparison handoff

**Depends:** E2-T13 complete. **Files:** E2 aggregate/comparison, scoped checksum inventory, `closeout_review.md`, ledger.

- [x] Verify declarations/config/contract/source/runtime/data/preprocessing/weights through full histories/checkpoints, selected analyses, selection and reports. State whether actual binaries were hashed or only supplied checksum references were checked; never invent observations from missing payloads.
- [x] Verify all three seeds, unrounded ddof=1 aggregate, and the **primary architectural E1 / contextual-complementary E1-HR** comparison hierarchy. Require the reviewed official comparator artifacts provided/materialized in the execution environment, with actual paths inventoried and exact SHA-256 matching the two pins above; no branch-path existence is presumed. Ensure E2-versus-E1-HR is not interpreted as controlled evidence for local views or resolution. Verify per-class/bucket/coverage/prediction/confusion/curve exports, attempt/failure/replay accounting and timing scopes. Inventory exact original bytes and retain limitations.
- [x] Present the concrete package for E2-CLOSEOUT review. Close only E2 train/validation on explicit review; E3 and study-wide final evaluation retain separate gates and test remains LOCKED.

## Coverage, preparation checks, and next review

| Spec sections | Plan owners |
|---|---|
| 1–2: scope, source authority, fixed inputs/access | T00b, T01/T02, T07–T09 |
| 3: frozen five-view construction | T02/T03, T10/T11 |
| 4: shared model/fusion/head/execution | T03/T05/T06, T10/T11 |
| 5–6: exact optimization/accounting/scheduler | T01/T03–T07/T11–T13 |
| 7: independent campaign/comparators/STOP | T05/T08/T12/T13 |
| 8: deterministic runtime/RNG/workers | T02/T03/T06/T07/T10/T11 |
| 9–10: own target acceptance, attempts/recovery | T06/T07/T10/T11, T12/T13 |
| 11–12: identity closure, interfaces, isolation | T00b/T01/T06–T10/T14 |
| 13: reports, comparisons, E3-compatible handoff | T04/T09/T13/T14 |
| 14–15: test lock and gate order | All entrypoint/data/report guards and the gate/ledger tables |

T00a/T00b preparation checks were document-only. G1 subsequently froze the exact candidate and authorized native T01-T10. Current CPU results, source closure and protected-input checks are recorded in the implementation log/readiness package. Target acceptance remains a future, separately authorized observation.

## E2-G1 approval record (2026-11-05)

E2-G1 is **PASS**, explicitly approved by the user for exact contract SHA-256 `ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72` at reviewed commit `c40dd00a22de2d3ef50440440107d66db7b7dc45`. The approved JSON bytes remain unchanged; the exact-byte sidecar and [G1 receipt](e2_g1_approval.json) record the freeze. Native task-by-task **E2-T01-T10 only** is authorized, with CPU RED -> GREEN -> scoped regression and commits. Earlier preparation-status text and frozen candidate snapshots are historical. Target/GPU work, target acceptance, tuning, finals, E3 and test images remain **NOT AUTHORIZED**. STOP after T10.
