# E1-HR Implementation Plan

> **For agentic workers:** HR-G1 selects native task-by-task execution using `executing-plans` for HR-T01-HR-T10 only. Keep RED -> GREEN -> regression -> scoped commits, then STOP before HR-T11 and separate target-work authorization. A fresh whole-branch reviewer checks the completed CPU/preparation work.

**Goal:** Implement and verify the resolution-only E1-HR experiment, independently tune its four LRs, and, after separate approval, train/report three final seeds under the unchanged eight-epoch protocol.

**Architecture:** Own HR configuration, data integration, model construction, checkpoints, target identity, orchestration, campaign validation and reports in `src/e1hr/`. Import the frozen preprocessor unchanged and narrowly reuse the unchanged `src.e1.engine` and `src.e1.metrics` numerical mechanics. Keep all E1 validators, source and completed evidence intact; do not parameterize E1 into a general experiment framework.

**Tech Stack:** The inherited Python 3.11.17 / Torch 2.8.0+cu128 / TorchVision 0.23.0+cu128 stack, Pillow, NumPy, pandas, scikit-learn, pytest, JSON/CSV and UIT Slurm. Local CPU fixtures are not target acceptance. No dependency changes are planned.

**Spec:** [specs.md](specs.md), Sections 1–10, with the two user-requested resume/material-failure clarifications. **Recorded 2026-11-02:** the specification is reviewed and HR-G0 is PASS; these design artifacts are approved at HR-G1. HR-G1 is PASS; HR-T01-HR-T10 implementation is authorized; all target/execution stages remain unauthorized.

**Status:** **HR-G1 PASS**. The [numeric contract](e1hr_numeric_contract_v1.json) is frozen at the explicitly approved digest; its preparation snapshot remains unchanged. HR-T01-HR-T10 implementation is authorized. [tasks.md](tasks.md) owns current task/gate statuses. E1 train/validation remains closed.

## Global Constraints

- Experiment `E1-HR`; one global WH `[1344,576]`, input `[N,3,576,1344]`, ConvNeXt-Tiny representation 768 and biased `Linear(768,124)` producing raw logits. No local views, fusion or E3 caching.
- Exact `baseline124-v1` membership: 34997 train, 7497 val; 124 outputs, 106 val-supported classes. Test counts/membership are metadata only. Split seed 406 never regenerates membership.
- Import `Preprocessor(experiment="e1hr", global_wh=(1344,576))` directly on decoded raw images. Preserve both raw layouts, frozen geometry, photometric formulas, float32 fill, tensor bilinear antialias, half-up rounding, letterbox and normalization. Never resize an E1 canvas to HR.
- Fresh pinned ImageNet initialization for every trial/final seed; exact filename/full SHA-256 from the contract; strict offline loading. No E1/pilot/tuning checkpoint warm-start.
- Full fine-tuning from epoch 1; natural shuffled unweighted CE, label_smoothing=0; maximum stochastic depth 0.1, head LR multiplier 1, no LLRD. Train brightness/contrast amplitudes 0.10/0.10; validation augmentation OFF.
- Effective batch 2 original images, microbatch 2, accumulation 1, drop_last=false; 17499 updates/epoch including a one-image tail. Eight epochs, eight full validations, 139992 updates; early stopping OFF.
- AdamW betas `[0.9,0.999]`, eps `1e-8`, clipping OFF; decay 0.05 on Conv/Linear weights, 0 on biases/LayerNorm/layer-scale; every trainable tensor exactly once.
- Full-horizon update warmup/cosine, W=1000, start=0.1, floor=1e-6, N=139992. Checkpoint selection never changes the training horizon.
- LR grid `[2.5e-5,5e-5,1e-4,2e-4]`, tuning seed 1001, final seeds `[2001,3001,4001]`. All four training IDs distinct within HR; 3406/406 forbidden. E1's selected LR is not an HR default.
- Chronological supported-validation Macro-F1 comparator: replace only for improvement >1e-6; earlier ties. Across four complete valid trials, use one global maximum M and choose the lowest LR satisfying M-score<=1e-6. NLL/Top-k/order cannot break ties.
- Exact deterministic FP32/software/RNG/loader policy in the contract. No architecture token in inherited seed derivation; preserve worker bootstrap and logical loader state on resume.
- Each Slurm attempt serves one declaration, with 09:00:00 per attempt. A run may span attempts. Boundary zero plus completed-epoch commits; replay an uncommitted epoch from its beginning from the previous `last.pt`. No extra completed budget, warmup reset, longer walltime or silent fresh replacement.
- HR identity, roots, receipts and evidence are separate. Only a material target-acceptance failure preventing correct execution of the frozen protocol may trigger explicit user review of inherited controls; minor operational failures do not reopen them. No failure authorizes unilateral changes.
- Test images stay LOCKED / NOT AUTHORIZED through HR closeout. No test evaluator. Later study-wide closure includes E3 if retained, followed by separate final-evaluation authorization.

## Review Focus

1. E1 receipts/checkpoints, a relabeled E1 run or a rehashed wrong-resolution declaration must fail HR identity validation before image opening/model allocation: HR-T01/T06/T07/T08.
2. A changed dataset root, symlink escape, missing/corrupt image or stale audit cannot pass on historical pilot conformance alone: HR-T02/T07/T11.
3. A forced termination mid-epoch, persistent-worker restart or crash between last/best/history publication must replay the previous completed boundary without RNG drift or extra budget: HR-T06/T07/T10/T11.
4. Short or failed trials, tolerance chains and mixed source/runtime identities cannot enter LR selection; final launch cannot follow automatically: HR-T08/T12.
5. Reporting from last.pt, rounding before aggregation or dropping cross-bucket false positives must fail selected-checkpoint/report tests: HR-T04/T09/T14.

## Authority, contract freeze and gates

Use the approved HR spec first, then the unchanged E1 numeric contract SHA-256 `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143` for inherited definitions and completed E1 methodology. Historical NEEDS REVIEW fields in older E1/master/pilot text do not reopen HR controls.

| Gate / milestone | Required evidence and permitted next work | Current state |
|---|---|---|
| HR-G0 | Approved HR specification; prepare design artifacts | PASS, explicit user confirmation |
| HR-G1 | Review plan/ledger and complete contract candidate; freeze exact JSON bytes and external `.sha256` sidecar; record approval binding the digest; select implementation method; explicitly authorize HR-T01–T10 implementation | PASS; exact contract frozen, native HR-T01-HR-T10 authorized |
| HR target-work authorization | Explicit permission for bounded HR GPU deployment/acceptance under the reviewed allocation | NOT GRANTED; CPU deployment preparation does not grant it |
| HR target acceptance, HR-T11 | Authorized current-source/current-root/current-stack integrity, real-weight HR forward/backward and boundary/replay evidence; unapproved acceptance candidate only | NOT STARTED |
| HR-G2 | Review CPU/regression and HR-T11 evidence, actual source/runtime/device/allocation identity and four concrete tuning declarations | NOT STARTED; does not authorize tuning |
| HR-G3a | Explicit authorization of only the four declared HR tuning runs after HR-G2 | NOT GRANTED |
| HR-T12 tuning handoff | Four valid full-budget outcomes, locked selection, attempt/failure ledger; mandatory STOP | NOT STARTED |
| HR-G3b | Separate review of that handoff, locked selected LR and three fresh final declarations; explicit final-training authorization | NOT GRANTED |
| HR-T13 final train/validation | Three complete eight-epoch/eight-validation runs and selected-checkpoint exports | NOT STARTED |
| HR-T14 closeout | Validate all lineage, aggregate and E1 comparison; user reviews train/validation closeout | NOT STARTED; never unlocks test |
| Final evaluation | All intended comparison protocols/checkpoints frozen and training/validation closed, including E3 if retained, then separate explicit authorization | LOCKED / NOT AUTHORIZED |

The draft contains no unresolved scientific input. Actual accepted source/runtime observations, approvals, selection and result hashes are later stage evidence, not defaults to invent. At HR-G1 compute the exact-byte digest outside the JSON, review it with the user, and create `e1hr_numeric_contract_v1.sha256` plus `hr_g1_approval.json` only following explicit approval. The draft's preparation-status snapshot may remain historical in frozen bytes; the approval receipt governs authorization. No self-digest is embedded in the JSON. Preserve LF UTF-8 bytes across checkouts and verify them after transfer; if attributes are needed, propose only HR-specific `.gitattributes` entries in HR-T01, without changing E1 rules.

## Planned files and bounded reuse

All files in this table are **future implementation deliverables**, not created by this design task.

| File | Responsibility | Owner |
|---|---|---|
| `src/e1hr/__init__.py`, `config.py` | HR contract freeze/receipt checks, strict immutable declaration/identity schemas | HR-T01 |
| `src/e1hr/data.py` | HR-owned fixed-manifest validation and raw-to-HR dataset; exact inherited streams/loaders | HR-T02 |
| `src/e1hr/model.py` | Offline pinned ConvNeXt-Tiny, isolated head reset, explicit optimizer groups | HR-T03 |
| `src/e1hr/metrics.py` | Bounded exports of unchanged `src.e1.metrics.Metrics124` / `rank_logits` | HR-T04 |
| `src/e1hr/engine.py` | Bounded exports of unchanged E1 update scheduler, train/evaluate/comparator and NumericalFailure | HR-T05 |
| `src/e1hr/checkpoint.py` | HR checkpoint kinds, atomic boundary commit/read/restore and RNG state | HR-T06 |
| `src/e1hr/runtime.py` | Current target inputs, clean source closure, software/allocation/runtime/device inspection and construction | HR-T07 |
| `src/e1hr/train.py` | One declared run, preflight approvals, lock, attempts/signals, history/summary and publication recovery | HR-T07 |
| `src/e1hr/campaign.py` | Four tuning declarations, strict completed-evidence validator, immutable ledger/selection, three final declarations | HR-T08 |
| `src/e1hr/report.py` | Selected-checkpoint exports, three-seed aggregation and official E1 comparison | HR-T09 |
| `tests/e1hr/conftest.py`, `test_config.py`, `test_data.py`, `test_model.py`, `test_metrics.py`, `test_engine.py`, `test_checkpoint.py`, `test_train.py`, `test_recovery.py`, `test_campaign.py`, `test_artifacts.py`, `test_report.py` | Isolated CPU fixtures and behavioral regressions | HR-T01–T09 |
| `tests/e1hr/test_entrypoints.py`, `test_gpu_smoke.py` | CPU CLI/wrapper checks and separately opted-in target acceptance | HR-T10/T11 |
| `scripts/slurm/e1hr_train.sbatch`, `docs/E1-HR/runbook.md` | Reviewed single-declaration attempt and multi-attempt recovery procedure | HR-T10 |
| `docs/E1-HR/implementation_log.md`, `target_acceptance.md`, `tuning_handoff.md`, `closeout_review.md` | Evidence-backed stage records written when their stages occur | Respective stages |

Reuse only the unchanged E1 engine/metrics public mechanics; their dependency closure is the engine, metrics and package marker. HR must not call E1 `load_recipe`, `validate_recipe`, `require_approval`, `resolve_run`, model builder, manifest constructor, runner, checkpoint parser or campaign/report validators. Their frozen contract, shapes and kinds are E1-specific. Adapt needed mechanics into HR-owned modules with HR checks; do not monkeypatch globals, accept E1 receipts as HR, modify E1 validators, or import pilot code at runtime. Keep E1 and existing preprocessing/pilot regression suites unchanged.

Proposed separate target roots, to review with HR-G1: `/datastore/binhvp/BuiTrongTan/CS406/runs/E1-HR` for runs and `/datastore/binhvp/BuiTrongTan/CS406/reports/E1-HR` for acceptance/campaign/approval/report evidence. Retain the inherited dataset/weight paths and `/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python`. No directories or receipts are created on UIT by this plan.

### Scientific/runtime identity and separate audit inventories

The scientific/runtime fingerprint is the canonical hash of an explicit repository-relative path → exact-byte SHA-256 map. Its reviewed consumed-file closure is:

- `src/__init__.py`, `src/e1hr/__init__.py`, `src/e1hr/config.py`, `src/e1hr/data.py`, `src/e1hr/model.py`, `src/e1hr/metrics.py`, `src/e1hr/engine.py`, `src/e1hr/checkpoint.py`, `src/e1hr/runtime.py`, `src/e1hr/train.py`, `src/e1hr/campaign.py`.
- `src/e1/__init__.py`, `src/e1/engine.py`, `src/e1/metrics.py` for normal package imports of reused mechanics.
- `src/preprocessing/__init__.py`, exact `src/preprocessing/geoguessr_v1.py`, frozen `configs/preprocessing/geoguessr_v1.json` and its runtime-read `configs/preprocessing/geoguessr_v1.sha256` sidecar.
- `requirements.txt`, `pyproject.toml`, and `scripts/slurm/e1hr_train.sbatch`, which fixes the launch environment.

Package markers are included because normal imports execute them; do not expand to unrelated preprocessing files or wildcard directories. The HR campaign module is consumed by final-run selection/preflight validation. The reporting-only `src/e1hr/report.py` belongs in the report's audit inventory, not this training closure unless implementation actually imports it on the training path. Check actual transitive imports at implementation/acceptance: missing files fail; any additional consumed repository file must be explicitly added to the reviewed closure before acceptance, never silently omitted or discovered through a broad wildcard. Any consumed-file change requires renewed target acceptance.

Keep a separate acceptance/test evidence inventory with exact hashes for `tests/e1hr/conftest.py`, `tests/__init__.py` and `tests/e1hr/__init__.py` if present, and `tests/e1hr/test_{config,data,model,metrics,engine,checkpoint,train,recovery,campaign,artifacts,report,entrypoints,gpu_smoke}.py`. Each entry records path, bytes, SHA-256 and role. Record any additional acceptance-only helpers and the actual evidence/output files by exact path/hash when they exist. Preserve the inventory used by each acceptance attempt; later test edits do not rewrite earlier evidence. These hashes and any reporting-tool hashes are audit metadata, excluded from the scientific/runtime fingerprint. An unrelated test-only change does not automatically invalidate acceptance; a correction affecting the validity of the acceptance proof requires evidence review.

Separate `identity.scientific_runtime` from `audit`: the former binds the reviewed consumed-source fingerprint/file map, HR contract, software/effective runtime/device/driver, reviewed allocation policy, actual data root/content, frozen preprocessing/weights and accepted conformance/receipt bindings; the latter records each clean Git commit, cleanliness observation, acceptance/test inventory and operational job/allocation identifiers. Record the clean Git commit at acceptance and every official attempt, but compare acceptance/run/checkpoint/campaign identity using `scientific_runtime`, not whole audit dictionaries or repository commit SHA. A docs-only or test-only commit after acceptance must remain eligible when consumed-source hashes and every other scientific/runtime binding are unchanged. Immutable declarations, full run configuration, stage approvals and accepted evidence hashes still require their existing checks; new audit metadata never permits a mismatched runtime, altered input or cross-run resume.

## Shared interfaces and schemas

Use plain serialized dictionaries and E1's canonical fingerprint convention (sorted keys, ensure_ascii=true, allow_nan=false), distinct from SHA-256 of exact file bytes. No generic configuration framework.

- `load_recipe(path: Path = CONTRACT_PATH) -> dict`, `validate_recipe(recipe: dict) -> dict`, `resolve_run(recipe: dict, *, stage: str, lr: float, seed: int, paths: dict) -> dict`, `validate_run(config: dict) -> dict` in HR config. Loading an execution recipe requires the exact frozen digest/sidecar and explicit HR-G1 receipt. Candidate JSON parsing during review confers no execution capability.
- Declarations: `kind=official_e1hr_run_config_v1`, `experiment=E1-HR`, `resolution_wh=[1344,576]`, stage `tuning|final`, seed, LR, original budget, full recipe, contract hash, paths, `scientific_sha256`, `config_sha256`; final adds locked selection exact-byte hash. Run IDs: `e1hr-{stage}-lr{lr:.8f}-seed{seed}`. HR identity/budget/resolution participate in both scientific and full-config validation. Explicit paths: data_root, weights, runs_dir, conformance_report, acceptance_record, authorization_record; final additionally selection. Reject unknown overrides.
- `FixedManifest(artifacts_dir: Path)` exposes fingerprints, labels, n_train, n_train_by_label, `records(split: str)` for train/val only and `verify_images(data_root: Path, splits=("train","val")) -> dict`. `E1HRDataset(manifest, data_root, split, seed, augmentation)` yields `(global_tensor,label_id,relative_path)`; only HR preprocessing is constructed. `make_loaders(manifest, data_root: Path, config: dict) -> dict` returns train/val loaders with explicit generators; `EpochSampler.set_epoch(epoch: int)` and `HRLoader.prepare_resume(completed_epoch: int)` preserve inherited mechanics.
- `build_e1hr(weights_path: Path, expected_sha256: str, initialization: dict) -> tuple[nn.Module,dict]`, `E1HRModel.encode(images) -> [N,768]`, `forward(images) -> [N,124]`; `build_optimizer(model, config: dict) -> tuple[Optimizer,dict]` returns enumerated parameter groups.
- Metrics and engine interfaces match E1: `Metrics124.update(logits,targets,loss_sum)`, `.compute(n_train,labels) -> dict`; `UpdateScheduler(optimizer,schedule,total_updates)` with step/state_dict/load_state_dict; `train_epoch(model,loader,optimizer,scheduler,scaler,*,device,runtime:dict,accumulation:int,clipping:dict) -> dict`, `evaluate(model,loader,*,device,runtime:dict,n_train:dict,labels:list[str],collect_predictions:bool=False) -> dict` and `select_checkpoint(current:dict,selected:dict|None) -> bool`. Runtime calls remain FP32/scaler=None. Tests must establish the reuse boundary rather than treating the adapter names as evidence.
- `save_checkpoint(path: Path, *, model,optimizer,scheduler,scaler,config,progress,identity,loaders) -> None`, `read_checkpoint(path: Path, *, config,identity) -> dict`, `restore_checkpoint(path: Path, *, model,optimizer,scheduler,scaler,config,identity,loaders) -> dict`. HR checkpoint kind `official_e1hr_checkpoint_v1`. Progress: completed_epoch, updates, history, selected, selected_weights. Last stores authoritative full state; best stores the selected weights for reporting. No validation stopping counters.
- `inspect_target_inputs(config: dict) -> dict` in runtime performs current integrity/conformance/software/source/allocation checks before model allocation, without requiring future HR-G2/G3 receipts. `prepare(config: dict, checked: dict) -> dict` applies/verifies runtime and constructs model/optimizer/full scheduler/loaders on the checked target. It is called only through authorized acceptance or official preflight paths; no CLI bypass of approval exists. Record actual driver/device/allocation; don't fix a MIG UUID or assume historical driver acceptance.
- `preflight(config: dict) -> dict` in train adds HR-G2 and stage receipt/identity checks to input inspection; `run(config: dict, run_dir: Path, *, resume: Path | None=None, stop_requested: Callable[[],bool]) -> dict` owns the one-run lifecycle. CLI `python -m src.e1hr.train --config ... [--check-config | --resume .../last.pt]`; configuration-only checks allocate no model/GPU and submit no jobs.
- Receipts require explicit `experiment`, resolution and HR contract binding. HR-G2 binds accepted `identity.scientific_runtime`/device and acceptance evidence digest; the clean acceptance Git commit and harness inventory are retained as separate audit metadata. Official attempts record their current clean commit and audit inventory without requiring commit equality. HR-G3a binds HR-G2 receipt hash plus exactly four run IDs. HR-G3b additionally binds locked selection bytes plus exactly three final IDs. Acceptance writes only `approved=false` candidates. Approvals are authored solely from explicit user decisions.
- `plan_tuning(recipe,paths) -> list[dict]`, `write_declarations(output,declarations) -> None`, `validate_completed_run(run_dir: Path) -> dict`, `select_lr(recipe,trial_records: list[dict]) -> dict`, `write_selection(path,recipe,selection) -> None`, `read_selection(path,recipe) -> dict`, `plan_final(recipe,selection,paths) -> list[dict]`. Score callers supply only run directories; parser verifies evidence. Kinds use `e1hr_..._v1`; neither declaration nor selection submits a job.
- `evaluate_selected(config,run_dir) -> dict`, `aggregate_final(run_dirs,recipe,selection) -> dict`, `compare_e1(hr_report: dict, e1_report_path: Path) -> dict`. Reports bind experiment/resolution/stage/seed/LR/budget/contract/accepted source-runtime-data-preprocessing-weights and declaration/history/checkpoint/selection lineage. Aggregate retains per-seed lineage. Comparison is read-only against the official E1 aggregate, with matching fixed benchmark/metric conventions and exact source file digest.

## Implementation sequence

Each implementation task below begins only after HR-G1 and its dependencies. Write the named behavioral tests, observe an assertion failure attributable to the missing behavior, implement the minimal scoped change, rerun the stated command, inspect the diff/protected hashes, then make a scoped commit through the chosen repository workflow. A missing import alone is not a completed behavioral red/green cycle. Steps are future work; no tests or implementation are claimed here.

### HR-T00 — Design preparation and contract binding candidate

**Files:** this plan, task ledger, draft contract and only the two requested spec wording clarifications. **Status:** COMPLETE; HR-G1 PASS.

- [x] Carry forward the approved scientific values and separate later observations/outputs.
- [x] Specify module boundaries, schemas, independent oracles and gate sequence.
- [x] Check draft JSON, inherited-value equality, source fingerprints, links and documentation scope; record review readiness in the handoff.
- [x] Record explicit HR-G1 approval of exact bytes/sidecar and native task-by-task HR-T01-HR-T10 implementation authorization; see hr_g1_approval.json.

### HR-T01 — Strict HR contract and declarations

**Depends:** HR-G1. **Files:** HR config/package, conftest/test_config; HR-only byte-preservation rules if needed. **Interfaces:** HR recipe/declaration APIs above.

- [x] Add `test_hr_contract_rejects_e1_and_rehashed_wrong_resolution`: E1 kind/hash/receipt and WH `[896,384]` fail even after caller recomputes a hash. Add missing-field/nonfinite/duplicate-key/seed-role/unknown-override/changed-invariant cases. Assert four exact LRs, 2/2/1 batch, 17499/139992 updates, eight epochs and locked test policy.
- [x] Run `python -m pytest tests/e1hr/test_config.py -q`; confirm the intended failures.
- [x] Implement exact frozen-byte/sidecar/HR-G1 validation and immutable explicit HR declarations; final declarations require a persisted locked selection reference. Never import E1 contract authority.
- [x] Rerun the same test file; require all cases PASS, inspect scope and commit. No execution approval is inferred from declaration validity.

### HR-T02 — Exact data and HR preprocessing/loaders

**Depends:** HR-T01. **Files:** HR data/test_data. **Interfaces:** FixedManifest/E1HRDataset/EpochSampler/HRLoader/make_loaders above.

- [x] Add `test_test_access_rejected_before_open`, artifact digest/length/provenance/label-support tests and missing/corrupt/path-escape/decoded-layout failures. Audit only train/val bytes; reject test in any split list before opening anything.
- [x] Add `test_hr_is_built_from_raw_for_both_layouts`: synthetic raw inputs produce only `[3,576,1344]`, directly equal frozen HR preprocessing with identical factors. Independent letterbox oracle: 1536×648 → 1344×567, padding L/T/R/B=0/4/0/5; 1030×689 → 861×576, padding=241/0/242/0. Include a textured input distinguishing direct HR construction from E1-canvas upscaling; test valid-scene jitter, deterministic validation and mean-filled interiors/padding without hard re-masking edges.
- [x] Pin inherited order/jitter against independent SHA-256 seed fixtures: `1001|0|order` → `8021788208606133414`, `1001|0|Aland/synthetic.jpg|jitter` → `1711155315711192502`, `1001|loader|train` → `381993971064482208`, `1001|loader|val` → `5335367272077000490`. The synthetic path is test-only. Assert no experiment token, full permutation/tail `[2,...,2,1]`, fixed val order and epoch propagation to persistent workers. Compare uninterrupted versus fresh-worker loader generator state at a completed boundary.
- [x] Run `python -m pytest tests/e1hr/test_data.py tests/test_preprocessing.py -q` to RED; implement HR-owned integration without frozen-file edits; rerun to PASS and commit.

### HR-T03 — Pinned model and optimizer

**Depends:** HR-T01. **Files:** HR model/test_model. **Interfaces:** model and optimizer APIs above.

- [x] Add exact filename/full-digest/strict-load rejection cases before load; tiny CPU mechanics verify retained features/avgpool/final LayerNorm, 768 features/124 raw logits, full trainability, explicit block-linear 0→0.1 stochastic depth and eval determinism. Isolated CPU head initialization uses the official seed and restores outer RNG.
- [x] Test exhaustive unique optimizer membership, decay 0.05 for Conv/Linear weight and 0 for bias/LayerNorm/all CNBlock layer_scale; unknown/duplicate/omitted categories fail. All groups use candidate LR and the same schedule.
- [x] Run `python -m pytest tests/e1hr/test_model.py -q` to RED; implement the HR builder without E1 load_recipe or downloads; rerun to PASS and commit. Real pinned weights/full HR batches belong to HR-T11.

### HR-T04 — Metrics reuse and independent oracles

**Depends:** HR-T01/T02. **Files:** HR metrics/test_metrics. **Interfaces:** Metrics124/rank_logits above.

- [x] Pin exact-tie logits: all equal ranks `[0,1,2,3,4]`; target 4 misses Top-1/3 and hits Top-5; logits stay unchanged. Rank all 124 outputs including unsupported labels.
- [x] Pin full-split cross-bucket oracle: GT labels `[0,1]`, predictions `[0,0]`, label 0 tail/label 1 head implies class-0 and tail F1=2/3, supported Macro-F1=1/3. Unsupported F1=null; supported zero-TP F1=0; no-support bucket metrics=null; NLL uses sample mean. Verify fixed bucket coverage from metadata.
- [x] Run `python -m pytest tests/e1hr/test_metrics.py -q` to RED; add the narrow reuse exports; rerun to PASS and commit. Do not duplicate metric implementations.

### HR-T05 — Scheduler, tails and comparator reuse

**Depends:** HR-T03/T04. **Files:** HR engine/test_engine. **Interfaces:** E1-compatible mechanics above.

- [x] Pin used-update LR oracle L=1,W=2,N=6,start=.1,f=.01: `[.1,.55,1,.7525,.2575,.01]`; final floor exact. Official endpoints u=0→.1L,1000→L,139991→1e-6; reject update 139992 and changed horizon on restore. Apply LR before optimizer update; increment completion afterward; validation advances neither.
- [x] Test true image-denominator weighting for the one-image tail and a generic short accumulation fixture against direct sample means; official batch remains 2/1. Nonfinite logits/loss/gradients and OOM propagate as attempt failures, never skipped updates.
- [x] Assert chronological tau/earlier: selected .5 retains later .5000005 despite better NLL, replaces at .500002; lower F1 never replaces. Flat/decreasing curves still require full runner budget in HR-T07.
- [x] Run `python -m pytest tests/e1hr/test_engine.py -q` to RED; expose unchanged E1 mechanics through HR engine; rerun to PASS and commit.

### HR-T06 — HR checkpoints and completed-epoch replay

**Depends:** HR-T01/T02/T03/T05. **Files:** HR checkpoint/test_checkpoint. **Interfaces:** save/read/restore above.

- [x] Test `official_e1hr_checkpoint_v1` and explicit experiment/resolution identity; reject E1/pilot kinds, cross-run configs, altered seed/LR/batch/source/runtime/budget and best.pt resume. Read best only for reporting. Boundary zero and epoch e require updates=e×17499 and full history/selection consistency (small CPU fixtures use explicitly nonofficial mechanics counts).
- [x] Compare uninterrupted versus discard/reconstruct/restore/replay with dropout: model/optimizer/full scheduler, inactive scaler, Python/NumPy/CPU RNG, loader generators, selected weights/epoch and scientific history. CUDA RNG is additionally verified at HR-T11. No minibatch boundary is an official training commit.
- [x] Inject a failed atomic last.pt write: previous boundary remains readable. Test incomplete/corrupt state and crash between last/best/history writes; last.pt is authoritative and selected_weights permits rebuilding best without resetting training.
- [x] Run `python -m pytest tests/e1hr/test_checkpoint.py -q` to RED; implement HR-owned identity/state handling; rerun to PASS and commit.

### HR-T07 — Current-target checks and one-run orchestration

**Depends:** HR-T01–T06. **Files:** HR runtime/train/test_train/test_recovery. **Interfaces:** inspection/preparation/preflight/run above.

- [x] Add zero-allocation preflight failures for missing HR receipts, wrong resolution/source/software/allocation/weights/root or stale conformance; current full train/val hash/containment audit must run before model allocation. Validate effective Torch settings, not only requested values. Fingerprint exactly the reviewed scientific/runtime closure above; hash acceptance/test tooling separately and record the clean target Git commit as audit metadata. Any changed consumed source needs renewed target acceptance.
- [x] Add identity-projection regressions: a different clean commit containing only docs/test changes, with identical consumed-source map and runtime bindings, remains eligible for acceptance/run/resume/campaign validation; changing any consumed file, runtime setting or fixed fingerprint fails before allocation and requires renewed target acceptance. A test-harness edit changes only its exact-hash audit inventory. Exercise real parser comparisons so whole-dictionary or commit-SHA equality cannot accidentally reintroduce invalidation; all immutable config/receipt/checkpoint guards remain active.
- [x] Exercise a real eight-epoch tiny CPU runner with flat/decreasing validation: eight train calls, eight full validations, unchanged full synthetic horizon and earlier selected weights; never introduce a production fixture/bypass flag.
- [x] Test interruption before epoch 1, mid-epoch failure, stop after committed epoch, repeated attempts and terminal-publication failure at epoch 8. Replay the entire uncommitted epoch from its start; reconstruct fresh persistent workers without an extra logical loader draw. Reject completed-run extension, changed declarations, wrong last.pt and second writers. Record a failure before boundary zero; retry only the same declaration and retain that history.
- [x] Persist immutable per-attempt request/environment/start-boundary/outcome records plus events. Keep clean Git commit and acceptance/test inventories as audit metadata and job/allocation IDs operational; compare the stable scientific/runtime projection. Explicitly record unknown partial work after abrupt termination. Each environment snapshot survives later retries. Attempt/replay work never increments committed epoch/update totals.
- [x] Run `python -m pytest tests/e1hr/test_train.py tests/e1hr/test_recovery.py -q` to RED; implement the one-run path and signal handling; rerun to PASS and commit. Failed acceptance records cannot become stage receipts.

### HR-T08 — Evidence-validated campaign and mandatory handoff

**Depends:** HR-T01/T07. **Files:** HR campaign/test_campaign/test_artifacts. **Interfaces:** campaign APIs above.

- [x] Test exactly four declarations/one tuning seed and separate fresh final declarations with three distinct seeds. Neither operation submits jobs. Exclusive creation prevents overwriting declarations/ledger/selection.
- [x] Pin global-maximum oracle: scores `[.5000000,.5000006,.5000012,.4900000]` in grid order select `5e-5`, unchanged under any record order. Reject missing/duplicate/extra/failed/short candidates and caller-supplied scores, mixed accepted identities, stale selection bytes and selected metrics from another checkpoint.
- [x] Add temporary synthetic parser-envelope tests for successful full-lineage parsing and rehashed wrong history/selected weights. Verify declaration↔summary↔8-row history↔best/last↔selection consistency, 34997/7497 coverage, 106-supported/124 outputs, exact 139992 updates, HR receipts/fingerprints and operation/attempt inventory. Fixture envelopes are never exported as target evidence.
- [x] Run `python -m pytest tests/e1hr/test_campaign.py tests/e1hr/test_artifacts.py -q` to RED; implement strict HR parsing/selection. Selection records `STOP_FOR_USER_HR_G3b_REVIEW` and `final_execution_authorized=false`; rerun to PASS and commit. Operational interruption resumes the same run; an uncompletable candidate stops for explicit review without scientific fallback.

### HR-T09 — Selected analyses, three-seed aggregate and E1 comparison

**Depends:** HR-T02–T08. **Files:** HR report/test_report. **Interfaces:** reporting APIs above.

- [x] Test a complete run with best earlier than last; every reported metric/prediction must reproduce the same selected checkpoint/history. Write overall/per-class/bucket/predictions/confusion/most-confused/learning-curves exports and a lineage-bearing analysis index last. Reject wrong checkpoint/history/identity; no test-split evaluator exists.
- [x] Require all three complete distinct final seeds at one locked HR selection and accepted identity; compute unrounded arithmetic mean/sample std ddof=1. Independent fractions `[.1,.2,.3]` give mean=.2,std=.1, display `20.0000% +/- 10.0000%`. Null unsupported coverage stays null; no seed cherry-picking or logit ensemble.
- [x] Read official `src/e1/e1_three_seed_validation_report_v1.json` without modifying it; verify its digest, E1 contract, three seeds, ddof and benchmark/metric definitions. Compare overall and fixed buckets; optional paired deltas join matching IDs. Never substitute pilot/best-seed metrics. Label validation-only scope, sampling/geographic/extreme-tail limitations and measured efficiency scopes.
- [x] Run `python -m pytest tests/e1hr/test_report.py -q` to RED; implement reports and comparison; rerun to PASS and commit. Every report envelope carries the HR identity fields, not just a renamed E1 kind.

### HR-T10 — CPU readiness, deployment preparation and runbook

**Depends:** HR-T01–T09. **Files:** future HR wrapper/runbook/test_entrypoints/test_gpu_smoke; no job creation/submission during preparation.

- [x] Write CLI/wrapper tests: one declaration, seed matching, strict unknown options, no test/search/batch/LR overrides, check-config without model/GPU allocation, own existing last.pt only, exact allocation/env before Python startup. Adapt E1 wrapper mechanics without editing E1 wrapper or tests.
- [x] Prepare the HR wrapper for normal/uit/mig2-35/gpu:2g.35gb:1, one task, 4 CPUs, 16384 MiB, 09:00:00. Document USR1 warning/boundary stop and forced-kill replay; a warning near walltime cannot guarantee finishing an epoch. Require review of recorded attempt outcomes before same-declaration resubmission. Do not silently auto-resubmit or request longer time.
- [x] Prepare opt-in HR target acceptance with new output directory per acceptance attempt, real pinned weights, both raw train layouts, actual `[2,3,576,1344]→[2,124]` FP32 path and reconstruction/replay; no production G2 bypass flag. Missing target inputs under explicit opt-in fail rather than skip. Without opt-in, skipped GPU tests are not acceptance.
- [x] Run `python -m pytest tests/e1hr -q -m "not gpu"`, `python -m pytest -q -m "not gpu"`, `python -m compileall -q src/e1hr tests/e1hr` and `bash -n scripts/slurm/e1hr_train.sbatch`; require PASS and unchanged protected hashes. Record local environment/limits, inspect diff and commit. STOP before target work pending its explicit authorization.

### HR-T11 — Separately authorized HR target acceptance

**Depends:** HR-T10 and explicit HR target-work authorization. **Files:** new HR acceptance evidence and target_acceptance.md; never E1 receipts/outputs.

- [ ] Verify the exact reviewed consumed-source file map/fingerprint, actual software/runtime/device/driver/allocation, full current train/val integrity and preprocessing conformance for the current root/packages before model allocation. Capture the clean Git commit and exact acceptance/test file/evidence hashes in the separate audit inventory. Scientific identity equality uses the consumed-source/runtime projection; a later docs/test-only clean commit does not invalidate it. Historical conformance can support frozen mechanics but cannot substitute HR checks, actual HR resolution or current audit.
- [ ] Run the bounded real-weight train-only HR forward/backward path for both raw layouts; capture finite logits/loss/all trainable gradients, input/logit shapes and actual peak allocated/reserved memory. No validation-score optimization or full pilot is needed.
- [ ] Save boundary zero and a clearly labeled test-only mechanics boundary; execute/discard an uncommitted segment, reconstruct model/optimizer/scheduler/workers, restore its own last.pt and replay. Compare model/optimizer/scheduler/scaler-policy/Python/NumPy/CPU/all-CUDA RNG/loader/selection/history progression to the uninterrupted reference. Require exact equality on the accepted deterministic stack; wall-clock measurements are excluded. Retain full eight-epoch scheduler parameters. The bounded fixture is not a completed official epoch and must fail campaign validation; CPU runner tests own full-epoch control flow.
- [ ] Record all acceptance attempts/failures and an unapproved HR-G2 candidate. Distinguish bounded reconstruction proof from an OS-kill/full-natural-epoch target claim. Classify failures: repair minor operational issues under unchanged controls; only a material inability to execute the frozen protocol correctly triggers explicit user review of inherited controls. No automatic fallback.
- [ ] Present CPU/target evidence, proposed current-source identity and four concrete declarations for HR-G2 review. A smoke pass alone approves neither HR-G2 nor HR-G3a.

### HR-T12 — Four official LR trials and STOP

**Depends:** HR-T11, HR-G2 PASS, HR-G3a explicit authorization. **Files:** HR tuning declarations/run artifacts/attempt ledger/locked selection/tuning_handoff.md.

- [ ] Review all four immutable declarations and multi-attempt recovery procedure. Execute each declared run through one-run Slurm attempts, preserving 09:00:00 per attempt and the original budget; never treat a partial prefix as success.
- [ ] Validate all four completed 8/8 trials and histories/checkpoints/attempts under the same accepted identity. Publish each candidate's selected epoch/score and all failures/retries; select LR with the global maximum rule and lock selection lineage.
- [ ] STOP. Hand off the complete tuning package and selected configuration for separate HR-G3b review. No final model is initialized or job submitted automatically. HR selected LR remains unknown until this stage succeeds.

### HR-T13 — Separately authorized final train/validation

**Depends:** HR-T12 complete, HR-G3b explicit authorization, HR-G2 still applicable. **Files:** three final HR declarations/runs and selected validation exports.

- [ ] Review the locked selection and authorize only seeds 2001/3001/4001. Start each independently from the pinned ImageNet file with fresh optimizer/scheduler; never from tuning/E1/pilot weights.
- [ ] Complete all eight epochs/eight validations/139992 updates per seed, across same-declaration attempts as needed. Preserve best/last, full histories, declarations, environment, attempts and failures. Produce selected-checkpoint validation analysis under authorized target execution.
- [ ] Verify full budgets/coverage and per-seed lineage before passing to closeout. No test access or automatic study-wide evaluation.

### HR-T14 — Validation closeout and comparison handoff

**Depends:** HR-T13 complete. **Files:** HR aggregate/comparison, scoped checksum inventory and closeout_review.md.

- [ ] Verify exact declaration/config/contract/source/runtime/data/preprocessing/weights bindings through histories, selected checkpoints, selection and exports. Distinguish actual binary-checkpoint hashing from checksum-reference verification if any evidence is supplied locally; state audit scope without inventing observations.
- [ ] Verify all three seeds, ddof=1, full-precision metrics, per-class/bucket coverage, manifest-keyed predictions/confusion and curves; compare against the official E1 aggregate. Preserve failures/replay timing and distinguish train-only, validation, checkpoint/report I/O and full-attempt scopes.
- [ ] Inventory originals with exact-byte hashes, record limitations and obtain review of HR train/validation closeout. Later experiments retain their own gates; final test remains LOCKED pending study-wide closure (including E3 if retained) and separate authorization.

## Design verification and handoff

Spec coverage: Sections 1–4 → HR-T01–T05/T07; Section 5 → HR-T05/T08/T12/T13; Section 6 → HR-T06/T07/T10/T11; Section 7 → HR-T01/T06–T09/T14; Section 8 → all entrypoint/data/report guards and HR-T14; Section 9 → HR-T04/T09/T14; Section 10 → the gate table and ledger. No planned task changes E1 scientific files/evidence, performs E3 work, adds LR candidates, changes inherited controls or exposes test evaluation.

Preparation verification is limited to document/contract parsing, inherited-value/digest/link checks, independent arithmetic and unchanged protected files. Implementation/CPU outcomes are recorded task-by-task in implementation_log.md; all target outcomes remain unexecuted. HR-G1 has approved the written artifacts and exact contract bytes/sidecar, with native task-by-task HR-T01-HR-T10 implementation authorization. Target work, tuning, final training, closeout and final evaluation retain their separate gates.

## HR-G1 approval record (2026-11-02)

HR-G1 is **PASS**, bound to exact contract SHA-256 `e23fa67e1f41c7bea43446e44961a215ca488aac72ae21f5e56395e097c0a35d` and [hr_g1_approval.json](hr_g1_approval.json). The [checksum sidecar](e1hr_numeric_contract_v1.sha256) freezes the approved bytes. Implementation is authorized for **HR-T01-HR-T10 only**, using native task-by-task RED -> GREEN -> regression -> scoped commits. Contract preparation-status/provenance snapshots remain unchanged. Target/GPU work, HR-T11 acceptance, HR-G2, HR-G3a/tuning, HR-G3b/final seeds and final-test access remain **NOT AUTHORIZED**. Stop after HR-T10 readiness evidence.
