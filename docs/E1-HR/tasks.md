# E1-HR Task Ledger

Updated: **2026-11-02**. **HR-G0 PASS; design preparation complete for HR-G1 review. HR-G1 PASS. HR-T01-HR-T10 implementation authorized; target work and official execution NOT AUTHORIZED. Final test access LOCKED / NOT AUTHORIZED.**

Authority: the approved [E1-HR specification](specs.md), with the user's two minor wording clarifications, and the unchanged [E1 inherited numeric reference](../E1/e1_numeric_contract_v1.json). Technical interfaces, test oracles and future commands are in [plan.md](plan.md). The [HR numeric contract candidate](e1hr_numeric_contract_v1.json) is **frozen at HR-G1**; its preparation snapshot is unchanged. This ledger owns current completion/gate state; historical E1 completion checkboxes do not transfer to HR.

The reviewed specification directly records HR-G0 PASS. The design artifacts are approved at HR-G1; HR-T01-HR-T10 implementation is authorized. Only HR-T01-HR-T10 implementation is authorized; target work, target acceptance, tuning, final training and test evaluation remain unauthorized. The current work changes only design artifacts. E1 train/validation is closed and its evidence stays unchanged.

## Status definitions

| Status | Meaning |
|---|---|
| COMPLETE — DESIGN ONLY | The named preparation artifact exists and its document checks are complete; no implementation or target execution is implied |
| AWAITING REVIEW | Concrete artifacts are ready for an explicit user decision; approval has not been granted |
| NOT STARTED — GATED | Future work has not begun and requires the named gate/dependencies |
| PASS | An explicit user gate decision is recorded, with its scoped binding when applicable |
| LOCKED / NOT AUTHORIZED | No access/execution is permitted under any earlier HR gate |

Future task completion requires actual evidence, not the presence of a plan. A skipped GPU test, pilot record or E1 receipt is not HR acceptance. Failure/interrupt/retry records remain visible and cannot count as full-budget completed runs.

## Approval and review ledger

| ID | State | Decision / evidence needed | Scope after approval |
|---|---|---|---|
| HR-G0 | **PASS** | User confirms approved revised HR spec and requests these artifacts | Design preparation only |
| HR-G1 | **PASS** | Approve plan/ledger and full draft contract; freeze exact JSON bytes and external `.sha256` sidecar; record digest-bound HR-G1 receipt; select implementation method; explicitly authorize implementation | HR-T01–T10 implementation/CPU/deployment preparation, no target jobs |
| HR-TARGET-AUTH | **NOT GRANTED** | Explicit authorization for bounded HR target deployment/acceptance under reviewed Slurm policy | HR-T11 target work only; not tuning |
| HR-TARGET-ACCEPTANCE | **NOT STARTED — GATED** | HR-T11 current-source/root/software/device/allocation evidence, real pinned weights, finite HR forward/backward and interrupted-state reconstruction/replay | Supplies evidence for HR-G2; does not approve it |
| HR-G2 | **NOT STARTED — GATED** | User reviews HR-T01–T11 evidence, actual accepted identity and concrete four tuning declarations; receipt binds experiment/resolution/contract/acceptance evidence | Implementation readiness accepted, no tuning permission by itself |
| HR-G3a | **NOT GRANTED** | Explicit tuning permission after HR-G2; bind HR-G2 receipt and exactly four HR tuning run IDs | HR-T12 four trials only |
| HR-TUNING-HANDOFF | **NOT STARTED — GATED** | Four full-budget valid trials, all attempt/failure/retry records and locked global-maximum LR selection | Mandatory STOP for HR-G3b review |
| HR-G3b | **NOT GRANTED** | Separate review of completed tuning handoff; bind selection bytes, HR-G2 and exactly three final declarations | HR-T13 final seeds 2001/3001/4001 only |
| HR-CLOSEOUT | **NOT STARTED — GATED** | HR-T14 verified final analyses/aggregate/comparison/lineage/inventory and user closeout review | Close HR train/validation; no final-test authorization |
| FINAL-EVALUATION | **LOCKED / NOT AUTHORIZED** | All intended comparison protocols/checkpoints frozen and train/validation closed, including E3 if retained; then separate explicit final-evaluation approval | Outside this plan |

Flow: **HR-G0 → HR-T00 → HR-G1 → HR-T01–T10 → HR-TARGET-AUTH → HR-T11 → HR-G2 → HR-G3a → HR-T12 → STOP → HR-G3b → HR-T13 → HR-T14/HR-CLOSEOUT**. Final evaluation remains a separate study-wide gate. Dependencies do not authorize jobs or request parallel agents.

## Frozen controls checklist

These are settled inputs, not review questions for another hyperparameter search:

- E1-HR global WH 1344×576 directly from frozen raw preprocessing; ConvNeXt-Tiny, 768 representation, Linear(768,124,bias=true), same pinned offline ImageNet checkpoint.
- Exact fixed split/124-label order and frozen preprocessing bytes; no test-image access.
- Natural unweighted CE, full fine-tuning from epoch 1, stochastic depth 0→0.1, jitter 0.10/0.10, same AdamW/init/RNG/FP32/loaders.
- Effective/microbatch/accumulation 2/2/1, retained one-image tail, 17499 updates per epoch, 139992 over eight epochs/eight validations; early stopping OFF.
- W=1000, start=.1, absolute floor=1e-6, exact full-horizon warmup/cosine; no warmup reset or budget extension on resume.
- Independent four-LR grid `[2.5e-5,5e-5,1e-4,2e-4]`, seed 1001; three fresh final seeds `[2001,3001,4001]` only after HR-G3b. HR selected LR is not yet known.
- Checkpoint improvement >1e-6 with earlier ties; LR choice anchored once at global maximum over all four complete selected scores, lowest eligible LR; mean/sample std ddof=1.
- 09:00:00 per attempt, one declaration per attempt, possibly multiple attempts per run; committed boundaries are boundary zero and completed epochs. Replay any uncommitted epoch from its beginning from the prior last.pt.
- Only a material target-acceptance failure preventing correct execution of the frozen protocol may trigger explicit user review of inherited controls. Minor operational failures do not reopen controls; no automatic fallback is authorized.

## Dependency-ordered task ledger

| Task | Depends on | Deliverable | Current status / evidence |
|---|---|---|---|
| HR-T00 | HR-G0 | Two spec wording clarifications; English plan/ledger; complete HR numeric draft | **COMPLETE — DESIGN ONLY**; document/contract checks, no implementation/target evidence |
| HR-T01 | HR-G1 | HR package/config/declaration schemas and isolated CPU fixtures | **NOT STARTED — GATED** |
| HR-T02 | HR-T01 | Exact manifest, direct raw-to-HR preprocessing, order/jitter/loaders | **NOT STARTED — GATED** |
| HR-T03 | HR-T01 | HR pinned model/head/full fine-tuning/optimizer groups | **NOT STARTED — GATED** |
| HR-T04 | HR-T01/T02 | Narrow metrics reuse and independent full-split/bucket oracles | **NOT STARTED — GATED** |
| HR-T05 | HR-T03/T04 | Narrow engine reuse; update scheduler, tail weighting and comparator | **NOT STARTED — GATED** |
| HR-T06 | HR-T01/T02/T03/T05 | HR full-state atomic checkpoints and completed-epoch replay | **NOT STARTED — GATED** |
| HR-T07 | HR-T01–T06 | Current-target preflight/runtime and one-run/attempt orchestration | **NOT STARTED — GATED** |
| HR-T08 | HR-T01/T07 | Four declarations, strict evidence ledger, locked LR selection and fresh final declarations | **NOT STARTED — GATED** |
| HR-T09 | HR-T02–T08 | Selected validation exports, three-seed aggregate and official E1 comparison | **NOT STARTED — GATED** |
| HR-T10 | HR-T01–T09 | CPU regression, single-run wrapper/runbook and opt-in acceptance probe preparation | **NOT STARTED — GATED**; no target work without HR-TARGET-AUTH |
| HR-T11 | HR-T10 + HR-TARGET-AUTH | Actual HR target acceptance evidence and unapproved HR-G2 candidate | **NOT STARTED — GATED**; separate from HR-G2 review |
| HR-T12 | HR-T11 + HR-G2 + HR-G3a | Four full-budget tuning trials, locked selection and mandatory handoff | **NOT STARTED — GATED** |
| HR-T13 | HR-T12 + HR-G3b + applicable HR-G2 | Three full-budget fresh final runs and selected validation exports | **NOT STARTED — GATED** |
| HR-T14 | HR-T13 | Full lineage/aggregate/E1 comparison/inventory review and HR train/validation closeout | **NOT STARTED — GATED** |

All production tasks HR-T01 onward require HR-G1, including tasks whose immediate dependencies list only prior task IDs.

## Scientific/runtime identity versus audit evidence

HR acceptance and official runs compare `identity.scientific_runtime`: the explicit consumed-file map/fingerprint specified in plan.md and the numeric contract, plus the unchanged contract/data/preprocessing/weights and software/effective runtime/device/driver/allocation-policy bindings. The closure enumerates exact HR declaration/data/model/metrics/engine/checkpoint/runtime/train/campaign modules, necessary package markers, reused E1 engine/metrics, exact `src/preprocessing/geoguessr_v1.py`, frozen preprocessing config and its runtime-read checksum sidecar, requirements/pyproject and the official Slurm wrapper. No wildcard or acceptance-only test belongs in that closure. Check actual transitive imports and review any additional consumed file before acceptance; a consumed-file change always requires renewed target acceptance.

Maintain exact path/bytes/SHA-256/role inventories of acceptance tests/helpers/evidence separately, including `tests/e1hr/test_gpu_smoke.py`, and preserve the inventory for each attempt. Record the current clean Git commit/cleanliness and these inventories as `audit` metadata at acceptance and official attempts; reporting-only tooling is likewise audited separately. A docs-only or test-only clean commit after acceptance does not invalidate an unchanged scientific/runtime identity. Commit SHA, acceptance-test inventory and operational job IDs are not scientific equality keys. Test-only changes affecting the validity of an acceptance proof require evidence review; unrelated test edits do not automatically require renewed target acceptance. All input/config/receipt/checkpoint integrity checks remain mandatory.

## Observable acceptance per task

### HR-T00 — Design preparation

- [x] Apply only the completed-epoch replay and material-acceptance-failure wording improvements to the spec.
- [x] Prepare English implementation plan, explicit-status ledger and complete numeric draft; no production code, GPU job or test evaluator.
- [x] Verify JSON structure, exact inherited numeric controls/fingerprints, referenced paths and independent arithmetic; verify protected E1/preprocessing/split/pilot bytes remain unchanged.
- [x] HR-G1 user approval/freeze, sidecar/receipt and native task-by-task implementation method recorded.

### HR-T01 — Contract and declaration guards

- [ ] RED→GREEN tests for exact HR bytes/sidecar/receipt, schema/identity/resolution, role/grid/budget and unknown/altered controls; E1/pilot receipts/configs rejected even with recomputed caller hashes.
- [ ] Implement and verify allocation-free HR recipe/declaration APIs; final declarations bind locked selection bytes. Scope-reviewed commit after checks.

### HR-T02 — Data/preprocessing/RNG/loaders

- [ ] RED→GREEN artifact/hash/metadata/path/corruption/test-access guards; test rejection before opening images.
- [ ] Both raw layouts produce direct frozen HR tensors with independent padding oracle, stable valid-scene jitter and deterministic validation; no locals or E1-canvas upscaling.
- [ ] Natural epoch coverage/one-image tail/fixed val order, exact seed derivation and persistent-worker restart equivalence; frozen regression and unchanged-input checks pass.

### HR-T03 — Model/optimizer

- [ ] RED→GREEN strict offline weight verification/loading, retained representation, isolated official-seed head initialization, full trainability/drop-path/eval checks.
- [ ] Every trainable parameter appears once with required decay/no-decay and same LR; unknown groups fail. CPU mechanics are not real-weight target acceptance.

### HR-T04 — Metrics

- [ ] RED→GREEN unchanged mechanics reuse tests: 124-output ties, supported-only Macro-F1, GT-weighted F1, sample-mean NLL and null unsupported/no-support buckets.
- [ ] Independent cross-bucket false-positive oracle and fixed train-frequency coverage pass; no duplicate metric implementation.

### HR-T05 — Engine

- [ ] RED→GREEN independent scheduler interior/endpoints/exhaustion/restore and sample-weighted tail tests; validate-before-step/update-after-step convention.
- [ ] Tau/earlier ignores NLL; validation never advances scheduler; nonfinite/OOM outcomes remain failures with no skipped updates or scientific fallback.

### HR-T06 — Checkpoints

- [ ] RED→GREEN HR kind/config/identity/budget/role guards; reject E1/pilot and best.pt resume.
- [ ] Boundary-zero and completed-epoch CPU reconstruction/replay preserve all scientific/RNG/loader/selection state; atomic-write failures retain previous last.pt.
- [ ] Authoritative last.pt repairs derived best/history after publication interruption; no extra budget or stopping counters.

### HR-T07 — Runtime/runner/recovery

- [ ] RED→GREEN current-root integrity, conformance/consumed-source/software/allocation/runtime/receipt failures before model allocation; refresh train/val hashes and containment. Verify the explicit runtime closure and separate exact-hash acceptance/test inventory.
- [ ] Docs/test-only clean-commit changes with identical consumed hashes/runtime remain eligible across acceptance/run/resume/campaign checks; any consumed-file or runtime-binding change fails and requires renewed target acceptance. Test-harness edits change audit inventory without changing scientific identity. Do not compare whole audit records or commit SHA as scientific identity.
- [ ] Eight-epoch flat/decreasing-curve integration and complete selected-state lineage; no production fixture bypass.
- [ ] Same-declaration boundary-zero/mid-epoch/epoch-eight publication recovery and single writer; reconstruct persistent workers without logical loader RNG drift.
- [ ] Immutable per-attempt environments/boundaries/outcomes and retained failure/replay/unknown-partial-work records; preserve clean commit and tooling inventories as audit metadata separate from scientific equality. Completed totals never include replayed work.

### HR-T08 — Campaign

- [ ] RED→GREEN exact four tuning/three final declaration counts and independent fresh initialization; no submission/automatic stage launch.
- [ ] Global-maximum tolerance oracle and order invariance; reject failed/short/missing/extra/duplicate/mixed-identity trials and stale/rehashed inconsistent evidence.
- [ ] Full parser lineage and attempt records; locked selection requires all four valid 8/8 outcomes and mandatory STOP for HR-G3b.

### HR-T09 — Reports/comparison

- [ ] RED→GREEN earlier-best selected exports, complete validation coverage, lineage and full-split class/bucket/prediction/confusion/curve consistency.
- [ ] Exactly three complete final seeds, shared locked selection/accepted identity, unrounded mean/sample std ddof=1, four-decimal percentages; no ensemble/cherry-pick.
- [ ] Read-only official E1 aggregate comparison, matching protocol/metric definitions and digest; retain validation-only and uncertainty limitations. No pilot comparator or test evaluator.

### HR-T10 — Deployment preparation and CPU readiness

- [ ] CLI/wrapper CPU guards pass; one immutable declaration and own last.pt only; exact allocation and pre-start environment variables.
- [ ] Prepare HR-only wrapper/runbook and opt-in acceptance probe; no jobs created/submitted. State nine-hour attempt limit, full-epoch replay and explicit retry review.
- [ ] HR and whole-repository CPU regressions, compile and shell syntax checks pass; protected E1/frozen/pilot bytes unchanged. Record evidence and STOP before target work until HR-TARGET-AUTH.

### HR-T11 — Actual target acceptance

- [ ] Explicit HR-TARGET-AUTH recorded before GPU deployment/acceptance.
- [ ] Exact reviewed consumed-source fingerprint and stack/root/data/weights/preprocessing/allocation/device/driver verified; full train/val audit and HR conformance before model allocation. Clean Git commit and exact acceptance/test hashes are captured as separate audit metadata; only scientific/runtime bindings determine identity equality across accepted runs.
- [ ] Real train-only batch `[2,3,576,1344]`, logits `[2,124]`, finite forward/backward/all trainable gradients and memory records.
- [ ] Bounded discard/reconstruct/restore/replay equivalence on actual deterministic stack; record fixture/boundary scope, all RNG/loaders and full horizon. No official epoch/optimization result claim.
- [ ] Retain failures/attempts and emit only approved=false HR-G2 candidate; hand off acceptance plus CPU checks/concrete declarations for separate HR-G2 review.

### HR-T12 — Authorized tuning only

- [ ] HR-G2 PASS and separate HR-G3a authorization bind the four reviewed tuning declarations.
- [ ] Exactly four valid eight-epoch/eight-validation/139992-update trials at seed 1001; same-declaration multi-attempt resumes, retained failures/retries, no dropped/replacement LR.
- [ ] Lock HR LR using the frozen rule and selected-checkpoint lineage. **STOP and hand off for HR-G3b**; no final execution is authorized by tuning completion.

### HR-T13 — Authorized final train/validation

- [ ] Separate HR-G3b binds the completed tuning handoff, locked selection and final seed declarations.
- [ ] Seeds 2001/3001/4001 independently start from pinned ImageNet and complete original 8/8 budgets, preserving every attempt/checkpoint/history.
- [ ] Selected-checkpoint validation exports and full per-seed lineage verified before closeout; no test images opened.

### HR-T14 — Closeout

- [ ] Exact-byte inventory and declaration↔config↔history↔checkpoint↔selection↔analysis↔aggregate consistency; clearly state actual binary-hash versus supplied-reference audit scope.
- [ ] Three-seed mean/sample std, per-class/frequency buckets, predictions/confusion/curves, official E1 comparison and efficiency/failure/replay scopes verified.
- [ ] User reviews HR train/validation closeout and later-stage handoff; final test remains locked through all intended study comparison closures, including E3 if retained, and separate approval.

## Current handoff

**HR-G1 PASS; HR-T01-HR-T10 implementation authorized.** No frozen scientific field remains unresolved in the draft. Actual accepted source/runtime/device observations, HR selected LR, attempt/job IDs, checkpoint/result hashes and measured metrics must be supplied by their later stages; they are not missing scientific defaults. The candidate contains policies/schema requirements for them without fabricated values. The exact-byte sidecar and digest-bound approval receipt now record the explicit HR-G1 freeze.

No HR implementation, acceptance, trial/final run, closeout or test evaluation is complete. E1 and pilot evidence cannot satisfy those statuses. Next action is native task-by-task HR-T01-HR-T10 implementation, followed by a mandatory stop before separate target-work authorization.

## HR-G1 approval record (2026-11-02)

HR-G1 is **PASS**, bound to exact contract SHA-256 `e23fa67e1f41c7bea43446e44961a215ca488aac72ae21f5e56395e097c0a35d` and [hr_g1_approval.json](hr_g1_approval.json). The [checksum sidecar](e1hr_numeric_contract_v1.sha256) freezes the approved bytes. Implementation is authorized for **HR-T01-HR-T10 only**, using native task-by-task RED -> GREEN -> regression -> scoped commits. Contract preparation-status/provenance snapshots remain unchanged. Target/GPU work, HR-T11 acceptance, HR-G2, HR-G3a/tuning, HR-G3b/final seeds and final-test access remain **NOT AUTHORIZED**. Stop after HR-T10 readiness evidence.
