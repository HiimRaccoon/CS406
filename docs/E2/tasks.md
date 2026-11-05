# E2 Task Ledger

Updated: **2026-11-05**. **Draft spec/plan/ledger preparation is authorized and complete as documentation. E2-G0 AWAITING REVIEW; E2-G1 NOT APPROVED. E2 implementation, target work, tuning and finals are NOT AUTHORIZED. Final test is LOCKED / NOT AUTHORIZED.**

Authority: [specs.md](specs.md) defines the scientific requirements; [plan.md](plan.md) defines file ownership, interfaces, numerical oracles, commands and approval stops. The user requested these draft planning artifacts; E2-G0 remains awaiting review. That preparation permission does not approve the spec, select an execution method, freeze a contract or inherit any E1/HR task status.

The E2 numeric-contract candidate has **not been created**. Its later preparation is E2-T00b; its exact digest, checksum sidecar and G1 receipt remain future artifacts. This ledger records current state and must be updated from observed task results and explicit user gate decisions.

## Status definitions

| Status | Meaning |
|---|---|
| COMPLETE — DRAFT DOCUMENTATION ONLY | Requested documents exist and preparation checks are complete; no scientific approval or implementation/execution is implied |
| AWAITING REVIEW | Concrete named artifacts await an explicit user decision |
| NOT STARTED — GATED | Work has not begun; all named dependencies/permissions must be satisfied |
| NOT GRANTED / NOT APPROVED | The corresponding scoped authorization/review decision has not been supplied |
| PASS | An explicit user gate decision has been recorded with its applicable contract/evidence/declaration bindings |
| LOCKED / NOT AUTHORIZED | No image access/evaluation is permitted by earlier E2 gates |

Later task completion needs actual results with scope, commands and evidence paths, not a checked plan step or copied E1/HR pass count. GPU skips, CPU fixtures and archived pilots are not E2 target acceptance. Synthetic fixtures never count as official runs. All attempts/failures/retries stay visible.

## Approval and handoff ledger

| ID | Current state | Required review / evidence | Scope after explicit approval |
|---|---|---|---|
| E2-G0 | **AWAITING REVIEW — NOT APPROVED** | Review spec and draft plan/ledger; confirm five-view execution, inherited controls and isolated design; authorize numeric-candidate preparation | Remaining design preparation only |
| E2-G1 | **NOT APPROVED** | Review completed T00b candidate plus plan/ledger, approve exact JSON digest, create exact-byte sidecar/digest-bound receipt, choose execution method and name implementation scope | Prospective T01–T10 implementation/CPU/deployment preparation only |
| E2-TARGET-AUTH | **NOT GRANTED** | Explicit E2 Slurm/GPU permission for bounded target acceptance under reviewed policy | T11 target acceptance only; no tuning |
| E2-TARGET-ACCEPTANCE | **NOT STARTED — GATED** | T11 actual-source/runtime/device/root/weights, five-view finite forward/backward, recovery and immutable evidence inventories | Evidence for G2, not a review decision |
| E2-G2 | **NOT APPROVED** | Review T01–T11 CPU/target evidence, accepted scientific/runtime identity and four concrete tuning declarations | Readiness accepted; no tuning by itself |
| E2-G3a | **NOT GRANTED** | Bind G2 receipt/accepted identity and exactly four tuning declaration digests/run IDs | T12 four LR trials only |
| E2-TUNING-HANDOFF | **NOT STARTED — GATED** | Four full-budget valid outcomes, full attempt ledger and locked global-maximum LR selection | Mandatory STOP for G3b review |
| E2-G3b | **NOT GRANTED** | Review complete tuning handoff; bind locked selection bytes, applicable G2 and exactly three final declarations | T13 final seeds 2001/3001/4001 only |
| E2-CLOSEOUT | **NOT STARTED — GATED** | T14 three-seed analyses/aggregate/both comparator reports, lineage/checksum inventory and user review | E2 train/validation closeout and handoff; no E3 execution/test permission |
| FINAL-EVALUATION | **LOCKED / NOT AUTHORIZED** | All intended comparison protocols/checkpoints frozen and train/validation closed, including E3 if retained, then separate explicit user approval | Outside this plan |

Flow: **E2-G0 → E2-T00b → E2-G1 → E2-T01–T10 → E2-TARGET-AUTH → E2-T11/E2-TARGET-ACCEPTANCE → E2-G2 → E2-G3a → E2-T12/STOP → E2-G3b → E2-T13 → E2-T14/E2-CLOSEOUT**. Current E2-T00a draft preparation is the user-authorized precursor to that flow. A dependency arrow neither launches jobs nor authorizes parallel agents.

## Scientific requirements carried by every task

- Global 896×384 + four local 320×320, frozen rectangles/order and shared raw photometric pair; one shared ConvNeXt-Tiny; per-view final LayerNorm; concatenate global with mean locals to 1536; fused Linear(1536,124,bias=True).
- Global encoder call before one batch-major B*4 local encoder call; at full batch, global B=2/internal locals=8/logits B=2; retain one-image tail.
- Exact fixed split/labels/preprocessing/weights; fresh ImageNet initialization per tuning/final run. E2 head initialization uses fan-in 1536 with isolated seed-bound CPU RNG.
- Natural fused CE, full fine-tuning, reviewed AdamW/decay/FP32/RNG/loader controls; effective/microbatch/accumulation 2/2/1; 17499 updates/epoch; 8 epochs/8 validations/139992 updates; no early stopping.
- Full eight-epoch scheduler: W=1000, start=.1, floor=1e-6, exact formula in spec; no warmup reset/horizon extension. Original-image loss denominator includes all tails.
- Independent grid `[2.5e-5,5e-5,1e-4,2e-4]` × seed1001, then fresh seeds `[2001,3001,4001]` only after G3b. Selected E2 LR and outcomes are unknown.
- Chronological checkpoint improvement >1e-6/earlier; LR anchored to one global maximum over four complete selected scores, lowest eligible LR; mean/sample std ddof=1 before display rounding.
- One declaration per 09:00:00 attempt, possibly several attempts per run; boundary zero and completed-epoch commits after validation; replay any uncommitted epoch from its beginning via same-run last.pt.
- Scientific/runtime consumed-source fingerprint is separate from acceptance/test/reporting audit inventory and clean Git commit metadata. Docs/test-only clean commits preserve eligibility when all scientific bindings match; consumed-source changes require renewed target acceptance.
- Only a material target-acceptance inability to execute the protocol correctly may trigger explicit user review of inherited controls. Minor operational failures do not reopen controls; no automatic fallback.
- E1/HR scientific artifacts and frozen preprocessing stay protected. Final test remains locked; E3 work and final evaluation are separate stages.

## Dependency-ordered tasks

| Task | Depends on | Deliverable | Current status / evidence |
|---|---|---|---|
| E2-T00a | User's draft-document requests | English spec, implementation plan, task ledger; preparation-status synchronization only | **COMPLETE — DRAFT DOCUMENTATION ONLY**; document/reference/arithmetic checks; no approval or implementation |
| E2-T00b | Explicit E2-G0 and remaining-design permission | Complete numeric-contract candidate and exact candidate digest for G1 review | **NOT STARTED — GATED**; no candidate/sidecar/receipt exists |
| E2-T01 | E2-G1 | E2 package/config/declaration identities and isolated CPU fixtures | **NOT STARTED — GATED** |
| E2-T02 | E2-T01 | Exact manifest, shared-jitter five-view dataset, deterministic loaders and dual transfer | **NOT STARTED — GATED** |
| E2-T03 | E2-T01/T02 | Strict pinned shared model, global/local execution/fusion, head fan-in 1536 and decay groups | **NOT STARTED — GATED** |
| E2-T04 | E2-T01/T02 | Narrow unchanged metrics reuse and independent ranking/bucket oracles | **NOT STARTED — GATED** |
| E2-T05 | E2-T02/T03/T04 | E2 multi-view train/evaluate, image-denominator accounting and scheduler/comparator reuse | **NOT STARTED — GATED** |
| E2-T06 | E2-T01/T02/T03/T05 | State primitives, strict atomic E2 checkpoints and completed-boundary replay | **NOT STARTED — GATED** |
| E2-T07 | E2-T01–T06 | Current-target inspection/preflight, deterministic one-run/attempt lifecycle | **NOT STARTED — GATED**; final preflight remains fail-closed until T08 selection integration |
| E2-T08 | E2-T01/T07 | Strict campaign/lineage parser, declarations, locked LR selection, final preflight integration | **NOT STARTED — GATED** |
| E2-T09 | E2-T02–T08 | Selected validation exports, ddof=1 aggregate and read-only E1/HR comparisons | **NOT STARTED — GATED** |
| E2-T10 | E2-T01–T09 | Complete CPU regression/readiness, single-run wrapper/runbook and opt-in acceptance preparation | **NOT STARTED — GATED**; STOP before target work |
| E2-T11 | E2-T10 + E2-TARGET-AUTH | Real E2 target acceptance, exact evidence inventories and unapproved G2 candidate | **NOT STARTED — GATED**; separate from G2 review |
| E2-T12 | E2-T11 + E2-G2 PASS + E2-G3a | Four complete official tuning trials, locked selection and mandatory handoff | **NOT STARTED — GATED** |
| E2-T13 | E2-T12 + E2-G3b + applicable E2-G2 | Three fresh final full-budget runs and selected validation exports | **NOT STARTED — GATED** |
| E2-T14 | E2-T13 | Full lineage/inventory/aggregate/both-comparator review and E2-CLOSEOUT handoff | **NOT STARTED — GATED** |

Strict dependency order is the default; tasks sharing an earlier dependency are not permission to fan out or start implementation now. Exact interfaces, test names/oracles, future commands and scoped commit requirements are in the corresponding plan task.

## Completion criteria and evidence to record

### Design preparation and freeze

- [x] E2-T00a: requested English documents exist; no G0/G1 PASS or fabricated E2 outcome is recorded. Spec changes only acknowledge draft preparation and link the two artifacts.
- [x] Document-only checks cover links/IDs/dependencies, exact common values, fusion/batching/head contract, approval separation, source-identity policy and preservation of completed E1/HR/input evidence.
- [ ] E2-G0: explicit review decision and scope recorded.
- [ ] E2-T00b: complete JSON candidate, exact fingerprints/provenance, arithmetic/oracle validation and exact candidate digest presented; scientific fields resolved without invented stage outputs.
- [ ] E2-G1: exact approved bytes verified, sidecar and digest-bound receipt created, method and named implementation scope recorded; preparation snapshots retained without mutating frozen contract bytes later.

### E2-T01–T10 implementation and CPU/deployment preparation

- [ ] Each task logs behavioral RED, GREEN, relevant unchanged regressions, source/protected-hash review and scoped commit. Use actual outcomes; imports alone and planned assertions are insufficient evidence.
- [ ] T02/T03/T05 prove shared raw jitter, both layouts, global/local transfer, B versus B*4 ordering, exact fusion, single shared parameter set, gradients and original-image tail/loss accounting.
- [ ] T06/T07 prove full state/worker replay, atomic failure preservation, same-run identity, complete eight-epoch control flow without early stopping, terminal-publication recovery without more training, and immutable attempt/failure records.
- [ ] T07/T08 prove scientific/runtime projection equality and docs/test-only commit eligibility; changed consumed source/runtime/inputs or immutable config/receipts/evidence fail before allocation.
- [ ] T08/T09 reject incomplete/rehashed wrong lineage, synthetic fixtures, missing candidates/seeds or stale selected metrics; prove both tolerance rules, STOP, unrounded ddof=1 and actual E1/HR comparator schemas.
- [ ] T10 records complete CPU regression/compile/shell checks, local environment/limits, actual consumed import closure, and the prepared full-size real-weight opt-in harness. GPU skips are reported as unexecuted.
- [ ] STOP after T10; present concrete CPU/deployment readiness for separate E2-TARGET-AUTH.

### Target, tuning, finals, and closeout

- [ ] E2-TARGET-AUTH recorded before any E2 target/GPU work.
- [ ] T11 records actual root/source/runtime/device/allocation/weights; full train/val integrity; real train-only full-size five-view path/tail; finite gradients; bounded state reconstruction with CUDA RNG; exact consumed-source and separate harness/evidence inventories. Fixture segments receive no official epoch/update credit.
- [ ] Preserve all acceptance attempts, label proof scope accurately, and present an unapproved G2 candidate plus four concrete tuning declarations.
- [ ] E2-G2 PASS recorded by user, then E2-G3a separately authorizes only those four trials.
- [ ] T12 validates four complete eight-epoch/eight-validation outcomes, all attempt lineage, and global-maximum LR selection. **STOP** for user review; no automatic final launch.
- [ ] E2-G3b binds selection and exactly three final declarations before T13.
- [ ] T13 completes seeds 2001/3001/4001, each fresh from pinned ImageNet, with all 139992 updates/eight validations and selected validation exports.
- [ ] T14 verifies actual supplied scope, checkpoint/history/selection/report lineage, unrounded aggregate, both comparator hashes, fixed buckets/predictions/confusion/curves, timing/replay records and checksum inventory; user reviews E2-CLOSEOUT.
- [ ] Preserve selected fused-feature/checkpoint definitions for E3 handoff without starting E3. Final test stays LOCKED pending the separate study-wide gate.

## Evidence log convention and current handoff

Future task entries record task ID, changed files, actual command and result, environment, fixture/target scope, evidence paths/hashes, commit, remaining dependencies and stop condition. Failures/retries are appended, not erased. Approval records bind the exact reviewed digest/declarations/selection appropriate to their gate; the ledger does not create permission by itself.

**Current handoff:** E2 spec plus draft plan/task ledger are ready for **E2-G0 review**. The numeric-contract candidate remains E2-T00b and prevents claiming a complete E2-G1 freeze package. Implementation method/scope, E2-G1, target work, G2/G3a/G3b, official results and closeout remain pending. No CPU/GPU implementation run, Slurm job, or test-image access occurred during draft preparation.
