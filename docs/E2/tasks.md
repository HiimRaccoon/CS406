# E2 Task Ledger

Updated: **2026-11-05**. **E2-G0 PASS — explicit user approval of the corrected specification/plan/ledger; E2-T00b only authorized and COMPLETE as candidate/documentation preparation. E2-G1 PASS; native T01?T10 implementation/CPU preparation authorized. Target/GPU tests, target work, official execution, tuning, finals and E3 remain NOT AUTHORIZED. Final test is LOCKED / NOT AUTHORIZED.**

Authority: [specs.md](specs.md) defines the scientific requirements; [plan.md](plan.md) defines file ownership, interfaces, numerical oracles, commands and approval stops. The user explicitly approved their current corrected form at E2-G0 and authorized E2-T00b only on 2026-11-05. This decision selects no implementation method, freezes no contract, grants no implementation/execution permission and inherits no E1/HR task status.

The complete [E2 numeric-contract candidate](e2_numeric_contract_v1.json) is **86,335 bytes**, exact SHA-256 **`ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72`**. E2-T00b preparation and document-only checks are complete. The approved `.sha256` sidecar and `e2_g1_approval.json` have **not** been created. The candidate is not frozen; E2-G1 remains NOT APPROVED.

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
| E2-G0 | **PASS — explicit user decision on 2026-11-05** | User approved the current corrected spec/plan/ledger and authorized E2-T00b only; exact reviewed document snapshot hashes are recorded in candidate provenance | E2-T00b candidate preparation only, now complete |
| E2-G1 | **NOT APPROVED** | Review completed T00b candidate plus plan/ledger, approve exact JSON digest, create exact-byte sidecar/digest-bound receipt, choose execution method and name implementation scope | Prospective T01–T10 implementation/CPU/deployment preparation only |
| E2-TARGET-AUTH | **NOT GRANTED** | Explicit E2 Slurm/GPU permission for bounded target acceptance under reviewed policy | T11 target acceptance only; no tuning |
| E2-TARGET-ACCEPTANCE | **NOT STARTED — GATED** | T11 actual-source/runtime/device/root/weights, five-view finite forward/backward, recovery and immutable evidence inventories | Evidence for G2, not a review decision |
| E2-G2 | **NOT APPROVED** | Review T01–T11 CPU/target evidence, accepted scientific/runtime identity and four concrete tuning declarations | Readiness accepted; no tuning by itself |
| E2-G3a | **NOT GRANTED** | Bind G2 receipt/accepted identity and exactly four tuning declaration digests/run IDs | T12 four LR trials only |
| E2-TUNING-HANDOFF | **NOT STARTED — GATED** | Four full-budget valid outcomes, full attempt ledger and locked global-maximum LR selection | Mandatory STOP for G3b review |
| E2-G3b | **NOT GRANTED** | Review complete tuning handoff; bind locked selection bytes, applicable G2 and exactly three final declarations | T13 final seeds 2001/3001/4001 only |
| E2-CLOSEOUT | **NOT STARTED — GATED** | T14 three-seed analyses/aggregate, hash-bound primary E1 and contextual/complementary E1-HR comparisons, lineage/checksum inventory and user review | E2 train/validation closeout and handoff; no E3 execution/test permission |
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

### Required comparator bindings and parameter-count oracles

**E1 is the primary architectural comparator** because its global input is also 896×384. **E1-HR is a contextual/complementary comparator**. T09 reports and T14 closeout must retain this hierarchy; E2-versus-E1-HR does not provide controlled evidence isolating local views or resolution.

| Reviewed official aggregate | Required exact SHA-256 |
|---|---|
| E1 — primary architectural comparator | `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e` |
| E1-HR — contextual/complementary comparator | `9cb44e8aa98d18065402ea3be750586187aeb4b0e4ff23cfa6badfcc484426cf` |

The reviewed official artifacts must be materialized/provided in the execution environment with explicit actual paths and verified bytes. No repository-relative report existence, additional path, or unsupported hash is presumed. Missing artifacts or hash mismatches fail comparison/closeout; reporting provenance records the supplied paths and exact hashes.

T00b must serialize both comparator roles/hashes as required scientific/reporting bindings and the design/acceptance oracles **190,588 classifier parameters** for `Linear(1536,124,bias=True)` and **28,010,716 total E2 parameters**. T03 checks the actual specified architecture; T11 verifies the real pinned-weight target model. These are required design oracles, not official future observations/results, and must not be changed to accommodate a mismatched implementation.

## Dependency-ordered tasks

| Task | Depends on | Deliverable | Current status / evidence |
|---|---|---|---|
| E2-T00a | User's draft-document requests | English spec, implementation plan, task ledger; preparation-status synchronization only | **COMPLETE — DRAFT DOCUMENTATION ONLY**; document/reference/arithmetic checks; no approval or implementation |
| E2-T00b | Explicit E2-G0 and remaining-design permission | Complete numeric-contract candidate, required comparator roles/exact hashes, parameter-count oracles and exact candidate digest for G1 review | **COMPLETE — CANDIDATE/DOCUMENTATION ONLY**; candidate/digest above; JSON/inheritance/hash/arithmetic checks passed; no sidecar/receipt or execution |
| E2-T01 | E2-G1 | E2 package/config/declaration identities and isolated CPU fixtures | **COMPLETE — CPU/PREPARATION ONLY**; 12 behavioral RED failures; 42 GREEN config/regression cases passed |
| E2-T02 | E2-T01 | Exact manifest, shared-jitter five-view dataset, deterministic loaders and dual transfer | **COMPLETE — CPU/PREPARATION ONLY**; 6 RED failures; 27 GREEN five-view/preprocessing/E1/HR data cases passed |
| E2-T03 | E2-T01/T02 | Strict pinned shared model, global/local execution/fusion, head fan-in 1536, exact 190,588 classifier / 28,010,716 total parameter oracles and decay groups | **COMPLETE — CPU/PREPARATION ONLY**; 5 behavioral RED failures; 16 GREEN model/E1/HR regression cases passed; exact architecture counts verified |
| E2-T04 | E2-T01/T02 | Narrow unchanged metrics reuse and independent ranking/bucket oracles | **COMPLETE — CPU/PREPARATION ONLY**; ranking behavioral RED failed; 11 GREEN metric/E1/HR cases passed |
| E2-T05 | E2-T02/T03/T04 | E2 multi-view train/evaluate, image-denominator accounting and scheduler/comparator reuse | **COMPLETE — CPU/PREPARATION ONLY**; dual-view loop RED failed; 17 GREEN engine/E1/HR cases passed; original-image tail matches direct updates |
| E2-T06 | E2-T01/T02/T03/T05 | State primitives, strict atomic E2 checkpoints and completed-boundary replay | **NOT STARTED — GATED** |
| E2-T07 | E2-T01–T06 | Current-target inspection/preflight, deterministic one-run/attempt lifecycle | **NOT STARTED — GATED**; final preflight remains fail-closed until T08 selection integration |
| E2-T08 | E2-T01/T07 | Strict campaign/lineage parser, declarations, locked LR selection, final preflight integration | **NOT STARTED — GATED** |
| E2-T09 | E2-T02–T08 | Selected validation exports, ddof=1 aggregate, exact-hash-bound primary E1 and contextual/complementary E1-HR comparisons from supplied official artifacts | **NOT STARTED — GATED** |
| E2-T10 | E2-T01–T09 | Complete CPU regression/readiness, single-run wrapper/runbook and opt-in acceptance preparation | **NOT STARTED — GATED**; STOP before target work |
| E2-T11 | E2-T10 + E2-TARGET-AUTH | Real E2 target acceptance including exact classifier/total parameter-count verification, exact evidence inventories and unapproved G2 candidate | **NOT STARTED — GATED**; separate from G2 review |
| E2-T12 | E2-T11 + E2-G2 PASS + E2-G3a | Four complete official tuning trials, locked selection and mandatory handoff | **NOT STARTED — GATED** |
| E2-T13 | E2-T12 + E2-G3b + applicable E2-G2 | Three fresh final full-budget runs and selected validation exports | **NOT STARTED — GATED** |
| E2-T14 | E2-T13 | Full lineage/inventory/aggregate, required comparator hashes and primary E1 / contextual-complementary E1-HR interpretation review, E2-CLOSEOUT handoff | **NOT STARTED — GATED** |

Strict dependency order is the default; tasks sharing an earlier dependency are not permission to fan out or start implementation now. Exact interfaces, test names/oracles, future commands and scoped commit requirements are in the corresponding plan task.

## Completion criteria and evidence to record

### Design preparation and freeze

- [x] E2-T00a: requested English documents exist; no approval or E2 outcome was fabricated during draft preparation. E2-G0 is now separately PASS by explicit user decision; E2-G1 remains NOT APPROVED.
- [x] Document-only checks cover links/IDs/dependencies, exact common values, fusion/batching/head contract, approval separation, source-identity policy and preservation of completed E1/HR/input evidence.
- [x] E2-G0: explicit user approval of the corrected spec/plan/ledger and E2-T00b-only scope recorded on 2026-11-05; reviewed document hashes are preserved as preparation snapshots in the candidate.
- [x] E2-T00b: complete JSON candidate with primary/contextual comparator roles, exact E1 SHA-256 `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e`, exact E1-HR SHA-256 `9cb44e8aa98d18065402ea3be750586187aeb4b0e4ff23cfa6badfcc484426cf`, and 190,588 classifier / 28,010,716 total E2 parameter-count oracles as required scientific/reporting bindings. Validate fingerprints/provenance/arithmetic and present exact candidate digest without inventing artifact paths or stage outputs. **Do not create a sidecar or approval receipt before explicit E2-G1.**
- [x] E2-T00b completeness: require both roles/hashes and JSON integer parameter oracles `190588`/`28010716` in the candidate itself. Missing or swapped roles, missing or incorrect hashes, and missing or incorrect counts block the E2-G1 review package; prose references or report-time defaults do not substitute. Comparator paths remain explicitly supplied execution-environment inputs, with no assumed repository-relative fallback.
- [ ] E2-G1: exact approved bytes verified, sidecar and digest-bound receipt created, method and named implementation scope recorded; preparation snapshots retained without mutating frozen contract bytes later.

### E2-T01–T10 implementation and CPU/deployment preparation

- [ ] Each task logs behavioral RED, GREEN, relevant unchanged regressions, source/protected-hash review and scoped commit. Use actual outcomes; imports alone and planned assertions are insufficient evidence.
- [ ] T02/T03/T05 prove shared raw jitter, both layouts, global/local transfer, B versus B*4 ordering, exact fusion, single shared parameter set, gradients and original-image tail/loss accounting. T03 verifies the actual architecture's classifier count 190,588 and total E2 count 28,010,716 against the design oracles.
- [ ] T06/T07 prove full state/worker replay, atomic failure preservation, same-run identity, complete eight-epoch control flow without early stopping, terminal-publication recovery without more training, and immutable attempt/failure records.
- [ ] T07/T08 prove scientific/runtime projection equality and docs/test-only commit eligibility; changed consumed source/runtime/inputs or immutable config/receipts/evidence fail before allocation.
- [ ] T08/T09 reject incomplete/rehashed wrong lineage, synthetic fixtures, missing candidates/seeds or stale selected metrics; prove both tolerance rules, STOP, unrounded ddof=1 and reviewed E1/HR comparator schemas. T09 consumes supplied/materialized official artifacts, enforces both pinned hashes and labels E1 primary / E1-HR contextual-complementary; missing inputs, hash mismatches or swapped roles fail without a repository-path fallback.
- [ ] T10 records complete CPU regression/compile/shell checks, local environment/limits, actual consumed import closure, and the prepared full-size real-weight opt-in harness. GPU skips are reported as unexecuted.
- [ ] STOP after T10; present concrete CPU/deployment readiness for separate E2-TARGET-AUTH.

### Target, tuning, finals, and closeout

- [ ] E2-TARGET-AUTH recorded before any E2 target/GPU work.
- [ ] T11 records actual root/source/runtime/device/allocation/weights; full train/val integrity; real train-only full-size five-view path/tail; finite gradients; mandatory verification of 190,588 classifier and 28,010,716 total E2 parameters against design/acceptance oracles; bounded state reconstruction with CUDA RNG; exact consumed-source and separate harness/evidence inventories. Fixture segments receive no official epoch/update credit; the parameter-count oracles are not prefilled official future results.
- [ ] Preserve all acceptance attempts, label proof scope accurately, and present an unapproved G2 candidate plus four concrete tuning declarations.
- [ ] E2-G2 PASS recorded by user, then E2-G3a separately authorizes only those four trials.
- [ ] T12 validates four complete eight-epoch/eight-validation outcomes, all attempt lineage, and global-maximum LR selection. **STOP** for user review; no automatic final launch.
- [ ] E2-G3b binds selection and exactly three final declarations before T13.
- [ ] T13 completes seeds 2001/3001/4001, each fresh from pinned ImageNet, with all 139992 updates/eight validations and selected validation exports.
- [ ] T14 verifies actual supplied scope, checkpoint/history/selection/report lineage, unrounded aggregate, reviewed official comparator artifacts supplied/materialized in the execution environment and both exact pinned hashes, fixed buckets/predictions/confusion/curves, timing/replay records and checksum inventory. Confirm primary architectural E1 / contextual-complementary E1-HR interpretation and no controlled local-view/resolution claim from E2-versus-E1-HR; user reviews E2-CLOSEOUT.
- [ ] Preserve selected fused-feature/checkpoint definitions for E3 handoff without starting E3. Final test stays LOCKED pending the separate study-wide gate.

## Evidence log convention and current handoff

Future task entries record task ID, changed files, actual command and result, environment, fixture/target scope, evidence paths/hashes, commit, remaining dependencies and stop condition. Failures/retries are appended, not erased. Approval records bind the exact reviewed digest/declarations/selection appropriate to their gate; the ledger does not create permission by itself.

**Current handoff:** E2-G0 PASS; E2-T00b COMPLETE — CANDIDATE/DOCUMENTATION ONLY. The candidate and its exact digest are submitted with the reviewed spec/plan/ledger for **E2-G1 review**. No unresolved scientific input or missing comparator artifact was found in the preparation checks. G1 must still explicitly approve the exact digest, select the implementation method and name its authorized scope; no freeze, sidecar or approval receipt exists yet. Implementation, CPU/GPU tests, declarations, target work, G2/G3a/G3b execution, official outcomes, E3 and final test remain unauthorized. **STOP at this handoff.**

T00b observed checks: strict JSON parse rejects duplicate keys/nonfinite values; UTF-8 LF bytes match between workspace and checkout; common optimizer/budget/scheduler/batch/loss/runtime/RNG/data fields agree with frozen E1/HR references; five-view/full-and-tail shapes, head fan-in and integer parameter oracles agree with the E2 spec. Update arithmetic and exact scheduler endpoints were checked without importing runtime modules. All 30 preparation-source byte lengths/hashes and both materialized comparator aggregates match their bindings; comparator schemas/seeds/ddof and unrounded aggregates were cross-checked. The explicit 21-path planned source closure matches the plan, with no fabricated E2 source hashes. The three G0-approved document hashes intentionally refer to their pre-status-update snapshots, avoiding a candidate/ledger digest cycle. No CPU/GPU tests, dataset-image reads, declarations or jobs occurred.

## E2-G1 approval record (2026-11-05)

E2-G1 is **PASS**, explicitly approved by the user for exact contract SHA-256 `ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72` at reviewed commit `c40dd00a22de2d3ef50440440107d66db7b7dc45`. The approved JSON bytes remain unchanged; the exact-byte sidecar and [G1 receipt](e2_g1_approval.json) record the freeze. Native task-by-task **E2-T01?T10 only** is authorized, with CPU RED ? GREEN ? scoped regression and commits. Earlier preparation-status text and frozen candidate snapshots are historical. Target/GPU work, target acceptance, tuning, finals, E3 and test images remain **NOT AUTHORIZED**. STOP after T10.
