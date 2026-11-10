# E2 Task Ledger

Updated: **2026-11-10**. **E2 CLOSED: T00a-T14 complete; E2-G0/G1/G2/G3a/G3b PASS; target authorization, tuning handoff, and closeout complete per user's server-completion confirmation. E3-specific approval and study-wide final evaluation/test access remain separate.** See [server closeout record](server_closeout_2026-11-10.md).

Historical authority record: [specs.md](specs.md) defines the scientific requirements; [plan.md](plan.md) defines file ownership, interfaces, numerical oracles and commands. The 2026-11-05 G0 decision authorized T00b only at that time. Later G1 approval and T01-T10 scope are recorded below. On 2026-11-10 the user confirmed all remaining E2 work/gates complete on the server; see [server closeout record](server_closeout_2026-11-10.md).

The [exact frozen E2 contract](e2_numeric_contract_v1.json) is **86,335 bytes**, SHA-256 **`ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72`**. The [sidecar](e2_numeric_contract_v1.sha256) and [G1 receipt](e2_g1_approval.json) record the explicit user freeze at reviewed commit `c40dd00a22de2d3ef50440440107d66db7b7dc45`, native method and T01-T10 scope. The JSON bytes remain unchanged.

## Status definitions

| Status | Meaning |
|---|---|
| COMPLETE — DRAFT DOCUMENTATION ONLY | Requested documents exist and preparation checks are complete; no scientific approval or implementation/execution is implied |
| AWAITING REVIEW | Concrete named artifacts await an explicit user decision |
| NOT STARTED — GATED | Work has not begun; all named dependencies/permissions must be satisfied |
| NOT GRANTED / NOT APPROVED | The corresponding scoped authorization/review decision has not been supplied |
| PASS | An explicit user gate decision has been recorded with its applicable contract/evidence/declaration bindings; server-reported passes are identified as such when artifacts are not mirrored locally |
| LOCKED / NOT AUTHORIZED | No image access/evaluation is permitted by earlier E2 gates |

The server-reported T11-T14 completion below is recorded from the user's explicit confirmation; the supporting server package is not mirrored here and was not independently audited in this checkout. Local task completion evidence elsewhere in this ledger retains its original scope. GPU skips, CPU fixtures and archived pilots are not E2 target acceptance. Synthetic fixtures never count as official runs. All attempts/failures/retries stay visible.

## Approval and handoff ledger

| ID | Current state | Required review / evidence | Scope after explicit approval |
|---|---|---|---|
| E2-G0 | **PASS — explicit user decision on 2026-11-05** | User approved the current corrected spec/plan/ledger and authorized E2-T00b only; exact reviewed document snapshot hashes are recorded in candidate provenance | E2-T00b candidate preparation only, now complete |
| E2-G1 | **PASS - explicit user decision on 2026-11-05** | Exact approved digest/reviewed commit frozen; sidecar/receipt created; native task-by-task method selected | T01-T10 implementation/CPU/deployment preparation only |
| E2-TARGET-AUTH | **COMPLETE / CLOSED — user-reported server completion** | User confirmed all E2 server work/gates complete; server receipt not mirrored here | T11 acceptance completed per user report |
| E2-TARGET-ACCEPTANCE | **COMPLETE / CLOSED — user-reported** | T11 target evidence is on server per user; not independently audited in this checkout | Evidence reported complete for G2 |
| E2-G2 | **PASS / CLOSED — user-reported server gate** | User confirmed gate completion; digest-bound server receipt not mirrored here | Readiness accepted per user report |
| E2-G3a | **PASS / CLOSED — user-reported server gate** | User confirmed gate completion; tuning declaration receipts not mirrored here | T12 completed per user report |
| E2-TUNING-HANDOFF | **COMPLETE / CLOSED — user-reported** | User confirmed E2 fully completed on server; detailed trial/attempt package not mirrored here | Handoff reported complete for G3b |
| E2-G3b | **PASS / CLOSED — user-reported server gate** | User confirmed gate completion; final declaration receipts not mirrored here | T13 completed per user report |
| E2-CLOSEOUT | **PASS / CLOSED — explicit user closeout instruction** | User confirmed all E2 work complete on server and instructed closing gates; full server inventory is not mirrored here | E2 closed for transition to E3; E3/test remain separately gated |
| FINAL-EVALUATION | **LOCKED / NOT AUTHORIZED** | All intended comparison protocols/checkpoints frozen and train/validation closed, including E3 if retained, then separate explicit user approval | Outside this plan |

Flow completed: **E2-G0 → E2-T00b → E2-G1 → E2-T01–T10 → E2-TARGET-AUTH → E2-T11/E2-TARGET-ACCEPTANCE → E2-G2 → E2-G3a → E2-T12 → E2-G3b → E2-T13 → E2-T14/E2-CLOSEOUT**. T11-T14 and later gate status are recorded from the user's explicit server-completion confirmation; see [closeout record](server_closeout_2026-11-10.md).

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
| E2-T06 | E2-T01/T02/T03/T05 | State primitives, strict atomic E2 checkpoints and completed-boundary replay | **COMPLETE — CPU/PREPARATION ONLY**; 3 behavioral RED failures; 9 GREEN checkpoint/E1/HR cases passed; strict official envelope separated from replay primitives |
| E2-T07 | E2-T01–T06 | Current-target inspection/preflight, deterministic one-run/attempt lifecycle | **COMPLETE — CPU/PREPARATION ONLY**; 7 behavioral RED failures; 63 GREEN lifecycle/source/runtime/E1/HR cases passed; fixture envelopes remain test-owned; all official receipts bind declaration digests |
| E2-T08 | E2-T01/T07 | Strict campaign/lineage parser, declarations, locked LR selection, final preflight integration | **COMPLETE — CPU/PREPARATION ONLY**; 12 RED failures; 55 GREEN campaign/artifact/preflight/E1/HR cases passed; all 24 LR trial permutations agree and selection mandates G3b STOP |
| E2-T09 | E2-T02–T08 | Selected validation exports, ddof=1 aggregate, exact-hash-bound primary E1 and contextual/complementary E1-HR comparisons from supplied official artifacts | **COMPLETE — CPU/PREPARATION ONLY**; 6 RED failures plus altered-export RED; 14 GREEN reporting/E1/HR cases passed; supplied exact comparator hashes verified read-only and deltas use percentage points |
| E2-T10 | E2-T01–T09 | Complete CPU regression/readiness, single-run wrapper/runbook and opt-in acceptance preparation | **COMPLETE - CPU/PREPARATION ONLY**; final 136 E2 / 426 repository CPU passed; 4 GPU cases deselected; compile/shell/protected-source checks passed; one review finding fixed; STOP before E2-TARGET-AUTH |
| E2-T11 | E2-T10 + E2-TARGET-AUTH | Real E2 target acceptance including exact classifier/total parameter-count verification, exact evidence inventories and unapproved G2 candidate | **COMPLETE / CLOSED per user's server-completion confirmation**; target package not mirrored here |
| E2-T12 | E2-T11 + E2-G2 PASS + E2-G3a | Four complete official tuning trials, locked selection and mandatory handoff | **COMPLETE / CLOSED per user's server-completion confirmation**; trial package not mirrored here |
| E2-T13 | E2-T12 + E2-G3b + applicable E2-G2 | Three fresh final full-budget runs and selected validation exports | **COMPLETE / CLOSED per user's server-completion confirmation**; run lineage not mirrored here |
| E2-T14 | E2-T13 | Full lineage/inventory/aggregate, required comparator hashes and primary E1 / contextual-complementary E1-HR interpretation review, E2-CLOSEOUT handoff | **COMPLETE / CLOSED per user's server-completion confirmation**; server closeout inventory not mirrored here |

Strict dependency order is the default; tasks sharing an earlier dependency are not permission to fan out or start implementation now. Exact interfaces, test names/oracles, future commands and scoped commit requirements are in the corresponding plan task.

## Completion criteria and evidence to record

### Design preparation and freeze

- [x] E2-T00a: requested English documents exist; no approval or E2 outcome was fabricated during draft preparation. E2-G0 is now separately PASS by explicit user decision; E2-G1 was subsequently approved explicitly.
- [x] Document-only checks cover links/IDs/dependencies, exact common values, fusion/batching/head contract, approval separation, source-identity policy and preservation of completed E1/HR/input evidence.
- [x] E2-G0: explicit user approval of the corrected spec/plan/ledger and E2-T00b-only scope recorded on 2026-11-05; reviewed document hashes are preserved as preparation snapshots in the candidate.
- [x] E2-T00b: complete JSON candidate with primary/contextual comparator roles, exact E1 SHA-256 `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e`, exact E1-HR SHA-256 `9cb44e8aa98d18065402ea3be750586187aeb4b0e4ff23cfa6badfcc484426cf`, and 190,588 classifier / 28,010,716 total E2 parameter-count oracles as required scientific/reporting bindings. Validate fingerprints/provenance/arithmetic and present exact candidate digest without inventing artifact paths or stage outputs. **Do not create a sidecar or approval receipt before explicit E2-G1.**
- [x] E2-T00b completeness: require both roles/hashes and JSON integer parameter oracles `190588`/`28010716` in the candidate itself. Missing or swapped roles, missing or incorrect hashes, and missing or incorrect counts block the E2-G1 review package; prose references or report-time defaults do not substitute. Comparator paths remain explicitly supplied execution-environment inputs, with no assumed repository-relative fallback.
- [x] E2-G1: exact approved bytes verified, sidecar and digest-bound receipt created, method and named implementation scope recorded; preparation snapshots retained without mutating frozen contract bytes later.

### E2-T01–T10 implementation and CPU/deployment preparation

- [x] Each task logs behavioral RED, GREEN, relevant unchanged regressions, source/protected-hash review and scoped commit. Use actual outcomes; imports alone and planned assertions are insufficient evidence.
- [x] T02/T03/T05 prove shared raw jitter, both layouts, global/local transfer, B versus B*4 ordering, exact fusion, single shared parameter set, gradients and original-image tail/loss accounting. T03 verifies the actual architecture's classifier count 190,588 and total E2 count 28,010,716 against the design oracles.
- [x] T06/T07 prove full state/worker replay, atomic failure preservation, same-run identity, complete eight-epoch control flow without early stopping, terminal-publication recovery without more training, and immutable attempt/failure records.
- [x] T07/T08 prove scientific/runtime projection equality and docs/test-only commit eligibility; changed consumed source/runtime/inputs or immutable config/receipts/evidence fail before allocation.
- [x] T08/T09 reject incomplete/rehashed wrong lineage, synthetic fixtures, missing candidates/seeds or stale selected metrics; prove both tolerance rules, STOP, unrounded ddof=1 and reviewed E1/HR comparator schemas. T09 consumes supplied/materialized official artifacts, enforces both pinned hashes and labels E1 primary / E1-HR contextual-complementary; missing inputs, hash mismatches or swapped roles fail without a repository-path fallback.
- [x] T10 records complete CPU regression/compile/shell checks, local environment/limits, actual consumed import closure, and the prepared full-size real-weight opt-in harness. GPU skips are reported as unexecuted.
- [x] STOP after T10; present concrete CPU/deployment readiness for separate E2-TARGET-AUTH.

### Target, tuning, finals, and closeout

- [x] E2-TARGET-AUTH and target acceptance complete on server per user confirmation; server receipt/evidence are not mirrored locally.
- [x] T11 records actual root/source/runtime/device/allocation/weights; full train/val integrity; real train-only full-size five-view path/tail; finite gradients; mandatory verification of 190,588 classifier and 28,010,716 total E2 parameters against design/acceptance oracles; bounded state reconstruction with CUDA RNG; exact consumed-source and separate harness/evidence inventories. Fixture segments receive no official epoch/update credit; the parameter-count oracles are not prefilled official future results.
- [x] T11 acceptance and G2 evidence package completed on server per user confirmation; artifacts are not mirrored locally.
- [x] E2-G2 and E2-G3a passed on server per user confirmation.
- [x] T12 four-trial tuning, locked selection and handoff completed on server per user confirmation.
- [x] E2-G3b passed on server per user confirmation.
- [x] T13 three final runs and selected validation exports completed on server per user confirmation.
- [x] T14 validation closeout and E2-CLOSEOUT completed on server per user confirmation; see [closeout record](server_closeout_2026-11-10.md).
- [x] E2 is closed for transition to E3. E3-specific execution and final test remain under separate authorization.

## Evidence log convention and current handoff

Future task entries record task ID, changed files, actual command and result, environment, fixture/target scope, evidence paths/hashes, commit, remaining dependencies and stop condition. Failures/retries are appended, not erased. Approval records bind the exact reviewed digest/declarations/selection appropriate to their gate; the ledger does not create permission by itself.

**Current handoff:** E2 is **CLOSED** per the user's 2026-11-10 server-completion confirmation. T11-T14 and G2/G3/closeout are marked complete in this ledger; server-side receipts and lineage are not mirrored in this checkout. See [server closeout record](server_closeout_2026-11-10.md). E3-specific approval and final-evaluation/test access remain separate.

T00b observed checks: strict JSON parse rejects duplicate keys/nonfinite values; UTF-8 LF bytes match between workspace and checkout; common optimizer/budget/scheduler/batch/loss/runtime/RNG/data fields agree with frozen E1/HR references; five-view/full-and-tail shapes, head fan-in and integer parameter oracles agree with the E2 spec. Update arithmetic and exact scheduler endpoints were checked without importing runtime modules. All 30 preparation-source byte lengths/hashes and both materialized comparator aggregates match their bindings; comparator schemas/seeds/ddof and unrounded aggregates were cross-checked. The explicit 21-path planned source closure matches the plan, with no fabricated E2 source hashes. The three G0-approved document hashes intentionally refer to their pre-status-update snapshots, avoiding a candidate/ledger digest cycle. No CPU/GPU tests, dataset-image reads, declarations or jobs occurred.

## E2-G1 approval record (2026-11-05)

E2-G1 is **PASS**, explicitly approved by the user for exact contract SHA-256 `ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72` at reviewed commit `c40dd00a22de2d3ef50440440107d66db7b7dc45`. The approved JSON bytes remain unchanged; the exact-byte sidecar and [G1 receipt](e2_g1_approval.json) record the freeze. Native task-by-task **E2-T01-T10 only** is authorized, with CPU RED -> GREEN -> scoped regression and commits. Earlier preparation-status text and frozen candidate snapshots are historical. Target/GPU work, target acceptance, tuning, finals, E3 and test images remain **NOT AUTHORIZED**. STOP after T10.

Final T10 review/verification: [review record](final_review.md), [readiness](cpu_readiness.md), [exact CPU inventory](cpu_readiness.json). No GPU/Slurm/test-image work occurred; E2-T11 and all later execution gates remain unapproved.
