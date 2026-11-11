# E3 Task Ledger

Updated: **2026-11-11 (Asia/Saigon)**. **E3-G0 PASS — documentation/design preparation only: plan/tasks and E3-T00b preparation. E3-G1, implementation, cache extraction, pilot execution and official head runs remain NOT AUTHORIZED. Final test remains LOCKED / NOT AUTHORIZED.**

Authority: [specs.md](specs.md) defines the scientific design, and [plan.md](plan.md) defines proposed files/interfaces, behavioral checks and task dependencies. This ledger records current scoped gate/task status. G0 permits plan/tasks and T00b design preparation only; the earlier publication of documentation on branch `e3` was separately requested and grants no implementation or execution permission.

The approved planning input was the revised spec snapshot with exact file SHA-256 **`2bb83aba8a1d1aa14dedbde747359facf1ada6ac410610a523065c2c6dd8a132`**. This identifies the pre-correction planning snapshot, not the current corrected file digest or an E3 numeric-contract freeze. Current G0 status is synchronized in the spec and this ledger; the wording/link corrections do not extend that approval. E1/E1-HR/E2 completion, source evidence and frozen scientific values remain unchanged.

## Status definitions

| Status | Meaning |
|---|---|
| COMPLETE — DOCUMENTATION ONLY | Named design documents exist; no implementation, model/cache execution or accepted scientific outcome is implied |
| PASS | Explicit user gate decision with the named scope; design/publish approval cannot authorize later execution |
| PENDING INPUTS / AWAITING REVIEW | Actual choices or complete concrete artifacts still need the specified review |
| NOT STARTED — GATED | Work has not begun and all listed dependencies/permissions are required |
| LOCKED / NOT AUTHORIZED | Access/execution remains outside every earlier gate |

## Fixed scientific guardrails

- Six main report arms: E1 original/E1-N/E1-cRT and E2 original/E2-N/E2-cRT. Source seeds exactly 2001/3001/4001; six original references plus twelve new head runs. No E1-HR main arm or StreetCLIP.
- PRIMARY cRT minus N within family/seed/cache/initialization; SECONDARY N minus original; EXPLORATORY matched-seed comparison of the two balancing effects. E3 does not reopen an E1-versus-E2 architecture experiment.
- ONE common official head protocol for all four retraining arms/all three source seeds: optimizer/config, LR/schedule, weight-decay policy, batch/accumulation, actual presentation/update budget, validation cadence, checkpoint comparator and numerics. Only mathematically required dimensional/initialization-bound differences are allowed.
- Exact N/cRT pairing additionally requires identical actual source/checkpoint/cache/initial tensors and fresh untrained optimizer states. Only sampling changes.
- Frozen E1 768-d feature; E2 1536-d `[global;mean(four locals)]`, per-view final LayerNorm and original execution order. Head-only linear counts 95356/190588; zero source representation gradients/trainables.
- Exact train/val split, 34997/7497 images, ordered 124-label mapping, frozen preprocessing; no jitter/stochastic depth during extraction, no normalization/projection/PCA or augmented feature bank.
- Natural image permutations preserve natural frequencies; cRT samples class-uniform over all 124 then image-uniform within class with replacement. Match presentations/updates/tails rather than nominal epochs. Source eight-epoch batch/LR/scheduler settings are not E3 head defaults.
- Unweighted CE/label smoothing 0; true group sample means; fixed complete head budget/early stopping OFF; supported-Macro-F1 chronological `>1e-6`/earlier comparator at the later reviewed cadence.
- Main Top-1/3/5, supported Macro-F1, Weighted-F1; all 124 outputs/stable exact ties; diagnostic NLL. Buckets `1-19`, `20-99`, `100-499`, `>=500` and overlapping weak-support `1-2`; full-split F1 retains cross-bucket false positives.
- Exactly three seeds per report arm, arithmetic mean/sample std ddof=1 before four-decimal percentage display; paired-delta std calculated from differences. No seed cherry-picking or ensembles.
- E3-specific caches, numeric contracts, acceptance, pilot, official head training and final test remain separately gated. No current document authorizes source fine-tuning or test access.

## Approval and handoff ledger

| Gate | State | Required binding / permitted scope |
|---|---|---|
| E3-G0 | **PASS — documentation/design preparation only** | Plan/tasks and E3-T00b source-manifest/cache-pilot-candidate preparation only. No numeric freeze, E3-G1, implementation, caching, pilot, official head-run or test authorization |
| E3-G1 | **PASS ? T01-T12 IMPLEMENTATION/CPU PREPARATION ONLY** | Explicit user approval; inline/native executing-plans; approved exact bytes and receipt in artifacts/E3/approvals/v1/E3-G1. No target/cache/pilot/final/test operation authorized |
| E3-TARGET-AUTH | **NOT GRANTED** | Explicit bounded target acceptance of selected-source loading, feature extraction/runtime and deterministic recovery; standalone-head fixtures only, no full-model training or bulk cache |
| E3-G2 | **NOT GRANTED** | User accepts actual bounded evidence, six-source handoff, scientific/runtime identity and cache declarations; no E1/E2 reopening or automatic extraction |
| E3-CACHE-AUTH | **NOT GRANTED** | Bind six sources and explicit full train/val extraction/original-head val replay scope |
| E3-CACHE-ACCEPT | **NOT GRANTED** | Review complete six bundles/twelve split payloads, checksums/indices/coverage and original-head equivalence |
| E3-PILOT-AUTH | **NOT GRANTED** | Bind accepted caches and bounded actual candidate/seed/source/selection/ceiling envelope; no final head permission |
| E3-HEAD-FREEZE | **NOT GRANTED** | Review all pilot outcomes and exactly one complete common official profile; approve exact bytes/digest/sidecar and paired/dimensional constraints |
| E3-FINAL-AUTH | **NOT GRANTED** | Bind the one frozen head protocol, accepted identities/caches and exactly twelve concrete final declarations |
| E3-CLOSEOUT | **NOT STARTED — GATED** | Review twelve completed head runs/six original references, complete selected analyses, paired report/lineage/inventory and limitations; close train/validation only |
| FINAL-EVALUATION | **LOCKED / NOT AUTHORIZED** | Separate study-wide approval after all intended protocols/checkpoints/train-validation closures; outside this plan |

Flow: **G0 → T00b → G1 → T01-T12 → TARGET-AUTH → T13 → G2 → CACHE-AUTH → T14 → CACHE-ACCEPT → PILOT-AUTH → T15 → STOP → HEAD-FREEZE → T16 → STOP → FINAL-AUTH → T17 → T18/CLOSEOUT**. These are the same gates as the spec, not additional gates. Dependencies never substitute for explicit execution permission or request parallel agents.

## PENDING inputs and freeze ownership

| Spec ID | Remaining input / owning task |
|---|---|
| P01 | Report observations verified in non-binding evidence; source registry/pins defined. Actual binary/analysis/history/current-data paths/hashes remain TARGET-AUTH/G2/cache blockers. |
| P02 | Candidate lossless <f4 .npy v2.0, 1024-row chunks, batch-16 extraction (2 workers, pinned memory, prefetch_factor=2, persistent workers) and atomic chunk replay defined; generated payload/index hashes and cache acceptance remain future. |
| P03 | Candidate logit atol/rtol 1e-5/1e-5 and NLL 1e-6/1e-5 with exact discrete replay defined; actual replay evidence remains future. |
| P04 | Pilot candidate predeclares E1/E2 source seed 2001, root 1001, all four arms; actual accepted-cache bindings remain future. |
| P05 | Two common AdamW LR candidates 0.001/0.003, eight complete head trials, finite work/attempt caps and common adequacy/selection rules defined; outcomes/winner pending. |
| P06 | Candidate isolated CPU linear-default initializer and exact root/stage/family/source stream defined; actual tensors/hashes and official HEAD-FREEZE adoption pending. |
| P07 | Complete optimizer/decay/clipping/LR-schedule candidates defined; final common profile pending authorized pilot/HEAD-FREEZE. |
| P08 | Candidate batch 512/accumulation 1, 20 windows, 699940 presentations/1380 updates and 20 validation passes per trial defined; official profile unselected. |
| P09 | Exact CPU natural/class/image/loader streams, keys, per-draw calls and serialized cursors/states defined; actual state bytes are future observations. |
| P10 | Explicit FP32 CUDA/Slurm/resource and completed-window atomic replay candidates defined; actual device/driver/audit/code acceptance remains future. |
| P11 | Concrete candidate record schemas, planned roots and prospective receipt roles defined; actual run IDs/code closure/generated indices/receipts/freeze bytes remain future. |

One common official protocol is already fixed, not a pending choice. Do not fill unresolved values from source LR 1e-4/5e-5, source batch 2, source horizon 8/139992 or unreviewed library defaults. Pilot candidate values must be complete before their execution even when final selection is still pending.

## Dependency-ordered tasks

| Task | Depends on | Deliverable | Current status |
|---|---|---|---|
| E3-T00a | User's spec acceptance/design-publish instruction | This plan/ledger; scoped G0 record; scientific spec retained | **COMPLETE — DOCUMENTATION ONLY** |
| E3-T00b | E3-G0 | Source registry/report checks, finite cache/replay/pilot/common-protocol candidates, explicit RNG/runtime/recovery policy, record schemas and prospective roots | **CANDIDATE PREPARED — AWAITING G1 REVIEW; SOURCE HANDOFF STILL PENDING** |
| E3-T01 | E3-G1 | E3 schemas/common-protocol/stage-permission guards and private CPU fixtures | **NOT STARTED — GATED** |
| E3-T02 | T01 | Fixed metadata/preprocessing and strict selected-source adapters/freeze | **COMPLETE — CPU EVIDENCE ONLY** |
| E3-T03 | T01/T02 | Immutable cache construction/validation and original-head replay mechanics | **COMPLETE — CPU EVIDENCE ONLY** |
| E3-T04 | T01/T03 | Natural/cRT samplers and deterministic resumable draw state | **COMPLETE — CPU EVIDENCE ONLY** |
| E3-T05 | T01/T03 | Paired linear heads, init hashes and approved common optimizer | **NOT STARTED — GATED** |
| E3-T06 | T01/T02 | Narrow metric reuse with independent exact bucket/ranking oracles | **NOT STARTED — GATED** |
| E3-T07 | T03/T04/T05/T06 | Head-only engine, sample-weighted tails/common schedule/full budget/comparator | **NOT STARTED — GATED** |
| E3-T08 | T01/T04/T05/T07 | Atomic head checkpoints and deterministic committed-boundary replay | **NOT STARTED — GATED** |
| E3-T09 | T01-T08 | E3 stage preflight/runtime, one-run lifecycle and attempt records | **NOT STARTED — GATED** |
| E3-T10 | T01/T09 | Bounded common-protocol pilot selection, strict complete-run parser and twelve final declarations | **NOT STARTED — GATED** |
| E3-T11 | T02/T03/T06/T10 | Complete selected exports, six-arm aggregate and PRIMARY/SECONDARY/EXPLORATORY report | **NOT STARTED — GATED** |
| E3-T12 | T01-T11 | CPU/deployment readiness and opt-in bounded acceptance harness | **NOT STARTED — GATED** |
| E3-T13 | T12 + TARGET-AUTH | Actual bounded source/extraction/runtime/recovery evidence and unapproved G2 candidate | **NOT STARTED — GATED** |
| E3-T14 | T13 + G2 + CACHE-AUTH | Six complete caches/twelve split payloads, original-head full-val replay; CACHE-ACCEPT handoff | **NOT STARTED — GATED** |
| E3-T15 | T14 + CACHE-ACCEPT + PILOT-AUTH | Bounded pilot outcomes and one common final protocol candidate; mandatory review stop | **NOT STARTED — GATED** |
| E3-T16 | T15 + HEAD-FREEZE | Record exact approved final protocol/sidecar/receipt and prepare twelve final declarations; no jobs | **NOT STARTED — GATED** |
| E3-T17 | T16 + FINAL-AUTH + applicable accepted identities | Twelve full-budget head runs and selected-val analyses; six original references retained | **NOT STARTED — GATED** |
| E3-T18 | T17 | Full lineage/export/paired-aggregate/inventory review and E3-CLOSEOUT handoff | **NOT STARTED — GATED** |

Every production task T01 onward requires G1, even where its immediate dependencies name earlier tasks only. T00b preparation grants no implementation permission. The future implementation method is not selected by this documentation task or the publication branch name.

## Completion evidence to record later

### T00a — Design preparation

- [x] Record the user's G0/design-publication scope without promoting it to implementation or target approval.
- [x] Write the requested plan/task ledger with module interfaces, behavioral tests, existing gates and all pending numeric choices.
- [x] Preserve the revised spec and protected completed experiments; prepare documentation for branch e3 only.

### T00b/G1 — Concrete input review

- [x] Verify both supplied external report bytes against fixed hashes; keep actual local paths only in non-binding evidence. Binary/analysis/current-data inputs remain PENDING and block applicable target/cache stages.
- [x] Prepare e3_source_manifest_v1.json, e3_cache_pilot_contract_v1.json, e3_numeric_contract_v1.json and schemas/e3_record_schemas_v1.json as non-executable candidates.
- [x] Define finite P02/P03 cache/replay, P04/P05 two-profile/four-arm pilot, P06-P09 common numeric/RNG and applicable P10 runtime/recovery candidates; no winner or official final profile selected.
- [x] Separate prospective roots/receipt schemas from source inputs, generated accepted caches and future approvals; no sidecar/receipt created.
- [x] Validate strict JSON, schema references/record constraints, arithmetic/fingerprints and inherited metadata; supply exact package byte/digest inventory in non-binding evidence.
- [x] User reviews choices, incomplete P01 handoff and stage boundaries, selects implementation method/named scope and grants exact-byte G1 approval.
- [x] Only after explicit G1 approval, create its approved sidecar/receipt; all future cache/pilot/head gates remain separate.

### T01-T12 — Implementation and preparation

- [ ] Each task completes its plan checklist with behavioral RED/GREEN commands, relevant unchanged regressions, protected-byte review and scoped evidence/commit.
- [ ] T02/T03 prove exact frozen features, both layouts, source-seed/cache identity, original-head replay and complete train/val order; no test access, transform or mixed cache.
- [ ] T04/T05/T07 prove natural versus class-then-image uniform sampling, paired actual initialization, ONE common profile, presentation-weighted tails and full budget/selection.
- [ ] T08/T09 prove exact cRT draw/RNG/state replay, atomic authoritative recovery, stage guards before access/allocation and no source-encoder gradients/training.
- [ ] T10/T11 reject incomplete/rehashed wrong lineage, hidden family-specific profiles and missing arms/seeds; prove paired-delta ddof=1, exact metrics/buckets and ranked comparison hierarchy.
- [ ] T12 records actual complete CPU/compile/wrapper results and separate actual source/harness inventories; unexecuted target checks remain explicitly unexecuted. Stop before TARGET-AUTH.

### T13-T18 — Separately authorized execution

- [ ] T13 executes only bounded source-loading/extraction/runtime/recovery acceptance under TARGET-AUTH, with actual bytes/device/runtime/evidence and no encoder optimization; hand off for G2.
- [ ] T14 executes only approved bulk train/val caches and original-head replay after G2/CACHE-AUTH; present all six source bundles for CACHE-ACCEPT.
- [ ] T15 executes only the finite PILOT-AUTH envelope, publishes every candidate/outcome/failure and proposes ONE common final profile; stop before HEAD-FREEZE.
- [ ] T16 records the explicit exact HEAD-FREEZE decision/sidecar/receipt, validates complete common-profile values and prepares twelve concrete declarations; stop before FINAL-AUTH.
- [ ] T17 executes only those twelve declared head runs, full common budget and selected analyses, preserving paired initial state/source/cache identity and every attempt/replay record.
- [ ] T18 validates all six original references/twelve complete heads, content-checked predictions/CSVs, ddof=1 paired reports, exact inventory and limitations; present for explicit E3-CLOSEOUT. Test remains locked.


## T00b candidate package for G1 review

These artifacts are **review candidates only**, not approved execution inputs:

| Artifact | Role |
|---|---|
| [e3_numeric_contract_v1.json](e3_numeric_contract_v1.json) | Fixed science, candidate policies and future-observation register |
| [e3_source_manifest_v1.json](e3_source_manifest_v1.json) | Exact six-source pins and actual materialization requirements |
| [e3_cache_pilot_contract_v1.json](e3_cache_pilot_contract_v1.json) | Complete finite cache/replay/pilot/init/optimizer/budget/RNG/runtime/recovery proposals |
| [e3_record_schemas_v1.json](schemas/e3_record_schemas_v1.json) | Normalized candidate/runtime record schemas; schema definition grants no receipt |
| [e3_t00b_preparation_evidence_v1.json](e3_t00b_preparation_evidence_v1.json) | Non-binding local observations, verification scope and preparation byte/hash inventory |

New numeric policies are proposed under the current T00b request, not inherited or already approved. G1 still needs explicit review. No sidecar/receipt, code, cache, model execution, pilot or official head run is created by preparation. Source handoff and generated/runtime observations remain separately gated.

## Evidence and current handoff

Future entries record actual task ID, files, command/outcome, environment, fixture-versus-target scope, path/hash inventory, commit and remaining authorization dependencies. Failed/partial/replayed work cannot count as a successful completed budget. Source/cache/head binary checks must say whether actual bytes or only supplied checksum references were inspected.

Current completion is **T00a documentation and T00b candidate preparation only**. The finite candidate package is prepared for G1 review; G0 still authorizes documentation/design preparation only. G1 approval, implementation-method selection, actual source handoff/target evidence and every later execution remain pending. No E3 code/cache/pilot/head training or final-test access has occurred under this ledger.
