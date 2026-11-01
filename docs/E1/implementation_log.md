# E1 implementation evidence

Plan: `docs/E1/plan.md`. Base: `b7a4f6ba706bf38948166fb053532cdfdcf4d15b`.
Client date: 2026-11-01. G1 is approved in `g1_approval.json`; the exact frozen contract bytes and sidecar remain unchanged. Implementation scope is T01-T09. T10 GPU work, G2, G3a, G3b, and test-image access remain blocked.

Baseline: `python -B -m pytest -q -m "not gpu" --basetemp=.tools/pytest-e1-baseline` — 109 passed, 1 deselected.

Ruling: use native task-by-task execution for the instruction to continue — implementation method was not explicitly named; no per-task agents are needed — cost if wrong: workflow preference only, with one independent review after T09.

Ruling: implement in the user's existing non-Git workspace and mirror scoped deliverables to the existing publishing checkout on branch `e1-implementation` — preserve the IDE workspace and avoid creating another worktree or modifying main — cost if wrong: synchronization must be verified before each commit.

Ruling: use native Python/PowerShell equivalents of skill ledger helpers on Windows — preserve per-task red/green output and commits without a WSL dependency — cost if wrong: ledger completeness is checked manually.

Preflight interfaces: T01 run/recipe dictionaries feed T02/T03/T07/T08; T02 yields `(tensor,target,path)` for T04/T05/T09; T03 produces model/optimizer for T05/T06; T05 shares metrics/evaluation with T07/T09; T06 restores full progress/loaders for T07; T07 complete summaries/checkpoints feed T08/T09. No interface conflicts found. CPU fixtures replace external setup edges; no production synthetic-evidence switch is permitted.

Tasks T01-T09: implemented and CPU-verified in order, with one independent whole-branch review and one correction pass. T10 and later stages remain pending.

T01: complete. RED: new task tests failed before implementation (see local .tools/e1-t01-red.log). GREEN: 15 passed; tests/e1/test_config.py. Task commit follows on e1-implementation; no GPU work.

T02: complete. RED: new task tests failed before implementation (see local .tools/e1-t02-red.log). GREEN: 8 passed (E1 data + preprocessing); expected CPU pin-memory warning. Task commit follows on e1-implementation; no GPU work.

T03: complete. RED: new task tests failed before implementation (see local .tools/e1-t03-red.log). GREEN: 4 passed; tests/e1/test_model.py. Task commit follows on e1-implementation; no GPU work.

T04: complete. RED: new task tests failed before implementation (see local .tools/e1-t04-red.log). GREEN: 3 passed; tests/e1/test_metrics.py. Task commit follows on e1-implementation; no GPU work.

T05: complete. RED: new task tests failed before implementation (see local .tools/e1-t05-red.log). GREEN: 4 passed; tests/e1/test_engine.py. Task commit follows on e1-implementation; no GPU work.

T06: complete. RED: new task tests failed before implementation (see local .tools/e1-t06-red.log). GREEN: 2 passed; tests/e1/test_checkpoint.py; fsync Windows writable-handle correction and valid atomic-failure fixture. Task commit follows on e1-implementation; no GPU work.

Ruling: last.pt carries a CPU snapshot of selected weights; best.pt is reporting-only and rebuilt from that snapshot after resume - repairs crashes between atomic replacements without rolling back optimizer budget - cost if wrong: an extra model-weight copy in last.pt; CPU interruption equivalence covers scientific state.

Ruling: private setup edges in CPU integration fixtures provide tiny counts and a synthetic identity, with no production bypass option - exercises the real eight-epoch runner/checkpoints/engine while campaign validation rejects such evidence - cost if wrong: target CUDA acceptance remains necessary at T10.

T07: complete. RED: new task tests failed before implementation (see local .tools/e1-t07-red.log). GREEN: 38 passed through T07; synthetic eight-epoch/flat/decreasing/replay/lock checks; expected CPU pin-memory warning. Task commit follows on e1-implementation; no GPU work.

T08: complete. RED: new task tests failed before implementation (see local .tools/e1-t08-red.log). GREEN: 8 passed; tests/e1/test_campaign.py; no automatic job launch. Task commit follows on e1-implementation; no GPU work.

T09: complete. RED: new task tests failed before implementation (see local .tools/e1-t09-red.log). GREEN: 3 passed; tests/e1/test_report.py; selected-checkpoint exports and sample-std aggregation. Task commit follows on e1-implementation; no GPU work.

Task commits, in order: T01 `82affd6`, T02 `2d675ac`, T03 `6f54add`, T04 `6ef667a`, T05 `caa1445`, T06 `f2f6740`, T07 `d8d107f`, T08 `fde7db0`, T09 `466d0fe`.

Final independent review: six important findings, no critical/minor findings; author corrections and limits are recorded in [implementation_review.md](implementation_review.md). All nine reproducing cases failed before the corrections (`.tools/e1-review-red.log`), then the corrected regressions plus runner/campaign checks passed 22/22. Three additional parser tests cover actual successful evidence validation/selection and rehashed bad histories/weights; fabricated envelopes remain temporary CPU test inputs.

Final: fixed current-data audit, zero-boundary retry, terminal publication recovery, persistent-loader RNG drift, mixed tuning identities, and lost attempt/replay evidence. Each finding has a named RED-to-GREEN regression in the review record. No second reviewer pass is claimed.

Final author check: a corrupt last.pt could make the failure handler's diagnostic reread mask the original load error and leave status running. `test_corrupt_resume_is_recorded_as_failure_without_masking_load_error` failed, then passed after the diagnostic probe treated unreadable state as unknown and preserved the original failure. This correction is part of the same final fix commit.

Ruling: verify current train/validation bytes once in every production preflight - stale audit records alone cannot prove immutability or path containment before allocation - cost if wrong: additional startup I/O, not extra optimizer work or image decoding.

Ruling: persistent-worker reconstruction reuses the original worker bootstrap state while preserving the committed logical loader generator - matches uninterrupted persistent-worker progression and keeps sample-keyed randomness - cost if wrong: depends on pinned Torch DataLoader lifecycle and still needs target T10 resume acceptance.

Final review limits: CUDA/full-size/real-weight acceptance stays at T10; Slurm deployment and G2/G3 execution are outside this delivery; trusted local artifact records are not cryptographic attestations against mutually fabricated inputs; cross-stack bitwise equivalence is not claimed. These limits preserve the reviewed gates rather than weakening implementation acceptance. Deferred minors: none reported.

Final verification:

- `python -B -m pytest -q -m "not gpu" --basetemp=.tools/pytest-e1-final2`: **171 passed, 1 deselected**, no warnings (62 E1 cases plus 109 existing cases). The opt-in GPU test was deselected, not executed.
- `python -B -m compileall -q src/e1 tests/e1`: PASS.
- CPU CLI help for `src.e1.train`, `src.e1.campaign`, `src.e1.report`: PASS; no model/job allocation.
- Frozen numeric contract/sidecar, four fixed split artifacts, preprocessing module/config, and referenced pilot source hashes: unchanged and verified. No raw test images opened, pretrained weights downloaded, Slurm/GPU jobs launched, tuning trials/final seeds trained, or test evaluation performed.

Next required stage: T10 deployment/acceptance under its own target-work authorization. G2, G3a, G3b and final test evaluation remain open. This implementation is not evidence that the required target checks have passed.

## T10 deployment delivery (2026-11-01)

Scope: T10 only, starting from `b55d0139bb15192ffce0cc7844b51809c211e7cf` on `e1-implementation`. User explicitly requests commit/push to this branch. T01-T09 scientific modules, numeric contract/sidecar, preprocessing, split artifacts and pilot files are preserved byte-for-byte. No official declaration, stage approval or GPU job is created by this delivery.

Delivered: one-run `scripts/slurm/e1_train.sbatch`, `docs/E1/runbook.md`, CPU entrypoint/wrapper tests, opt-in E1 GPU/resume test, and `docs/E1/t10_acceptance.json`. The allocation is normal/uit/mig2-35/gpu:2g.35gb:1, one task, 4 CPUs, 16384 MiB, 09:00:00. The wrapper exports the official seed and reviewed thread/CUBLAS variables before the first Python process; it accepts only one declaration and optionally that run's existing last.pt. Production preflight still enforces actual allocation and G2/stage approval.

Evidence provenance: the user's current 62-pass UIT CPU suite, H200 MIG environment, pinned-weight synthetic and real-train/frozen-preprocessing forward/backward results are accepted exactly as supplied, without agent rerun. Unreported driver, source commit, full package inventory and resume results remain unreported. The acceptance record distinguishes those attestations from future machine-captured identity. G2/G3a/G3b remain unapproved; final T10 target resume acceptance and readiness/declaration review are pending.

Ruling: use existing `_setup`, `train_epoch`, `E1Loader` and checkpoint APIs in an opt-in test, with a minimal test-only checkpoint configuration mapping one two-image segment to a mechanics boundary - exercises real model/optimizer/full scheduler/RNG/persistent-worker reconstruction without a shortened official run or production bypass - cost if wrong: bounded state reconstruction does not prove a full natural GPU epoch, Slurm termination delivery, or cross-stack bitwise equivalence; existing CPU runner tests cover interruption control flow and actual official runs must still pass preflight/full budget.

The final target smoke imports passing historical preprocessing conformance, refreshes current train/validation hash/containment integrity before GPU model allocation, decodes only two train members through frozen preprocessing, saves a committed last.pt, discards subsequent uncommitted work, reconstructs state/workers and requires bitwise model/optimizer/scheduler/RNG/loader equality after replay. Full 139992-update scheduler parameters, batch 2 and all canonical training decisions stay unchanged. Its checkpoint/configuration cannot be validated as an official declaration. It writes only an unapproved G2 candidate, never stage authorization.

RED: initial new CPU deployment tests failed because wrapper and smoke helpers did not exist (`.tools/t10-red.log`: 3 failed, 8 passed). An earlier invocation used inaccessible system pytest temp and was rerun with workspace basetemp. GREEN: initial entrypoint checks 11 passed; initial E1 regression 73 passed, 1 GPU deselected. The wrapper test exposed an apostrophe inside Bash parameter-expansion error text; removing that apostrophe fixed the syntax, verified by actual Bash execution and `bash -n`. Two additional CPU checks verify configuration-only dispatch and missing-field rejection. No GPU test was executed by the agent.

Local validation and independent review results follow below. T10 remains partially complete until the user-executed final target smoke and readiness review exist. The exact next UIT command and evidence paths are in [runbook.md](runbook.md).

T10 independent review: a fresh read-only reviewer inspected the scoped wrapper/tests/runbook/acceptance/ledger diff against the frozen contract and existing E1 APIs. Blockers: none; important findings: none; minors: none. The reviewer did not execute tests, target work or jobs. No review fix pass was necessary.

Final: Ruling: reviewer set aside user-attested UIT results - accept the user's explicit confirmation without agent re-verification - cost if wrong: supplied attestations lack machine-captured raw logs; final smoke captures current identity separately.

Final: Ruling: reviewer set aside actual target smoke outcome - leave it pending for the user's next authorized UIT execution - cost if wrong: target acceptance cannot be claimed from local CPU success.

Final: Ruling: reviewer set aside OS termination and full-epoch GPU resume - retain the documented bounded state-discard/reconstruction probe and existing CPU interruption tests - cost if wrong: no actual Slurm-kill/full-natural-epoch GPU equivalence is established by this delivery.

Final: Ruling: reviewer set aside unchanged T01-T09 behavior beyond consumed APIs - preserve all scientific source bytes and rely on existing regression evidence - cost if wrong: pre-existing defects outside this T10 diff are not independently re-reviewed here.

Final local validation: full CPU regression `python -m pytest -q -m "not gpu"` (workspace basetemp) passed **184 tests, 2 deselected**. `python -m compileall -q src/e1 tests/e1` and `bash -n scripts/slurm/e1_train.sbatch` passed. The 34 protected scientific/input files match their pre-T10 SHA-256 values; frozen contract remains `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`. The GPU/resume test remains unexecuted by the agent. G2/G3a/G3b remain open; no tuning/final jobs or test-image access occurred. Exact scoped E1 final count is recorded with delivery below.

Final required E1 command: `python -m pytest tests/e1 -q -m "not gpu"` (workspace basetemp) passed **75 tests, 1 deselected** in 48.21 seconds. Full CPU regression passed **184 tests, 2 deselected** in 52.81 seconds. Both are local CPU results, separate from the user's manually reported UIT 62-pass result. No deferred minor findings. Delivery commit contains only the seven T10 files listed above, including this log and the updated ledger; push is explicitly requested by the user.
