# SDD ledger ? plan: docs/E1-HR/plan.md

Base: 5367ba16923450037b445d9366df87e5d86ac389
Ruling: native execution in the existing IDE workspace; publish exact scoped trees to the explicitly authorized e1hr-design-review branch through GitHub tools. The existing publishing checkout stays unchanged ? preserves non-Git IDE workspace and E1 branch ? cost if wrong: local/remote bytes must be verified at every commit.
Ruling: use Python/PowerShell ledger helpers under .tools/e1hr on Windows instead of Bash skill helpers ? same task briefs, RED/GREEN logs and commit records ? cost if wrong: manual ledger completeness checks.
Pre-flight: T01 config -> T02 data/T03 model/T07 runtime/T08 campaign; T02 samples -> T04 metrics/T05 engine/T06 loaders; T03 model/optimizer -> T05/T06/T07; T05 engine -> T06/T07/T09; T06 scientific identity projection/checkpoints -> T07/T08/T09; T07 runtime/runner -> T08/T09/T10; T08 complete parser/selection -> T09; T10 probe -> runtime builder without official-run bypass. No interface conflict; acceptance-only construction remains separate from approved official-run preflight.

HR-T01: complete - 30 passed (15 HR config plus 15 unchanged E1 config cases); baseline 184 passed, 2 GPU deselected. RED/GREEN logs: .tools/e1hr/hr-t01-red.log and -green.log. No target/GPU work.

HR-T02: complete - 36 passed (HR config/data and unchanged E1 data/preprocessing regression); direct HR/padding/RNG/isolation oracles. RED/GREEN logs: .tools/e1hr/hr-t02-red.log and -green.log. No target/GPU work.

HR-T03: complete - 36 passed (HR config/data/model and unchanged E1 model regression); isolated CPU head RNG, strict offline loading and exhaustive groups. RED/GREEN logs: .tools/e1hr/hr-t03-red.log and -green.log. No target/GPU work.

Ruling: HR-T04/HR-T05 are bounded import adapters to already implemented unchanged E1 numerical code. Their initial RED is the missing HR interface; independent literal numerical oracles run against the real exported mechanics after GREEN rather than introducing a deliberately wrong duplicate algorithm. Cost if wrong: adapter-only RED proves interface absence, while behavioral assurance comes from the independent HR and unchanged E1 regression suites.

HR-T04: complete - 7 passed; independent full-split cross-bucket/tie/metadata bucket oracles and unchanged E1 metrics regression. RED/GREEN logs: .tools/e1hr/hr-t04-red.log and -green.log. No target/GPU work.

HR-T05: complete - 12 passed; scheduler independent interior/endpoints, tail weighting, comparator and unchanged E1 engine regression. RED/GREEN logs: .tools/e1hr/hr-t05-red.log and -green.log. No target/GPU work.

HR-T06: complete - 20 passed; full-state atomic boundary replay, HR kind/shape and audit-only commit acceptance with runtime-change rejection. RED/GREEN logs: .tools/e1hr/hr-t06-red.log and -green.log. No target/GPU work.

Ruling: keep runtime/preflight source-identity regressions in tests/e1hr/test_runtime.py for focused review; include this exact additional CPU test path in the separate audit inventory under the frozen contract additional-files policy. Cost if wrong: an extra audit entry, never a scientific identity key.

HR-T07: complete - 61 HR CPU cases passed; eight-epoch runner/replay, projection identity, transitive-source rejection and rehashed E1-evidence guard. RED/GREEN logs: .tools/e1hr/hr-t07-red.log and -green.log. No target/GPU work.

HR-T08: complete - 73 HR CPU cases passed; complete parser lineage, attempts, independent LR oracle, audit-only source projection and locked selection guards. RED/GREEN logs: .tools/e1hr/hr-t08-red.log and -green.log. No target/GPU work.

HR-T09: complete - 8 report checks passed (5 HR plus 3 unchanged E1); earlier-best exports, ddof=1, source projection and read-only official E1 comparator. RED/GREEN logs: .tools/e1hr/hr-t09-red.log and -green.log. No target/GPU work.

HR-T10 initial full regression: 1 failed, 276 passed, 3 GPU deselected. The failed case was test_flat_and_decreasing_scores_still_finish_all_eight_epochs[0.0], a Windows WinError 5 during os.replace(last.pt). The isolated case passed on immediate reproduction. A new deterministic injected-sharing test failed, then passed with bounded retry of the same atomic replacement for Windows winerror 5/32/33 only; generic/permanent failures still propagate and retain the prior boundary. Targeted checkpoint/runner/recovery regression: 15 passed. Original failure retained in .tools/e1hr/full-regression-initial.log.

Ruling: tolerate transient Windows sharing denial with at most five atomic-replace attempts (0.75 seconds total waiting); no retry of optimizer work, samples, scientific budget or jobs, and no Linux permission-error substitution. The holder of the transient external handle was not observed. Cost if wrong: a permanent Windows denial is reported up to 0.75 seconds later, with the prior committed boundary preserved.

HR-T10: complete - 278 passed, 3 GPU deselected in full CPU regression; HR/CLI/atomic-retry coverage, compileall and Git Bash syntax PASS; no target work. RED/GREEN logs: .tools/e1hr/hr-t10-red.log and -green.log. No target/GPU work.

Final independent review of 9b47464: four Important findings, no Critical or Minor findings. Root grading accepts all four; one fix pass, no second review.

Final: fixed resume validation ordering - test_wrong_resume_is_rejected_before_target_inspection_or_model RED -> GREEN; validate kind/config/role/budget/accepted identity before target inspection or GPU setup. The CPU fixture adapts its explicitly synthetic checkpoint counts only at its external read edge; production guards remain strict.

Final: fixed attempt integrity - test_completed_attempt_requires_initialized_records and test_attempt_event_cannot_reference_missing_directory RED -> GREEN; initialized records and environment hashes are mandatory, and event IDs reconcile all attempt directories. Additional positive parser fixtures cover real preflight failures and explicitly abandoned partial request/outcome records; altering preserved partial bytes is rejected.

Final: fixed acceptance audit coverage - test_acceptance_inventory_includes_executed_root_conftest RED -> GREEN; exact tests/conftest.py bytes/hash join the audit-only inventory, outside scientific equality.

Final: fixed immutable attempt publication/recovery - test_attempt_snapshot_never_exposes_partial_final_and_never_overwrites and request/environment/outcome publication-kill cases RED -> GREEN. Complete flushed temporary files publish through exclusive atomic hard links; existing evidence cannot be replaced. Interrupted raw metadata is preserved with an exact-byte abandonment observation, and missing publication events are reconciled explicitly.

Final fix-pass RED: 9 failed, 18 passed, each failure reproducing the review findings. Initial GREEN caught three CPU fixture configuration mismatches after the new early validation; the fixture read edge was aligned with its existing explicit synthetic counts. Focused GREEN: 32 passed. Dedicated HR CPU: 106 passed, 1 GPU deselected.

Final: Ruling: exclusive atomic hard-link publication is required for immutable attempt JSON on the reviewed Linux/NTFS filesystems; never fall back to an overwrite - cost if wrong: unsupported hard links fail operationally and require repair before target acceptance, without scientific fallback.

Final: Ruling: actual CUDA determinism, memory, allocation and full-natural-epoch target recovery remain unverified because HR-T11 is unauthorized; the prepared bounded probe and CPU proofs do not claim them - cost if wrong: target acceptance may expose a material execution blocker requiring separate review.

Final: Ruling: real campaign metrics, final exports and binary-checkpoint closeout remain future HR-T12-HR-T14 evidence; synthetic parser/report fixtures prove control flow only - cost if wrong: actual stage evidence may fail its later lineage/closeout audit.

Final: Ruling: retain the explicitly approved scientific controls and exact contract; review implementation conformance without reselecting methodology - cost if wrong: methodological changes require explicit user review and cannot be hidden in implementation.

Final deferred minors: none.

Final full regression attempt after review: 1 failed, 289 passed, 3 GPU deselected in 113.96s. The unchanged E1 boundary-zero initialization-recovery test hit Windows WinError 5 replacing events.json. Isolated unchanged E1 rerun: 1 passed in 2.05s. Failure and isolated logs are retained; E1 remains unchanged. Repeat full regression is required before publication.

Final verification after the single fix pass: **290 passed, 3 GPU deselected** in the full CPU regression; **106 passed, 1 GPU deselected** in the dedicated HR suite. compileall, actual Git Bash syntax, all 21 consumed-file/import closure checks, 17 exact audit-only file entries and all 159 protected file hashes PASS. Four Important findings fixed with observed RED -> GREEN; no Critical or deferred Minor findings. Prior failures remain visible in the archive.

All historical `.tools/e1hr/<name>` evidence references below/above are preserved by basename under [evidence/cpu_2026-11-02](evidence/cpu_2026-11-02). Those logs are local CPU/synthetic fixtures, never official HR trials or target acceptance. A skipped/deselected GPU test provides no target evidence. The task workspace is removed after verified publication; archived evidence and scoped commits remain.

**STOPPED after HR-T10.** HR-T01-HR-T10 are COMPLETE - CPU/PREPARATION ONLY. No observed implementation blocker remains before requesting separate HR target-work authorization; actual target compatibility/determinism/memory/reconstruction remains unverified until separately authorized HR-T11. No GPU work, Slurm submission, official tuning/final run, real dataset/test-image access or E1/preprocessing/data/pilot modification occurred.

Scoped implementation commits (published to e1hr-design-review):

| Step | Commit |
|---|---|
| HR-G1 freeze | `094192f8cf21d035a078fce98907f18313fefdb3` |
| HR-T01 | `5887c5047cde2b373d1e69d2cd59344d426af851` |
| HR-T02 | `b124268c63635599a025fe0e01f2cc00e0d033c7` |
| HR-T03 | `f8e638035a703f5efb3a939f5088c6acbfe94a6d` |
| HR-T04 | `b04e146df5ecb15450be83ba17f28c639b4722ab` |
| HR-T05 | `22754f0f246ce46b593d9725f2c38ddbcc8ea97a` |
| HR-T06 | `eb27262f0a874e0b18c2b1dad42aac3b388b78a1` |
| HR-T07 | `c10d56037cb2bb6ee8ebec80ee43917ab55042ce` |
| HR-T08 | `fe811d152d861328451001b5d8f4d178cbc8d010` |
| HR-T09 | `d6212d382530b268ceffe5e0a4146f263fd4046d` |
| HR-T10 | `9b47464fc74c7f6040e61acc3ec499b9c595b689` |

The following review-fix/evidence commit binds the final source and tests; its exact SHA is supplied in the handoff and Git history.
