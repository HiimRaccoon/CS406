# E1-HR implementation evidence

Plan: [plan.md](plan.md). Spec: [specs.md](specs.md). Base: `5367ba16923450037b445d9366df87e5d86ac389`. Client date: 2026-11-02.

HR-G1 PASS; exact approved contract `e23fa67e1f41c7bea43446e44961a215ca488aac72ae21f5e56395e097c0a35d` verified before freeze. [hr_g1_approval.json](hr_g1_approval.json) authorizes HR-T01-HR-T10 only, native task-by-task. Target/GPU work, HR-T11/HR-G2/HR-G3a/HR-G3b and test images remain unauthorized.

Ruling: implement in the user's existing IDE workspace and publish scoped exact file trees to e1hr-design-review through GitHub tools; preserve the existing E1 publishing checkout. Cost if wrong: verify remote bytes and protected hashes on every commit.

Ruling: frozen provenance entries are preparation-input snapshots, including the pre-freeze specification digest; status updates do not rewrite the frozen contract. Cost if wrong: auditors must distinguish snapshots from current status records.

Pre-flight: reviewed config/data/model/metrics/engine/checkpoint/runtime/campaign/report interfaces match. Scientific/runtime identity equality excludes clean Git commit and tooling audit inventory, while preserving consumed-file/runtime/data/receipt integrity.

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

HR-T10 second full regression: 1 failed, 277 passed, 3 GPU deselected. Failure moved to unchanged E1 test_flat_and_decreasing_scores_still_finish_all_eight_epochs[0.01], Windows WinError 5 replacing events.json in src/e1/checkpoint.py. Its isolated rerun passed 1/1. No E1 source or evidence was changed. The failed run is retained in .tools/e1hr/full-regression-e1-sharing.log; full-suite verification is repeated for a clean result.

HR-T10: complete - 278 passed, 3 GPU deselected in full CPU regression; HR/CLI/atomic-retry coverage, compileall and Git Bash syntax PASS; no target work. RED/GREEN logs: .tools/e1hr/hr-t10-red.log and -green.log. No target/GPU work.

Final pre-review CPU evidence: full regression 278 passed, 3 GPU deselected in 110.90s; added Windows retry case included and unchanged E1 regression green. compileall and actual Git Bash -n PASS. Dedicated HR prior run: 93 passed, 1 GPU deselected; targeted final checkpoint/runner/recovery: 15 passed; entrypoints/wrapper: 14 passed. All 159 protected files unchanged. Receipt SHA-256 728a15d169fa7508de6d6ef026108db26d5a1cc720560e4ff741a114424ba694. No real dataset/test images, weights download, GPU probe, Slurm job or official run executed. See cpu_readiness.json; final independent review follows.
