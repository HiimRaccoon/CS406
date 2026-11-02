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
