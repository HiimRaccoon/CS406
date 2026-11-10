# E1-HR completed CPU/preparation review

Base: 5367ba16923450037b445d9366df87e5d86ac389. Review current local src/e1hr, tests/e1hr, scripts/slurm/e1hr_train.sbatch and docs/E1-HR. IDE root is non-Git; .tools/publish-pilot is unchanged E1, not the HR branch. Scoped HR commits are published through GitHub tools.

Spec docs/E1-HR/specs.md; approved plan docs/E1-HR/plan.md; exact frozen contract docs/E1-HR/e1hr_numeric_contract_v1.json. Authorization stops at HR-T10; no target/GPU/test images.

Read .tools/e1hr/progress.md and docs/E1-HR/implementation_log.md Ruling lines. cpu_readiness.json records 278 CPU passes, 3 GPU deselections, protected and source/tooling snapshots. Not target acceptance.

## Review Focus

1. E1 receipts/checkpoints, a relabeled E1 run or a rehashed wrong-resolution declaration must fail HR identity validation before image opening/model allocation: HR-T01/T06/T07/T08.
2. A changed dataset root, symlink escape, missing/corrupt image or stale audit cannot pass on historical pilot conformance alone: HR-T02/T07/T11.
3. A forced termination mid-epoch, persistent-worker restart or crash between last/best/history publication must replay the previous completed boundary without RNG drift or extra budget: HR-T06/T07/T10/T11.
4. Short or failed trials, tolerance chains and mixed source/runtime identities cannot enter LR selection; final launch cannot follow automatically: HR-T08/T12.
5. Reporting from last.pt, rounding before aggregation or dropping cross-bucket false positives must fail selected-checkpoint/report tests: HR-T04/T09/T14.


Read-only review: do not edit files, change Git state, run tests/jobs or spawn subagents. Inspect requirements/code and report Critical/Important/Minor findings with file:line, triggers/effects/fixes. List every declined-to-judge behavior and why. Review acceptance wiring without executing it.
