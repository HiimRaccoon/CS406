# E1 implementation evidence

Plan: `docs/E1/plan.md`. Base: `b7a4f6ba706bf38948166fb053532cdfdcf4d15b`.
Client date: 2026-11-01. G1 is approved in `g1_approval.json`; the exact frozen contract bytes and sidecar remain unchanged. Implementation scope is T01-T09. T10 GPU work, G2, G3a, G3b, and test-image access remain blocked.

Baseline: `python -B -m pytest -q -m "not gpu" --basetemp=.tools/pytest-e1-baseline` — 109 passed, 1 deselected.

Ruling: use native task-by-task execution for the instruction to continue — implementation method was not explicitly named; no per-task agents are needed — cost if wrong: workflow preference only, with one independent review after T09.

Ruling: implement in the user's existing non-Git workspace and mirror scoped deliverables to the existing publishing checkout on branch `e1-implementation` — preserve the IDE workspace and avoid creating another worktree or modifying main — cost if wrong: synchronization must be verified before each commit.

Ruling: use native Python/PowerShell equivalents of skill ledger helpers on Windows — preserve per-task red/green output and commits without a WSL dependency — cost if wrong: ledger completeness is checked manually.

Preflight interfaces: T01 run/recipe dictionaries feed T02/T03/T07/T08; T02 yields `(tensor,target,path)` for T04/T05/T09; T03 produces model/optimizer for T05/T06; T05 shares metrics/evaluation with T07/T09; T06 restores full progress/loaders for T07; T07 complete summaries/checkpoints feed T08/T09. No interface conflicts found. CPU fixtures replace external setup edges; no production synthetic-evidence switch is permitted.

Tasks T01-T09: pending; evidence appended after each task's red/green check.

T01: complete. RED: new task tests failed before implementation (see local .tools/e1-t01-red.log). GREEN: 15 passed; tests/e1/test_config.py. Task commit follows on e1-implementation; no GPU work.

T02: complete. RED: new task tests failed before implementation (see local .tools/e1-t02-red.log). GREEN: 8 passed (E1 data + preprocessing); expected CPU pin-memory warning. Task commit follows on e1-implementation; no GPU work.

T03: complete. RED: new task tests failed before implementation (see local .tools/e1-t03-red.log). GREEN: 4 passed; tests/e1/test_model.py. Task commit follows on e1-implementation; no GPU work.

T04: complete. RED: new task tests failed before implementation (see local .tools/e1-t04-red.log). GREEN: 3 passed; tests/e1/test_metrics.py. Task commit follows on e1-implementation; no GPU work.

T05: complete. RED: new task tests failed before implementation (see local .tools/e1-t05-red.log). GREEN: 4 passed; tests/e1/test_engine.py. Task commit follows on e1-implementation; no GPU work.

T06: complete. RED: new task tests failed before implementation (see local .tools/e1-t06-red.log). GREEN: 2 passed; tests/e1/test_checkpoint.py; fsync Windows writable-handle correction and valid atomic-failure fixture. Task commit follows on e1-implementation; no GPU work.

Ruling: last.pt carries a CPU snapshot of selected weights; best.pt is reporting-only and rebuilt from that snapshot after resume ? repairs crashes between atomic replacements without rolling back optimizer budget ? cost if wrong: an extra model-weight copy in last.pt; CPU interruption equivalence covers scientific state.

Ruling: private setup edges in CPU integration fixtures provide tiny counts and a synthetic identity, with no production bypass option ? exercises the real eight-epoch runner/checkpoints/engine while campaign validation rejects such evidence ? cost if wrong: target CUDA acceptance remains necessary at T10.

T07: complete. RED: new task tests failed before implementation (see local .tools/e1-t07-red.log). GREEN: 38 passed through T07; synthetic eight-epoch/flat/decreasing/replay/lock checks; expected CPU pin-memory warning. Task commit follows on e1-implementation; no GPU work.

T08: complete. RED: new task tests failed before implementation (see local .tools/e1-t08-red.log). GREEN: 8 passed; tests/e1/test_campaign.py; no automatic job launch. Task commit follows on e1-implementation; no GPU work.

T09: complete. RED: new task tests failed before implementation (see local .tools/e1-t09-red.log). GREEN: 3 passed; tests/e1/test_report.py; selected-checkpoint exports and sample-std aggregation. Task commit follows on e1-implementation; no GPU work.
