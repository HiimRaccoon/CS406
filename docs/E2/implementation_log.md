# E2 implementation evidence

Base/reviewed commit: `c40dd00a22de2d3ef50440440107d66db7b7dc45`. Client date: 2026-11-05. E2-G1 explicitly approved; native T01-T10 only. Contract SHA-256: `ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72`. Target/GPU, tuning, finals, E3 and test images remain unauthorized.

Ruling: implement in the IDE workspace and mirror exact task files into the existing e2 checkout; no new worktree or implementation agents. This follows the user-selected branch/method; cost if wrong is synchronization, checked per commit.

Pre-flight interfaces: config -> data/model/runtime/campaign; dictionary views -> model/engine/report; scheduler/metrics -> engine; full-state primitives -> checkpoint/runner/acceptance; completed-run parser -> selection/report. No interface conflict; evidence-root path will be explicitly required consistently.

Ruling: use Python/PowerShell bookkeeping on Windows instead of Bash ledger scripts; preserve task logs and scoped commits. No scientific behavior changes. Cost if wrong: evidence bookkeeping drift; archived commands/results and per-commit byte comparisons mitigate it.

Initial baseline invocation failed before temporary fixtures: 99 passed, 191 errors, 3 deselected because .tools/e2 did not yet exist. Reproduction in unchanged E1 config showed the same missing parent. Parent now created; no scientific files changed.

G1: freeze exact approved contract and scoped native authorization Protected hashes verified: 266; frozen contract unchanged. No target work.

G1: freeze exact approved contract and scoped native authorization Protected hashes verified: 266; frozen contract unchanged. No target work.
G1 commit: `83f7044538393900ed45183002bb497fe833db28`.

T01: 12 behavioral RED failures; 42 GREEN config/regression cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.

T01: 12 behavioral RED failures; 42 GREEN config/regression cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.
T01 commit: `fe93051433a2c13a55a9cbc2eab9fe01e5bc2120`.

T02: 6 RED failures; 27 GREEN five-view/preprocessing/E1/HR data cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.
T02 commit: `d0c805f104f51cdcaca8f34994cab4299bc5325d`.

T03: 5 behavioral RED failures; 16 GREEN model/E1/HR regression cases passed; exact architecture counts verified Protected hashes verified: 266; frozen contract unchanged. No target work.
T03 commit: `4a72a098e294d348e1a3c7c520085968842d1537`.

T04: ranking behavioral RED failed; 11 GREEN metric/E1/HR cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.
T04 commit: `89e60c81eb7ec281dbba6a4830a94984b4cb6c78`.

T05: dual-view loop RED failed; 17 GREEN engine/E1/HR cases passed; original-image tail matches direct updates Protected hashes verified: 266; frozen contract unchanged. No target work.
T05 commit: `f8747f355de11765666a2372ee2059282828ac0f`.

T06: 3 behavioral RED failures; 9 GREEN checkpoint/E1/HR cases passed; strict official envelope separated from replay primitives Protected hashes verified: 266; frozen contract unchanged. No target work.
T06 commit: `59b92abe6006fe1bcf96c652920d25ef63af8be1`.

T07: 7 behavioral RED failures; 63 GREEN lifecycle/source/runtime/E1/HR cases passed; fixture envelopes remain test-owned; all official receipts bind declaration digests Protected hashes verified: 266; frozen contract unchanged. No target work.
T07 commit: `6b1d79e9dba34be191cfdca40a1de0582efdfcb0`.

T08: 12 RED failures; 55 GREEN campaign/artifact/preflight/E1/HR cases passed; all 24 LR trial permutations agree and selection mandates G3b STOP Protected hashes verified: 266; frozen contract unchanged. No target work.
T08 commit: `29d857915860c5e2b158420140341b4da01fee80`.

T09: 6 RED failures plus altered-export RED; 14 GREEN reporting/E1/HR cases passed; supplied exact comparator hashes verified read-only and deltas use percentage points Protected hashes verified: 266; frozen contract unchanged. No target work.
T09 commit: `f659672b8195b52c0a73564fa35b137322b4f575`.

T10 observed extra readiness checks: 10 initial deployment behavioral RED failures; 14 wrapper/permission GREEN cases. Current source/harness inventory checks and allocation/stale-conformance sentinels passed 21 cases. Rehashed negative NLL/wrong support/confusion/F1 reproduced four RED failures, then 22 campaign/artifact cases passed. Lost authoritative checkpoint after completed/interrupted progress reproduced two RED failures, then 39 recovery/runner/E1/HR cases passed. No inherited source was changed; these corrections are E2-owned guards required by the existing protocol.

Local CPU environment: Windows 10 build 26200, Python 3.11.18, Torch 2.8.0+cpu, TorchVision 0.23.0+cpu, Pillow 11.3.0, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1. This is explicitly not the target software/device stack. compileall and Git Bash -n returned exit 0. Whole-suite observations before the final recovery guard: 407 passed/4 GPU deselected, then 411 passed/4 GPU deselected. A fresh final rerun is recorded in the CPU check inventory.

T10 portability check: a checkout without repository-relative comparator aggregates reproduced a reporting-fixture RED. Seven reporting cases passed after private test-only comparator I/O doubles replaced that assumption. Production comparator hashes/roles remain fixed and have no override. Earlier T09 tests also read the supplied real aggregates and verified both exact pins without changing their bytes.

T10 source closure: all 21 explicitly reviewed files exist and actual transitive repository imports fit the map; report.py remains reporting audit tooling. Acceptance test/helper inventory is separate. Native implementation used no implementation agents; one fresh-context final branch reviewer follows the executing-plans requirement. No target/GPU/Slurm/test-image activity occurred.

T10: 10 deployment RED failures plus metric/recovery/portability RED checks; 123 E2 CPU and 413 full CPU cases passed; 4 GPU tests deselected; compileall and bash syntax passed; STOP before target authorization Protected hashes verified: 266; frozen contract unchanged. No target work.

T10: CPU/readiness complete; 123 E2 and 413 repository CPU cases passed; raw Windows evidence preserved byte-for-byte with scoped Git attributes; STOP before E2-TARGET-AUTH Protected hashes verified: 266; frozen contract unchanged. No target work.
