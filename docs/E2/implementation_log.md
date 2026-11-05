# E2 implementation evidence

Base/reviewed commit: `c40dd00a22de2d3ef50440440107d66db7b7dc45`. Client date: 2026-11-05. E2-G1 explicitly approved; native T01-T10 only. Contract SHA-256: `ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72`. Target/GPU, tuning, finals, E3 and test images remain unauthorized.

Ruling: implement in the IDE workspace and mirror exact task files into the existing e2 checkout; no new worktree or implementation agents. This follows the user-selected branch/method; cost if wrong is synchronization, checked per commit.

Pre-flight interfaces: config -> data/model/runtime/campaign; dictionary views -> model/engine/report; scheduler/metrics -> engine; full-state primitives -> checkpoint/runner/acceptance; completed-run parser -> selection/report. No interface conflict; evidence-root path will be explicitly required consistently.

Ruling: use Python/PowerShell bookkeeping on Windows instead of Bash ledger scripts; preserve task logs and scoped commits. No scientific behavior changes.

Initial baseline invocation failed before temporary fixtures: 99 passed, 191 errors, 3 deselected because .tools/e2 did not yet exist. Reproduction in unchanged E1 config showed the same missing parent. Parent now created; no scientific files changed.

G1: freeze exact approved contract and scoped native authorization Protected hashes verified: 266; frozen contract unchanged. No target work.

G1: freeze exact approved contract and scoped native authorization Protected hashes verified: 266; frozen contract unchanged. No target work.
G1 commit: `83f7044538393900ed45183002bb497fe833db28`.

T01: 12 behavioral RED failures; 42 GREEN config/regression cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.

T01: 12 behavioral RED failures; 42 GREEN config/regression cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.
T01 commit: `fe93051433a2c13a55a9cbc2eab9fe01e5bc2120`.

T02: 6 RED failures; 27 GREEN five-view/preprocessing/E1/HR data cases passed Protected hashes verified: 266; frozen contract unchanged. No target work.
