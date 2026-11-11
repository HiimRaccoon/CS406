# E3 implementation ledger

Plan: docs/E3/plan.md; baseline commit: 05334cb43f769a05aca88aefe26d33bd78a0e553.

2026-11-11: User explicitly approved E3-G1 and T01-T12 implementation in src/e3. Method: executing-plans inline/native. Exact pre-implementation bytes retained in artifacts/E3/approvals/v1/E3-G1/approved_bytes; candidate files remain historical, non-executable snapshots. TARGET-AUTH/G2/cache/pilot/final/test gates remain unapproved.

Ruling: use the existing separate checkout .tools/publish-pilot on e3 for scoped commits and mirror source/tests into the IDE workspace; no extra worktree needed.
Ruling: preserve approved candidate bytes and their historical G0 fields; current authorization is recorded by G1 receipt and task ledger, avoiding a self-referential rehash of approved bytes.

Pre-flight shared interfaces: T01 schemas/payload bindings -> T02/T03/T09; T02 fixed rows/source identity -> T03; T03 cache rows -> T04/T07/T11; T04 sampler state -> T08; T05 exact paired state -> T07/T08/T10; T06 metrics -> T07/T11; T07 counters/scheduler -> T08/T09/T10; T08 atomic boundary -> T09/T10; T09 stage wrappers -> T10/T12; T10 completed-run parser -> T11. Ruling: declarations use approved normalized schemas; execution wrappers alone join declaration hashes and operation receipts; pre-target inputs do not require evidence they will produce.

T01: complete. Behavioral RED: tests/e3/test_config.py (5 failures, unimplemented strict/profile/permission functions). GREEN: E3 plus unchanged E1/E2 config tests, 32 passed. Added pinned JSON Schema validator in requirements-e3.txt; package installation is CPU preparation, not target-stack acceptance.

T02: complete — RED: five missing source/data behaviors; GREEN: source/preprocessing/E1/E2 model regressions, 20 passed; baseline 426 passed, 4 GPU deselected after workspace basetemp. Protected source/data/preprocessing bytes unchanged. Nonofficial CPU evidence only.

Correction after T02: IDE root lacked the checkout .gitattributes; mirroring a new E3 file replaced inherited rules. Restored all pre-E3 rules from the parent commit and appended only E3 exact-byte rules before further staging.

T03: complete — RED: five absent cache/replay/publication behaviors; GREEN: cache/source/preprocessing regressions, 13 passed; committed chunks are normalized schema records and only index publication marks complete. Protected source/data/preprocessing bytes unchanged. Nonofficial CPU evidence only.

T04: complete — RED: three absent sampling behaviors; GREEN: sampler and cache tests, 8 passed; literal cRT class/image oracle and exact restored suffixes; explicit cache_root added to sampler to resolve relative row payloads. Protected source/data/preprocessing bytes unchanged. Nonofficial CPU evidence only.

T05: complete — RED: three missing paired-head/optimizer behaviors; GREEN: E3 config/model plus E1/E2 model regressions, 20 passed; exact paired tensors, outer CPU RNG restoration and LR-independent initialization. Protected source/data/preprocessing bytes unchanged. Nonofficial CPU evidence only.

T06: complete — RED: two missing metric behaviors; GREEN: E3 and unchanged E1/E2 metrics, 9 passed; independent tie/F1/cross-bucket/NLL and fixed extreme-tail coverage oracles. Protected source/data/preprocessing bytes unchanged. Nonofficial CPU evidence only.

T07: complete — RED: three missing head engine behaviors; GREEN: E3 engine/metrics and unchanged E1/E2 engine regressions, 18 passed; direct 4+1 sample-mean tail oracle, schedule endpoints/interior/exhaustion and nonfinite no-update. Protected source/data/preprocessing bytes unchanged. Nonofficial CPU evidence only.
