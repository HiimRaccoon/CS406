# Final independent review (read-only, 9b47464)
Critical: none. Minor: none. Four Important findings:
1. train.run validates resume identity only after preflight image access/_setup GPU allocation; validate against approved identity first.
2. campaign attempt parser makes environment/start boundary optional and does not reconcile attempt directories with event IDs.
3. acceptance inventory omits executed tests/conftest.py autouse fixture.
4. attempt snapshot writes final path directly, allowing partial immutable JSON on kill; atomic exclusive publication plus explicit abandonment/reconciliation needed.
Declined: actual CUDA/determinism/memory/allocation/full-natural-epoch target recovery (HR-T11 unauthorized); real campaign/final export checksum closeout (HR-T12-14 gated); frozen scientific controls (approved).
Root grading: all four Important; fix one pass with regressions, no second review. No target evidence inferred.
