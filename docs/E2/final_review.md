# E2 native whole-branch review

Date: 2026-11-05. One fresh-context, read-only reviewer (`gpt-6-astra`)
reviewed `c40dd00a22de2d3ef50440440107d66db7b7dc45` through
`62d36ec2c00717c8fbb33db95fb098b27cb4c7d2`. No implementation agents or
second reviewer were used. The reviewer performed an in-memory CPU probe and
made no file, Git, image, GPU or target changes.

The reviewer found no Critical or Minor issues and one Important issue:
`_selected_analysis` checked aggregate metrics and export hashes but accepted
missing/incorrect predictions or truncated CSVs after their hashes were updated.
An analysis could claim 7497 validation samples with zero predictions.

The author retained the Important grade: incomplete evidence must not pass
aggregation or closeout. The single fix pass now requires complete ordered
validation membership and GT labels, five distinct valid label IDs, matching
Top-1, and metrics/confusion reconstructed from the predictions. All seven CSVs
are checked against validated analysis/history content as well as their hashes.
No scientific training control, metric definition or numeric-contract byte changed.

Evidence: six initial behavioral reproductions failed as expected; the expanded
matrix of twelve malformed prediction/export cases also failed as expected.
After the fix, 27 E2/E1/E1-HR reporting/regression cases passed. Whole-suite
verification and exact source/audit inventories are recorded in
[the CPU check ledger](evidence/cpu_2026-11-05/checks.json) and
[implementation log](implementation_log.md).

The review confirmed the shared encoder, view execution/fusion/counts, image-loss
accounting, authoritative recovery, source-versus-audit separation, stage gates,
unapproved target candidates, comparator roles and protected scope.

The reviewer declined to judge two explicitly unavailable observations. The
author records the following rulings rather than treating them as proven:

- Actual UIT numerics/memory/signal delivery/recovery remain unobserved because
  target execution is not authorized. Cost if wrong: a target-specific failure
  can still block T11; explicit acceptance must establish those facts.
- Comparator payloads were absent from the review checkout. Earlier author
  checks used the supplied real aggregates and verified both exact hashes
  read-only; production requires explicit paths and pins. Cost if unavailable
  later: comparison/closeout blocks; no repository path or score is substituted.

There are no deferred Minor findings. The fix is covered by RED-to-GREEN tests;
the native workflow calls for no second review. This record grants no target,
tuning/final, E3 or test permission. STOP before E2-TARGET-AUTH.
