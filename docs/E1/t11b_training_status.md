# E1 final-training status and closeout checklist

Recorded: **2026-11-02**, branch `e1-implementation`. The user explicitly
confirms **G3b PASS** and supplies `src/e1/e1_final_3seeds_8epochs.csv` for
documentation review. Final LR is **`1e-4`**. G2/G3a remain PASS and T11a
remains COMPLETE. This is a CSV-based status summary, not the official
checkpoint-backed report or a new runtime authorization receipt.

**Final training execution is complete for seeds 2001/3001/4001; T11b and E1
closeout remain INCOMPLETE pending official report/checkpoint-lineage
verification. Test access remains LOCKED.**

## CSV-supported completion

The supplied CSV contains exactly 24 rows: epochs 1 through 8 for each of the
three distinct final seeds, all at base LR `1e-4`, disjoint from tuning seed
1001. Each epoch records 17499 optimizer updates and validation metrics; each
run totals 139992 updates and ends at LR `1e-6`. All three complete epoch 8,
including seed 3001 whose selected checkpoint is earlier.

Checkpoint epochs are recalculated chronologically using supported validation
Macro-F1: replace only when `current_metric - selected_metric > 1e-6`,
otherwise retain the earlier epoch. Every metric below comes from that same
selected row; no metric independently chooses its own epoch.

| Final seed | Base LR | Completed epochs / validation records | Selected epoch | Supported Macro-F1, raw fraction |
|---|---|---|---|---|
| 2001 | `1e-4` | 8 / 8 | **8** | `0.44054009787587106` |
| 3001 | `1e-4` | 8 / 8 | **7** | `0.439045583574677` |
| 4001 | `1e-4` | 8 / 8 | **8** | `0.44865981679022027` |

## Selected-checkpoint aggregate

Arithmetic mean and **sample standard deviation (`ddof=1`)** across all three
selected rows, calculated before display rounding; no ensemble or seed
selection. Values use percentage units, with standard deviation in the same
units, rounded to four decimal places.

| Validation metric | Mean ± sample std |
|---|---|
| Supported Macro-F1 | **44.2748 ± 0.5174%** |
| Weighted-F1 | **74.2949 ± 0.1534%** |
| Top-1 | **75.5058 ± 0.0505%** |
| Top-3 | **89.0000 ± 0.1426%** |
| Top-5 | **92.6104 ± 0.1286%** |

These are independently recalculated CSV summaries. They do not replace
selected-checkpoint validation analysis or the lineage-validated T09 aggregate.
The unchanged CSV retains the original floating-point metric values.

## Evidence boundary and provenance

- Input: `src/e1/e1_final_3seeds_8epochs.csv`, 5243 exact bytes.
- SHA-256: `da2b0bbade159b7ce59f36fed3d8dc7d406cc28a6361a61c1b94308389541687`.
- Frozen contract SHA-256:
  `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`.
- G3b PASS is recorded from the user's explicit instruction. The CSV supports
  seed/LR/epoch/update coverage and the selected-row arithmetic; it does not
  itself supply the original approval receipt or prove its timing.

Read-only inspection of the available workspace found no official E1 final
`best.pt`/`last.pt`, lineage-bearing `validation_analysis/analysis.json` or
official three-seed aggregate to verify. Pilot artifacts and CPU fixtures are
not substitutes. Remote UIT artifacts were not accessed or regenerated; their
existence is not ruled out. No checkpoint initialization, full validation
sample coverage, immutable resume identity or failure/retry lineage is inferred
from this scalar CSV alone. Scientific code, frozen protocol, selection/run
evidence and input CSV are unchanged; the CSV is not added to Git by this
documentation-only update. The agent did not execute final training.

## Remaining before final E1 closeout

- [ ] Verify all three official declarations, complete run summaries and
  histories, G3b/selected-LR linkage, accepted source/runtime/data identity and
  failure/interruption/retry inventory. Confirm fresh pinned ImageNet starts,
  full split/sample coverage and original budget on any resume.
- [ ] Verify each selected `best.pt` and resume `last.pt`, their hashes and
  identities, selected epochs 8/7/8, and matching committed history. Preserve
  checkpoints and their lineage; create no E3 cache.
- [ ] Verify the T09 selected-checkpoint analyses and official three-seed
  mean/sample-std report against those same artifacts: overall/per-class and
  frequency-bucket metrics, manifest-keyed predictions, confusion/most-confused
  pairs and histories/learning curves. Archive the output inventory and review
  limitations. This CSV-derived table alone does not satisfy that requirement.
- [ ] Complete the validation closeout review and handoff for separately
  authorized final evaluation. **Keep test access LOCKED**; neither G3b nor
  completed training authorizes test evaluation or E1-HR/E2/E3 work.
