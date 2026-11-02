# E1 final training and verified train/validation closeout

Updated: **2026-11-02**, branch `e1-implementation`. G2/G3a/G3b PASS;
T10/T11a/T11b COMPLETE. Final LR **1e-4**, seeds **2001/3001/4001**, each with
eight epochs/eight validations; selected epochs **8/7/8**.

**E1 train/validation closeout DONE. Ready to hand off to E1-HR planning/review.**
Official report, validation exports and the supplied closeout lineage package
have been verified; see [closeout_review.md](closeout_review.md) for exact
scope, hashes and limitations. Final test access is **LOCKED / NOT AUTHORIZED**;
final-test evaluation is **NOT COMPLETE**. E1-HR is not implemented or running.

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

These figures now match the independently verified official aggregate JSON/Markdown,
selected-checkpoint analyses, original selected histories and unchanged CSV.
Per-class and long-tail/bucket report outputs are also verified. The original
evidence retains full floating-point values; no report is regenerated here.

## Evidence boundary and provenance

- Input: `src/e1/e1_final_3seeds_8epochs.csv`, 5243 exact bytes.
- SHA-256: `da2b0bbade159b7ce59f36fed3d8dc7d406cc28a6361a61c1b94308389541687`.
- Frozen contract SHA-256:
  `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`.
- G3b PASS is recorded from the user's explicit instruction. The CSV supports
  seed/LR/epoch/update coverage and the selected-row arithmetic; it does not
  itself supply the original approval receipt or prove its timing.

The previously missing closeout links are now supplied under
`src/e1/closeout_evidence/`. Declaration/config fingerprints, exact history file
hashes, selected metrics/epochs, checkpoint checksum references and selection
fingerprint all match summaries, analyses and the official aggregate. The
supplied best-checkpoint checksum list is used without reading or changing
binary checkpoints. See [closeout_review.md](closeout_review.md) for the 42-file
evidence inventory, verification boundaries and reviewed limitations.

## Completed final E1 closeout

- [x] Verify three declarations/resolved configs, complete summaries/histories,
  frozen controls and full epoch/sample/update coverage; selection bindings agree.
- [x] Verify history/config fingerprints and best checkpoint checksums across
  the supplied manifest, summaries, analyses and report; preserve checkpoint references.
- [x] Verify official per-seed selected analyses and sample-std aggregate,
  per-class/bucket reports, predictions/confusion and learning curves; record
  official metrics and limitations without modifying evidence.
- [x] Complete T11b/E1 **train/validation** closeout and hand off to E1-HR
  specification/planning review. No E1 evidence blocker remains for this handoff.

Final-test evaluation remains a separate, uncompleted task: **LOCKED / NOT
AUTHORIZED**. E1-HR needs its own reviewed specification/plan and implementation/
target/execution authorization; this closeout does not implement or run it.
