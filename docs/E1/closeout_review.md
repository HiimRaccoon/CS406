# E1 final train/validation closeout review

Reviewed: **2026-11-02**, branch `e1-implementation`.

**T11b COMPLETE. E1 train/validation closeout DONE. Ready to hand off to
E1-HR planning/review.** All requested local evidence checks pass; no evidence
inconsistency was found. G2/G3a/G3b remain PASS. Final test access is
**LOCKED / NOT AUTHORIZED**; final-test evaluation is **NOT COMPLETE**.

This review reads the supplied official aggregate JSON/Markdown, all three
validation-analysis folders, tuning/final CSVs and the complete supplied
`src/e1/closeout_evidence/` package. No scientific code, frozen contract,
checkpoint, selection or original evidence is modified. No training, GPU
work, dataset-image access or test evaluation is performed.

## End-to-end evidence checks

| Check | Result |
|---|---|
| Final LR, seeds and selected epochs | PASS: 1e-4, 2001/3001/4001, 8/7/8 |
| Declaration JSON to config_resolved.json | PASS: equal JSON content; scientific/config fingerprints recompute correctly against the exact frozen recipe |
| Config to selection | PASS: all three selection file SHA-256 bindings match the supplied selection_v1.json |
| Run summaries to run/config/selection identity | PASS: final stage, correct seed/LR/run_id and config hash; complete status and matching accepted identity |
| Complete budgets and sample coverage | PASS: each summary and eight history rows show 8 epochs/8 validations, 34997 train and 7497 val samples per epoch, 17499 updates/epoch and 139992 total |
| History to selected metrics | PASS: recomputed tau=1e-6/earlier comparator gives 8/7/8; selected metrics match histories, CSVs, analyses and report |
| Original history file hashes | PASS: exact history.json bytes match both summary and analysis history_sha256 |
| Best checkpoint checksum references | PASS: final_best_checkpoint_sha256.txt, summary, analysis and aggregate agree for every run |
| Selection file and fingerprint | PASS: LR 1e-4, tuning seed 1001, frozen rule, all four candidates, canonical fingerprint recomputed and equal to report |
| Aggregate JSON to three analyses | PASS: all run IDs, best hashes, selected epochs, metrics, coverage, per-class and bucket aggregates agree |
| Aggregate sample std and Markdown | PASS: ddof=1, arithmetic mean/sample std from unrounded values; Markdown exactly matches the existing renderer for the JSON |
| Test access records | PASS: test_images_opened=0 in all summaries, analyses and aggregate; audit opened no dataset images |

Selection file SHA-256:
`d96a26bc96ff5f35ca389123f4befbea004ebd997a854310413c8b35ae9668ea`.
Recomputed selection fingerprint:
`24301f7e6e97ebc29e32f50b9fe1f89666de82088b5551cd90b04c319a7a11cc`.
The selection's `STOP_FOR_USER_G3b_REVIEW` / `final_execution_authorized=false`
fields are its immutable tuning-stage handoff state. They do not contradict
subsequent separate G3b PASS; that selection artifact is deliberately unchanged.

The checkpoint check uses the **supplied checksum evidence** and the matching
recorded summary/analysis/report hashes. Binary best.pt/last.pt files were not
copied, deserialized or rehashed locally; summary last.pt hashes are retained
as recorded evidence. Remote original attempt/approval archives were not
re-downloaded. This is the supplied-package closeout verification requested
by the user, not a new GPU execution or a claim to have re-executed training.
Declarations bind fresh pinned ImageNet initialization and the unchanged
original budget. No checkpoint/init operation was repeated during this audit.

## Per-seed checkpoint lineage

| Seed | Selected epoch | Agreed best.pt SHA-256 | Verified history.json SHA-256 | Verified config fingerprint |
|---|---|---|---|---|
| 2001 | 8 | `e9c6202568715d81cf0949f4a7cfcd2546839b9cb12f577cdfc2a603de910a4f` | `3fcad2ae7b556e3438bf8a5613108a85eecff28b12354065434eb237927b940d` | `e93fe2f1b86d120d12aed191afaa9b9f6e5be58bb096bfbebd6f19b4d08b9ca8` |
| 3001 | 7 | `2855d17ac932c3986a64363259d12334aa0a5af6f8f60f3fb9258a8be9544fdf` | `1f6b4ea6e1f54a2aae0592e8b29fc182134b32425e51f87195650deb1d132fe7` | `9183e563f3480ba6e11e3575abfaca5839c26d7238fb0114685b73569e5449a1` |
| 4001 | 8 | `eb74126015b3854cad78fb9c2dff7a59a2352b3299a8998ef963ca7f72c68ef2` | `aaf68c34d96f55dafe8d12c332dd1d22e46c2150b116376c5f443fd869a40866` | `70810563db83ad245b6505daed8980e0d5937c8a7ee5559ca78e8536297ca0e7` |

The three analysis identities match the frozen software/data/preprocessing/
pretrained controls and source commit
`c7a8058898835d6bfdfacec4060472a4b8634be9`. Recorded source fingerprint
`1442ae5ea1d79153a02fe9ef982333d5e6c3772168a3a7eea3918bb65937376c`
independently matches the local scientific source.

## Final selected-checkpoint validation metrics

Arithmetic mean and sample std (`ddof=1`), calculated before four-decimal
percentage display. All metrics use the same selected checkpoint per seed.

| Metric | Mean +/- sample std |
|---|---|
| Supported Macro-F1 | 44.2748% +/- 0.5174% |
| Weighted-F1 | 74.2949% +/- 0.1534% |
| Top-1 | 75.5058% +/- 0.0505% |
| Top-3 | 89.0000% +/- 0.1426% |
| Top-5 | 92.6104% +/- 0.1286% |

## Per-class, long-tail and validation exports

For each seed, all 7497 prediction paths and labels exactly match the fixed
validation manifest in order. Predictions reproduce the 124x124 confusion
matrix, Top-1/3/5, per-class F1, supported Macro-F1 and Weighted-F1. Coverage
is 124 outputs and 106 GT-supported labels. Unsupported F1 remains null.

All eight exports in each of the three validation-analysis folders were read
and cross-checked: analysis.json, overall.csv, per_class.csv, buckets.csv,
predictions.csv, confusion.csv, most_confused.csv and learning_curves.csv.
Per-class exports have 124 rows; curves match all eight original history/CSV
rows. Most-confused pairs match the confusion matrix. Bucket F1 retains
full-split false positives and Top-k uses GT-conditioned bucket samples.

| Train-support bucket | Classes | Val-supported classes | Val samples per seed | Supported Macro-F1 mean +/- sample std |
|---|---|---|---|---|
| `1-19` | 42 | 24 | 49 | 15.7738% +/- 3.2410% |
| `20-99` | 37 | 37 | 445 | 39.9711% +/- 2.0203% |
| `100-499` | 29 | 29 | 1513 | 57.5470% +/- 0.9512% |
| `>=500` | 16 | 16 | 5490 | 72.9232% +/- 0.4079% |
| `1-2` | 20 | 2 | 2 | 0.0000% +/- 0.0000% |

The 1-2-image subset has only two validation-supported classes and two
validation images. Its zero observed Macro-F1 is not evidence of reliable
extreme-tail generalization.

## Evidence inventory and limitations

The audit inventories **42 supplied evidence files**: two CSVs, aggregate
JSON/Markdown, 24 validation exports and 14 closeout-evidence files. Exact file
hashes were captured before/after documentation edits; originals are unchanged.
The [documentation inventory](closeout_inventory.json) preserves all 42 exact
file sizes/SHA-256 values and the verified per-seed lineage references. It is
an audit index, not a new run, selection or approval artifact.
Evidence remains under src/e1/ and is not republished by this docs-only commit.
Frozen contract SHA-256:
`c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`.

Report limitations are retained:

- The exact fixed split does not establish geographic/panorama independence.
- Extreme-tail support is small; seed variation does not remove sampling uncertainty.
- These are validation results only. Final test evaluation requires separate authorization.

The supplied package resolves the previous missing declaration/history/
selection-lineage blocker. E1 train/validation is closed, with no unresolved
inconsistency in the requested checks. Later verification of original binary
checkpoint bytes or full remote operational archives remains outside this
local checksum-package audit; no claim of such execution is made.

## Handoff

- [x] Verify supplied declarations, resolved configs, summaries, full histories,
  checkpoint checksum references, selection binding and official report lineage.
- [x] Verify official per-seed analyses, sample-std aggregate, per-class/bucket
  outputs, predictions/confusion and learning curves; record metrics/limitations.
- [x] Close T11b and E1 **train/validation** as DONE and record the E1-HR handoff.
- [ ] Authorize or execute final-test evaluation: **LOCKED / NOT AUTHORIZED**.

**Next experiment: E1-HR.** There is no remaining E1 evidence blocker to its
planning handoff. E1-HR still needs its own reviewed specification/plan,
implementation authorization and separate target/execution approval. It is
NOT implemented, NOT running and NOT automatically authorized by E1 closeout.
