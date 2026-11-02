# E1 official tuning handoff

Recorded: **2026-11-01**, on `e1-implementation`. Authority: the user's explicit
instruction to record completed official tuning, G2 PASS and G3a PASS, together
with the supplied `src/e1/e1_tuning_4lr_8epochs.csv`. This is a documentation
status record, not a new selection artifact or runtime authorization receipt.

| Gate / stage | Current status |
|---|---|
| G1 | PASS; frozen contract unchanged |
| T10 | **COMPLETE**; user-executed UIT GPU/resume smoke, job 8198, pytest 1 passed, evidence.json PASS |
| G2 | **PASS**, as explicitly confirmed by the user |
| G3a | **PASS**, for the completed official tuning stage |
| T11a | **COMPLETE**; all four reviewed LR trials completed eight epochs with seed 1001 |
| G3b | **PASS**, per subsequent user confirmation; see t11b_training_status.md |
| T11b | **Training execution complete**, seeds 2001/3001/4001 at 1e-4 for eight epochs each; **closeout INCOMPLETE** pending official report/checkpoint-lineage verification |
| Final test access | **LOCKED**; separate later authorization required |

## Completed tuning evidence

The supplied CSV has 32 epoch rows: exactly the frozen four LR candidates,
each with epochs 1 through 8 and seed 1001. Each epoch records validation
metrics and 17499 optimizer updates, giving eight validation records and
139992 updates per trial. Every trial ends with LR `1e-6`; checkpoint selection
did not shorten the eight-epoch budget.

| Base LR | Seed | Completed epochs / validation records | Selected epoch | Supported validation Macro-F1 |
|---|---|---|---|---|
| `2.5e-5` | 1001 | 8 / 8 | 8 | `0.4184037454532644` |
| `5e-5` | 1001 | 8 / 8 | 7 | `0.4519227584900749` |
| **`1e-4`** | 1001 | 8 / 8 | **7** | **`0.45248836792097563`** |
| `2e-4` | 1001 | 8 / 8 | 7 | `0.36256515948775014` |

Selection follows the unchanged frozen rule. Within each trial, process epochs
chronologically and replace the selected checkpoint only when
`current_metric - selected_metric > tau`, with `tau = 1e-6`; otherwise retain
the earlier checkpoint. Across the four completed selected-checkpoint scores,
let `M` be their global maximum, keep candidates with `M - score <= 1e-6`,
then choose the lowest base LR among them. Here `M = 0.45248836792097563`,
and only `1e-4` is eligible. Its margin over `5e-5` is
`0.00056560943090073`, greater than `tau`. NLL and trial order do not break ties.

**Selected LR: `1e-4`. Selected checkpoint epoch: 7. Best supported validation
Macro-F1: `0.45248836792097563`.** All four runs still completed epoch 8.

## Provenance and preservation

The input CSV was read from the user's workspace without modification:

- Path: `src/e1/e1_tuning_4lr_8epochs.csv`.
- Exact size: 7081 bytes.
- SHA-256: `e569c69696ef9cf2fac831a575e66a6187095ce3be658f9943bf43f865ac055c`.
- Frozen contract SHA-256:
  `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`.

G2/G3a PASS and stage completion are recorded from the user's confirmation;
the CSV independently supports epoch/update coverage and the selection
calculation. The user additionally confirms completed T10 smoke job 8198
(recorded 2026-11-02): pytest 1 passed, evidence.json PASS, current train/val
hash audit passed, real H200 MIG environment verified, finite forward/backward,
bitwise-equal resume replay model/optimizer/scheduler/RNG/loader state and zero
test images opened. The agent did not execute or rerun this smoke. See the
runbook's completed user-executed T10 smoke record. The CSV itself does not
contain those smoke logs, allocation/source identity, checkpoint hashes or a
failure/retry inventory; none is fabricated or reconstructed here.
Existing selection/run evidence is left unchanged; the
workspace CSV is not added to Git by this documentation-only update.
`t10_acceptance.json` remains the historical preparation snapshot, not the
current gate status. This handoff and `tasks.md` supply the current status.

## Post-tuning gate disposition

T11a stopped at its mandatory handoff for separate G3b review. The user has
subsequently confirmed **G3b PASS** and supplied completed final-training
histories for seeds 2001/3001/4001. Final LR is **1e-4**, eight epochs each,
selected epochs **8/7/8**. See [t11b_training_status.md](t11b_training_status.md)
for the selected-row aggregates and remaining official report/checkpoint-lineage
checks. **T11b/E1 closeout is not complete. Test access remains LOCKED.**
No E1-HR/E2/E3 work or further execution is authorized by this status update.
