# E3 T12 CPU preparation readiness

E3-G1 is approved for T01–T12 implementation via inline/native `executing-plans`. The exact G1 subjects and user decision are retained under `artifacts/E3/approvals/v1/E3-G1/`. The official final head protocol remains unselected.

Observed on the local Windows CPU environment, 2026-11-11:

- Baseline with a workspace-owned pytest temporary directory: 426 passed, 4 GPU tests deselected.
- E3 tests including both final review regressions: 56 passed as part of the final repository suite.
- Final repository CPU regression after independent review fixes: 482 passed, 4 GPU tests deselected (251.42 seconds).
- `compileall` for `src/e3` and `tests/e3`: passed.
- Actual Git Bash syntax checks for both E3 Slurm wrappers: passed. Behavioral wrapper tests checked seed rejection and environment exports before the first interpreter.
- The pinned record schema passed JSON Schema draft 2020-12 metaschema validation. All eight approved preparation snapshots matched their recorded exact bytes/hashes.
- Private CPU head recovery exercised both families and samplers, full batch 512 and tail 181, 32 optimizer updates and 13,736 feature presentations including replay, with exact logical state equality.

The first baseline attempt used the default Windows temporary directory and encountered 295 PermissionError setup errors. Rerunning with `--basetemp` beneath `.tools` resolved them; source/tests of E1/E1-HR/E2 were preserved.

The local schema dependencies are pinned separately in `requirements-e3.txt` and included in the consumed source inventory. They do not establish target compatibility. Actual Slurm resource fields, H200 MIG class, driver, reviewed software lock, source/data audit and code closure are checked only under a separately approved target operation.

No actual selected source binary or dataset image was loaded during this preparation. No target acceptance, bulk cache, pilot, official head training, HEAD-FREEZE or test evaluation was executed. T13 remains blocked by E3-TARGET-AUTH and materialization of the six actual selected binaries/full source evidence/current bound data audit. CPU preparation and an unexecuted target check do not satisfy G2.

See `runbook.md` for explicit inventories, receipt roles, CLI commands, wrappers and recovery. See `implementation_log.md` for task evidence, rulings and the final review/verification record.

Independent review completed: no Critical or Minor findings; two Important findings were addressed with observed failing-then-passing regressions. The consumed closure now includes package ancestor initializers and both E3 launchers. The schema/policy source-presentation conflict (48 versus 384) is rejected before target work and remains a T13 blocker requiring explicit reviewed amendment. See evidence/final_review.md.
