# E1-HR CPU readiness and future UIT deployment

HR-G1 is PASS for native task-by-task **HR-T01-HR-T10 implementation only**. The frozen numeric contract is `e23fa67e1f41c7bea43446e44961a215ca488aac72ae21f5e56395e097c0a35d`; its sidecar and `hr_g1_approval.json` record the exact-byte approval. The contract's preparation snapshots stay unchanged. No target/GPU work, HR-T11 acceptance, HR-G2, tuning, final seeds or test evaluation is authorized by this delivery. The commands below are preparation instructions for later authorized stages; none has been submitted by the agent.

## Current handoff

Review the implementation/CPU evidence in [implementation_log.md](implementation_log.md) and [tasks.md](tasks.md). The next required permission is separate **HR target-work authorization**, scoped to HR-T11 bounded acceptance only. A successful acceptance probe emits an unapproved HR-G2 candidate and does not grant HR-G2 or launch tuning. E1 acceptance and the archived HR pilot cannot replace this stage.

## Source, paths and identity

Use a clean checkout of `e1hr-design-review` at `/datastore/binhvp/BuiTrongTan/CS406/repo`, with the reviewed interpreter `/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python`. It must already contain pytest and the frozen package versions; these commands install or upgrade nothing. Verify the numeric-contract bytes and sidecar after transfer.

Data remains `/datastore/binhvp/BuiTrongTan/CS406/data/geoguessr`, and pinned offline weights remain `/datastore/binhvp/BuiTrongTan/CS406/pretrained/convnext_tiny-983f1562.pth`. No download or alternative checkpoint is permitted. Runs use `/datastore/binhvp/BuiTrongTan/CS406/runs/E1-HR`; acceptance/receipts/campaign evidence use `/datastore/binhvp/BuiTrongTan/CS406/reports/E1-HR`.

The frozen contract explicitly enumerates 21 consumed runtime files. All must exist; additional repository imports outside the reviewed closure fail. Source equality uses the exact path/hash map and canonical fingerprint, software/effective FP32 runtime, device/driver/allocation policy, data root/content and frozen fingerprints. Clean Git commit and exact acceptance/test inventories are audit metadata, not equality keys. A clean docs/test-only commit with unchanged consumed bytes/runtime does not invalidate acceptance; any consumed-file change requires renewed target acceptance. Stage receipts, immutable declarations and checkpoint/evidence hashes still validate independently. Reporting-only code is hashed separately in report audit inventories.

The acceptance audit inventory includes the executed root `tests/conftest.py`, HR fixtures/tests and package markers with exact byte counts and hashes. These are separate from the consumed scientific/runtime map.

## HR-T11 acceptance, only after separate authorization

After an explicit user decision, author a reviewed external receipt with gate `HR-TARGET-AUTH`, approved=true, experiment `E1-HR`, resolution_wh `[1344,576]`, the exact frozen contract digest, authorized_tasks `["HR-T11"]` and target_gpu_authorized=true. No such receipt is created by HR-T10, and the HR-G1 receipt cannot substitute for it. Preserve every receipt/evidence file with its exact hash.

The acceptance probe first checks that separate authorization. It verifies the actual clean consumed source, packages, allocation and pinned weights, refreshes all 34997 train/7497 validation byte hashes and containment, then verifies deterministic HR preprocessing on one actual train member per raw layout before model allocation. It imports the existing frozen preprocessing conformance record only as supporting evidence of the same root/packages/policy; it also performs fresh current-root integrity and actual HR construction, so the historical pilot pass alone cannot qualify.

Set `CS406_E1HR_CONFORMANCE_REPORT` to an actual passing frozen-preprocessing conformance record for the current root/packages. If those conditions changed, obtain a new conformance record first; do not relabel an old report. The archived record path below is usable only if its actual root/packages/fingerprints match and the current checks pass.

After authorization only, from the clean reviewed checkout:

```bash
cd /datastore/binhvp/BuiTrongTan/CS406/repo
export CS406_E1HR_GPU_SMOKE=1
export CS406_E1HR_TARGET_AUTH=/datastore/binhvp/BuiTrongTan/CS406/reports/E1-HR/approvals/hr_target_work.json
export CS406_E1HR_SMOKE_DIR=/datastore/binhvp/BuiTrongTan/CS406/reports/E1-HR/target-acceptance/attempt-001
export CS406_E1HR_CONFORMANCE_REPORT=/datastore/binhvp/BuiTrongTan/CS406/repo/reports/pilot/evidence/uit_2026-10-31/conformance/conformance_report.json
export PYTHONHASHSEED=1001
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset NVIDIA_TF32_OVERRIDE
srun --partition=normal --account=uit --qos=mig2-35 --gres=gpu:2g.35gb:1 \
  --ntasks=1 --cpus-per-task=4 --mem=16384M --time=09:00:00 --export=ALL \
  /datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python \
  -m pytest tests/e1hr/test_gpu_smoke.py -q -m gpu
```

The output directory must be new, inside the separate HR evidence root and outside the checkout/official runs. Use a new suffix for any reviewed retry and retain earlier failures. Without opt-in the probe skips; a skip is not acceptance. With opt-in, missing permission, CUDA, source, software, paths, weights, allocation or integrity fails.

The actual probe uses pinned weights, full trainable ConvNeXt-Tiny, inherited AdamW/init/RNG/loaders/FP32 and real train-only `[2,3,576,1344] -> [2,124]` forward/backward. It checks finite logits/loss/all trainable gradients and records peak allocated/reserved memory. Boundary zero and a two-image test-only mechanics boundary are saved. Subsequent work is discarded; the model/optimizer/full 139992-update scheduler/workers are rebuilt, its own last.pt is restored, and the uncommitted segment is replayed. Model/optimizer/scheduler/Python/NumPy/CPU/all-CUDA RNG/loader state and loss must be exactly equal. Restored history/selection/selected weights are also checked. No official epoch, validation score, convergence result, OS-kill or full-natural-epoch target equivalence is claimed. CPU integration tests cover actual eight-epoch runner/replay control flow.

On success retain `evidence.json`, the test-only `last.pt` and `g2_candidate.json` (approved=false). Failures retain evidence with status failed. Return the evidence/candidate and pytest output for review; keep the large checkpoint on UIT. Only explicit HR-G2 review may approve accepted scientific/runtime identity and concrete four-trial declarations. Minor operational errors do not reopen scientific controls; only material inability to execute the frozen protocol correctly may trigger explicit user review, with no automatic fallback.

## Later official declarations and attempts

All of this section remains gated. `python -m src.e1hr.campaign declare-tuning --input paths.json --output declarations` creates exactly four immutable declarations and a nonexecuted ledger, never jobs. The explicit paths JSON includes data_root, weights, runs_dir, conformance_report, acceptance_record and authorization_record. Review its four LRs and tuning seed 1001 before HR-G2/HR-G3a. An approved HR-G2 receipt binds the acceptance evidence exact hash and identity; HR-G3a separately binds that receipt's SHA-256 and exactly the four HR tuning run IDs.

The wrapper fixes normal/uit/mig2-35/gpu:2g.35gb:1, one task, four CPUs, 16384 MiB and **09:00:00 per attempt**. It accepts one declaration and optionally its own existing last.pt, with no architecture/LR/batch loop or auto-resubmission. Before every Python process it exports the declared official seed and exact thread/CUBLAS settings. The runner rechecks actual allocation and effective runtime, current data/preprocessing/weights, consumed source, HR-G2 and the stage receipt before model allocation.

Only after the applicable stage authorization:

```bash
export CS406_REPO=/datastore/binhvp/BuiTrongTan/CS406/repo
export CS406_PYTHON=/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python
export CS406_CONFIG=/absolute/path/to/one/reviewed/e1hr-run.json
export CS406_E1HR_SEED=1001
cd "$CS406_REPO"
mkdir -p logs
sbatch --export=ALL scripts/slurm/e1hr_train.sbatch
```

A run may require multiple nine-hour attempts. The archived HR timing does not show that eight epochs finish in one job. USR1 warning 180 seconds before walltime requests a completed-boundary stop, but cannot guarantee enough time to finish an epoch. Boundary zero and completed epochs after full validation are atomic commits. If termination happens mid-epoch, discard that epoch's uncommitted work and replay it from its beginning from the previous committed last.pt. No minibatch resume, warmup reset, shortened horizon or extended budget is permitted.

After reviewing the recorded attempt outcome and applicable authorizations, resubmit the **same declaration**, seed/LR/identity/output directory and its own last.pt:

```bash
sbatch --export=ALL scripts/slurm/e1hr_train.sbatch \
  --resume /datastore/binhvp/BuiTrongTan/CS406/runs/E1-HR/e1hr-tuning-lr0.00010000-seed1001/last.pt
```

This is a path-format example, not a selected HR LR or authorization to create that job. Preserve all immutable attempt environments/boundaries/outcomes, events, failures and reviewed retries; abrupt unknown partial work stays explicitly unknown. Replays never add completed work beyond eight epochs/eight validations/139992 updates. last.pt is authoritative if best/history publication was interrupted. A completed epoch-eight boundary can finish publication without another optimizer update; a published completed run cannot be extended. Never resume best.pt or silently initialize a fresh replacement.

Resume validates the checkpoint kind, declaration, role, budget and accepted scientific identity before current-target image inspection or model allocation. Attempt metadata uses flushed temporary files and exclusive atomic hard-link publication; existing records cannot be overwritten. An abrupt attempt with missing or interrupted metadata receives a separate abandonment observation binding any original partial bytes. Original records remain intact. Campaign validation reconciles every attempt directory with its events, requires initialized request/environment/start records for initialized attempts, and verifies every environment hash. A genuine failure before initialization remains auditable without inventing an environment or boundary.

## Tuning handoff, finals and reporting

After all four valid full-budget trials, `select-lr` validates declaration/receipt/history/checkpoint/attempt lineage, locks the global-maximum/tolerance/lower-LR result and **STOPs for HR-G3b review**. Failed/incomplete candidates cannot be dropped or replaced. Final declarations require that persisted selection; three final seeds require a separate HR-G3b receipt binding HR-G2, selected-file bytes and exactly seeds 2001/3001/4001. Finals initialize from ImageNet, never tuning state.

Selected-checkpoint evaluation uses `python -m src.e1hr.report evaluate-selected --run-dir ...` under the reviewed allocation/pre-start seed environment and applicable stage authorization. It opens validation only, checks the full completed run and selected history/checkpoint, then writes the eight validation exports. Offline aggregation uses `aggregate-final --run-dirs <three finals> --selection <locked HR selection> --e1-report <official E1 aggregate JSON> --output <HR report JSON>`. Supply the original official E1 aggregate from the retained E1 evidence package; the expected digest is pinned and no pilot replacement is accepted. The command never downloads or modifies E1 evidence. It produces the HR JSON/Markdown and a separate `_vs_e1.json`, using unrounded arithmetic mean/sample std ddof=1. Preserve geographic/extreme-tail/seed-sampling limitations and label all results validation-only.

## Local CPU readiness commands

```bash
python -m pytest tests/e1hr -q -m "not gpu"
python -m pytest -q -m "not gpu"
python -m compileall -q src/e1hr tests/e1hr
bash -n scripts/slurm/e1hr_train.sbatch
```

These are CPU mechanics/CLI/deployment checks, not HR-T11 or GPU evidence. No test-image evaluator is present. Final test stays LOCKED / NOT AUTHORIZED until all intended comparison experiments, including E3 if retained, freeze protocols/checkpoints and close training/validation, followed by separate explicit final-evaluation authorization.
