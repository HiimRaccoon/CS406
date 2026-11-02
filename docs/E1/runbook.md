# E1 UIT deployment and T10 acceptance

Current status (2026-11-02): **T10/T11a/T11b COMPLETE, G2/G3a/G3b PASS.
E1 train/validation closeout DONE.** Final LR **1e-4**, seeds
**2001/3001/4001**, eight epochs/eight validations each, selected epochs **8/7/8**.
Official metrics, per-class/bucket outputs and the supplied checkpoint/history/
declaration/selection lineage package are verified in
[closeout_review.md](closeout_review.md). **Ready to hand off to E1-HR planning/review;
E1-HR NOT IMPLEMENTED / NOT RUNNING. Final test access LOCKED / NOT AUTHORIZED;
final-test evaluation NOT COMPLETE.** No new training or test work is authorized.
The frozen [numeric contract](e1_numeric_contract_v1.json) and
[sidecar](e1_numeric_contract_v1.sha256) bind SHA-256
`c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`.

The user supplied and confirmed current UIT results: CPU E1 regression 62
passed; CUDA available on NVIDIA H200 MIG 2g.35gb (32.5 GiB); Torch
2.8.0+cu128, CUDA 12.8, cuDNN 91002; real-weight synthetic and real-train-data
forward/backward at batch 2, input `[2,3,384,896]`, logits `[2,124]`, finite
gradients and approximately 1.754 GiB peak reserved; 27,915,484 parameters.
These are accepted as user attestations, without agent reruns. See
[t10_acceptance.json](t10_acceptance.json) for historical T10 preparation
provenance; that snapshot is unchanged. Current gate status is in the handoff
and task ledger.

## Current next action: E1-HR specification/planning review

E1 train/validation is closed with no remaining inconsistency in the supplied
evidence. Prepare/review the E1-HR specification and stage plan before separate
implementation/target/execution approval. This documentation handoff does not
implement or run E1-HR. Do not rerun completed E1 tuning/final jobs. Retained
commands below are historical deployment reference. Final test remains
**LOCKED / NOT AUTHORIZED** under its separate later gate.

## Completed user-executed T10 smoke

The user confirms the following completed UIT acceptance evidence, recorded
on 2026-11-02. The agent did not execute or rerun the smoke; existing run
evidence and the historical preparation record remain unchanged.

| Check | User-executed result |
|---|---|
| Slurm job | **8198** |
| Pytest | **1 passed** |
| `evidence.json` | **PASS** |
| Current train/val hash audit | **Passed** |
| Real H200 MIG environment | **Verified** |
| Forward/backward | **Finite** |
| Resume replay model/optimizer/scheduler/RNG/loader state | **Bitwise equal** |
| Test images opened | **0** |

T10 is complete. G2/G3a remain PASS and T11a COMPLETE; G3b is now separately
PASS; all three final runs and the supplied lineage/report package are
verified. T11b/E1 train/validation closeout is DONE. Final test access
remains LOCKED / NOT AUTHORIZED. Retained commands do not request repeating job 8198 or
resubmitting any completed run.

## Historical T10 GPU/resume smoke instructions

Use the clean `e1-implementation` checkout at
`/datastore/binhvp/BuiTrongTan/CS406/repo`. Update it to the delivered commit
with a fast-forward pull before running. The reviewed Python executable is
`/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python`; it must already contain
pytest and the exact contract software versions. No installation or package
upgrade is performed by the wrapper or smoke. If the checkout/environment is
elsewhere, use its actual reviewed path consistently.

From a UIT login shell, after updating the checkout, run exactly this bounded
acceptance command. It is for the user to execute under target-work
authorization; it is not an official tuning or final run:

```bash
cd /datastore/binhvp/BuiTrongTan/CS406/repo
export CS406_E1_GPU_SMOKE=1
export CS406_E1_SMOKE_DIR=/datastore/binhvp/BuiTrongTan/CS406/reports/E1/t10-final-smoke-v1
export PYTHONHASHSEED=1001
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun --partition=normal --account=uit --qos=mig2-35 --gres=gpu:2g.35gb:1 \
  --ntasks=1 --cpus-per-task=4 --mem=16384M --time=09:00:00 --export=ALL \
  /datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python \
  -m pytest tests/e1/test_gpu_smoke.py -q -m gpu
```

`CS406_E1_SMOKE_DIR` must be a new directory outside the checkout and outside
`runs/E1`. For a reviewed retry, use a fresh attempt suffix; retain the earlier
directory and failure. Opt-in with missing CUDA, allocation, paths, software,
weights or integrity fails. Without `CS406_E1_GPU_SMOKE=1`, the GPU test skips;
a skip is not acceptance. Run directly under the allocation above, not inside
an unrelated allocation with different resources. No test images are opened.

The probe checks the exact pinned offline weights, actual clean source commit,
package/runtime/device/driver/allocation identity, and frozen preprocessing.
It imports the existing passing preprocessing conformance report at
`reports/pilot/evidence/uit_2026-10-31/conformance/conformance_report.json` only
as historical conformance evidence. It requires matching root, fingerprints
and packages, and refreshes train/validation byte hashes and containment via
`FixedManifest.verify_images` before GPU allocation. This integrity read is
separate from the user's already-passed forward/backward checks; it neither
reruns the pilot nor decodes validation images. To use another reviewed passing
conformance record, export `CS406_E1_CONFORMANCE_REPORT` to its absolute path.

The real dataset root is
`/datastore/binhvp/BuiTrongTan/CS406/data/geoguessr`; weights are
`/datastore/binhvp/BuiTrongTan/CS406/pretrained/convnext_tiny-983f1562.pth`.
There is no download or fallback. The smoke decodes two fixed train members,
one per supported raw layout, through `E1Dataset` and the unchanged frozen
preprocessor. It uses the actual ConvNeXt-Tiny, optimizer, deterministic FP32
runtime and full 139992-update schedule through existing E1 modules.

The interruption check is deliberately bounded: it saves boundary zero,
commits one two-image segment, executes a second uncommitted segment, discards
that state and recreates the model, optimizer, scheduler and persistent
workers, restores `last.pt`, and replays the second segment. It requires
bitwise equality of model/optimizer/scheduler, Python/NumPy/CPU/CUDA RNG and
loader generator state, as well as identical loss. Forward hooks assert exact
input/logit shapes and FP32; the engine checks finite logits/loss/gradients.
No full training epoch or validation score is claimed. The checkpoint fixture
maps one two-image segment to a mechanics boundary, carries a test-only
configuration and identity, and is rejected as an official run declaration.
It preserves the production recipe, model, batch, augmentation, LR and full
schedule. This is a state-discard/reconstruction test, not an OS kill or
cross-stack reproducibility claim; CPU runner tests cover signal/partial-epoch
control flow.

A successful attempt leaves:

- `evidence.json`: actual environment, source/data/weight lineage, current
  integrity audit, train members, finite/shape/memory results and resume proof.
- `last.pt`: retained test-only committed boundary, never an official warm start.
- `g2_candidate.json`: matching runtime identity/device, **`approved: false`**.

On failure, `evidence.json` records the failure and no G2 candidate is emitted.
Return the evidence JSON, candidate JSON and pytest output for review. Keep
large checkpoints on UIT. At the T10 preparation stage, G2 still required
user review of the actual evidence, implementation readiness and concrete
tuning declarations. The user has now confirmed G2/G3a PASS and completed
tuning; the later final CSV now records completed final training. Current next
action is E1-HR specification/planning review after completed E1 train/validation closeout. An automatic smoke pass never
creates approval or submits a job.

## One declared official run per Slurm job (later gates only)

Reference only: tuning and all three final training histories are complete;
do not submit them again. G3b is PASS; E1 train/validation closeout is DONE. Declarations and accepted-source identity must match the reviewed
commit. No wrapper option can bypass preflight. G3a covers exactly the four
reviewed tuning trials; after selection, stop for the separate G3b review.

The wrapper `scripts/slurm/e1_train.sbatch` fixes partition `normal`, account
`uit`, QoS `mig2-35`, `gpu:2g.35gb:1`, one task, four CPUs, 16384 MiB RAM,
and `09:00:00`. A walltime is not a new scheduler horizon. Its four required
variables select one persisted declaration, without a loop:

```bash
export CS406_REPO=/datastore/binhvp/BuiTrongTan/CS406/repo
export CS406_PYTHON=/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python
export CS406_CONFIG=/absolute/path/to/one/reviewed/run.json
export CS406_E1_SEED=1001   # Must equal this declaration; finals require G3b.
cd "$CS406_REPO"
mkdir -p logs
# After the separate stage authorization only:
sbatch --export=ALL scripts/slurm/e1_train.sbatch
```

Set `CS406_CONFIG` to an actual declaration, not the numeric contract.
Declaration paths must include matching G2 acceptance, stage authorization and
conformance records. G2 receipt schema: `gate: "G2"`, `approved: true`, frozen
`contract_sha256`, exact accepted `identity` and `device`. A stage receipt must
bind the G2 file's SHA-256 and exact `run_ids`; G3b additionally binds the
locked selection file's SHA-256. Approval receipts are authored only after
explicit user decisions. The smoke candidate intentionally cannot pass this
gate. The runner rechecks actual allocation, software/runtime, current image
hashes, immutable declaration, source/device and approvals before model setup.

The wrapper exports `PYTHONHASHSEED` from the declared official seed,
`OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before even the declaration-check Python
process starts. A seed mismatch, unknown CLI option or other checkpoint path
fails before `srun`. Keep `NVIDIA_TF32_OVERRIDE` unset or `0`.

To resume, retain exactly the same declaration, run directory, identity,
original eight-epoch budget and authorizations. After checking the recorded
failure/interruption, use the same exports and submit only its own `last.pt`:

```bash
# Example path must correspond exactly to CS406_CONFIG's run_id/runs_dir:
sbatch --export=ALL scripts/slurm/e1_train.sbatch \
  --resume /datastore/binhvp/BuiTrongTan/CS406/runs/E1/tuning-lr0.00010000-seed1001/last.pt
```

Never resume from `best.pt`, substitute a seed/configuration, overwrite a run,
or warm-start finals from tuning. `USR1` is requested 180 seconds before
walltime; the runner records interruption at a completed epoch boundary when
possible. A forced termination can lose uncommitted work, which is replayed
from the last atomic completed boundary without extending the budget. Preserve
events, immutable attempt records, failures and retries. Scheduler/selection
and early-stopping rules remain unchanged.

## Local checks

```bash
python -m pytest tests/e1 -q -m "not gpu"
python -m compileall -q src/e1 tests/e1
bash -n scripts/slurm/e1_train.sbatch
```

CPU fixtures are not target evidence. Test-image evaluation, pilot reruns,
P1-v2, other architectures, search loops and automatic job/stage promotion are
outside T10.
