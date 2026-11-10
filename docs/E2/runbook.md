# E2 deployment and recovery runbook

Historical operations note: the procedures below describe the formerly gated
T11-T14 workflow. On 2026-11-10, the user confirmed all E2 server work/gates
complete and instructed closure. See [server closeout record](server_closeout_2026-11-10.md).
The corresponding server receipts and run packages are not mirrored here; do
not treat this historical runbook as evidence that they were locally audited.
E3-specific execution and final evaluation/test remain separately gated.

Use the exact frozen contract and G1 receipt in this directory. The source
checkout must be clean. Review the explicit 21-file consumed-source map and
actual transitive imports; acceptance/test/reporting inventories are separate.
A different clean docs/test-only commit is eligible if every scientific/runtime
binding remains unchanged. Any consumed-source change needs renewed acceptance.

## Target acceptance preparation

Before an authorized target attempt, supply a digest-bound E2-TARGET-AUTH receipt
with `authorized_tasks=["E2-T11"]`, `target_gpu_authorized=true`, the E2 contract
digest and exact topology. G1 cannot substitute. Supply an explicit
`e2_target_acceptance_request_v1` created by `config.acceptance_request`, with
seed1001, LR2.5e-5, the full frozen recipe and 139992-update horizon, the receipt
path, and all explicit paths. Official run/config validation rejects this kind.

The reviewed interpreter is
`/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python`. The dataset and offline
weights remain the exact deployment paths in the contract. Supply reviewed,
separate E2 run/evidence roots; there are no output-root defaults. Use a new
acceptance attempt directory below the evidence root, outside the clean source
checkout and official run root. Never reuse or overwrite an attempt.

Inside a separately authorized normal/uit/mig2-35 allocation with
`gpu:2g.35gb:1`, one task, four CPUs, 16384 MiB and 09:00:00, export before Python:

```bash
export PYTHONHASHSEED=1001 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CS406_E2_GPU_SMOKE=1
export CS406_E2_TARGET_AUTH=/explicit/reviewed/target_receipt.json
export CS406_E2_ACCEPTANCE_REQUEST=/explicit/reviewed/acceptance_request.json
export CS406_E2_SMOKE_DIR=/explicit/e2/evidence/new_attempt
"$CS406_PYTHON" -m pytest tests/e2/test_gpu_smoke.py -q -m gpu
```

These are templates for a later authorized allocation, not launch instructions
for the current stage. No Slurm submission or auto-retry command is provided.
Without opt-in the probe skips; with opt-in missing permission or inputs fails.

The probe audits all 34997 train and 7497 validation image bytes before model
allocation, then decodes only three deterministic train members spanning both
layouts. It checks two encoder calls per batch, full/tail shapes, every local
input gradient, parameter counts 190588/28010716, finite FP32 fused CE, and
production state capture/restore. It saves one two-update mechanics boundary,
executes an uncommitted second segment, reconstructs model/optimizer/workers,
restores and replays that segment. Four fixture updates receive zero official
epoch/update credit. Equality covers numerical state, all CUDA RNGs, both loader
generators, history and selected state; measurement time is excluded. This is
bounded state reconstruction, not a full-natural-epoch or OS-kill target claim.

Retain failures, source/harness inventories, scoped timing and allocated/reserved
memory. Successful evidence emits only an **unapproved** G2 candidate. Review it
separately with the CPU package and four concrete declarations; passing does not
approve G2 or launch tuning. No such declarations are materialized by T01-T10.

## Later official runs

Only after G2 and G3a may exactly four tuning trials run. Each stage receipt binds
the exact G2 receipt hash and a `declarations` map from all authorized run IDs to
their config SHA-256 digests, in addition to `run_ids`. G3b also binds the exact
locked-selection file SHA-256. No declaration/receipt hash cycle is introduced.

The training wrapper handles one immutable declaration and one attempt:

```bash
export CS406_REPO=/explicit/clean/accepted/checkout
export CS406_PYTHON=/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python
export CS406_CONFIG=/explicit/reviewed/one_declaration.json
export CS406_E2_SEED=1001
# After separate run authorization, invoke the wrapper in the reviewed allocation.
# Optional argument: --resume /same/declared/run/last.pt
```

The wrapper exports deterministic environment variables before its first Python
process, checks seed/arguments/resume path, then uses one `srun` invocation. The
runner independently enforces G1/G2/stage receipts, current inputs, software,
allocation and accepted consumed-source identity before allocating the model.
`python -m src.e2.train --config PATH --check-config` is CPU-only schema checking;
it does not establish execution preflight or authorize a run. Unknown overrides,
test/search flags, alternative batches/views/LRs and foreign checkpoints fail.

## Attempt boundaries and interruption

One attempt serves one declared run; multiple attempts may be needed. Nine hours
is not a promise that eight epochs fit. USR1 at 180 seconds requests a stop at a
completed boundary; an epoch may exceed that warning. A forced kill discards the
entire uncommitted epoch. There is no automatic resubmission or walltime change.

`last.pt` is authoritative at boundary zero and after each completed training
epoch, full validation and checkpoint selection. Atomic replacement preserves the
previous readable boundary on failure. Resume restores model, optimizer, the
original scheduler horizon/position, inactive scaler, Python/NumPy/CPU/all-CUDA
RNGs, both loader generators, epoch/order, history and selected weights. Derived
best/history files are repaired from the committed last checkpoint. Replay work
never earns additional budget credit. After boundary eight, publication recovery
does not train again; a published complete run cannot be extended.

Retain immutable request/environment/start/outcome records and event hashes for
every attempt. Unknown partial work stays explicitly unknown. A declaration that
failed before boundary zero can initialize again only with its unchanged config;
all failure records remain. Resume permission/retry follows later user review.

After four full-budget outcomes, validate their complete lineage, lock LR using
the global-maximum tolerance rule and STOP for G3b. Finals require three fresh
seeds 2001/3001/4001. E3 and final test require separate later authorization.

## Validation reporting

Use `src.e2.report` to evaluate the selected checkpoint and aggregate all three
complete final seeds. Supply both `--e1-report` and `--e1hr-report` as actual paths
to reviewed aggregates. Exact pins and roles come from the frozen contract;
there is no repository-relative fallback. Deltas use percentage points. Reports
retain all five metrics, ddof=1, unrounded machine values, selected lineage and
hashed CSV exports. E1 is primary; E1-HR is contextual/complementary and cannot
isolate local views or resolution. No test evaluator or E3 cache is exposed.
