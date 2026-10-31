# E1 T00 review and user decision record

Client date: **2026-10-31** (Asia/Saigon). **G0 PASS; G1 NOT APPROVED.**

The user explicitly approved the E1 specification, implementation plan, and task ledger as the planning basis for T00 only. The instruction below supplies all applicable R1-R4/R6-R9 values and preserves R5. It supersedes the prior uppercase package-path instruction: implementation stays under `src/e1/` with `tests/e1/`.

This record and the complete [numeric contract v1](e1_numeric_contract_v1.json), [checksum](e1_numeric_contract_v1.sha256), [reconciled plan](plan.md), and [task ledger](tasks.md) are submitted for separate G1 review. The contract contains complete numeric inputs; it is not a launched run, an implemented configuration loader, or evidence of G1 approval. The specification's original review register is retained as historical source material; these explicit user decisions resolve its open numeric items for the submitted contract.

G1 must explicitly freeze the reviewed contract bytes/hash, authorize implementation, and select the execution method. No method has been selected. G1 approval must be recorded separately and reference the reviewed contract hash; a preparation status or checksum alone is not approval. Target smoke, tuning, final seeds, and test access retain their separate gates.

The [planning diff](t00_planning.diff) compares against the exact incoming local plan/ledger, verified by their before-edit SHA-256 values. It preserves the user's already-newer G3a/G3b ledger. Its only additional supporting change is LF rules in `.gitattributes` for `docs/E1/*.json`, `docs/E1/*.sha256`, and `docs/E1/*.md`; the publishing checkout has `core.autocrlf=true`, so these rules preserve contract and source-record bytes across checkouts.

Preparation source checkout: `.tools/publish-pilot`, branch `main`, commit `e25eda086745f175772acb160a0cfcf5fea4df3c`, remote `https://github.com/HiimRaccoon/CS406.git`. This is documentation provenance, not the future official implementation source identity. The workspace's newer G3a/G3b task-ledger revision is preserved; it must not be replaced with the older publishing-checkout copy. No commit or push is part of this G1 submission.

The contract's source hashes identify preparation snapshots. Historical plan/ledger snapshot hashes are provenance, not requirements that future progress-ledger bytes never change. Frozen data/preprocessing hashes remain runtime integrity requirements. Official source/environment/conformance/checkpoint identities will be generated and verified in their authorized implementation/acceptance/execution stages; they are not fabricated T00 measurements or unresolved training defaults.

R7 makes the approved conceptual pilot mechanics explicit: UTF-8 pipe-separated seed keys, SHA-256 first eight bytes interpreted little-endian modulo `2**63-1`, CPU epoch permutation and two jitter draws, worker seeds from `torch.initial_seed()%2**32`, and independent train/val loader generators. This is a documented binding of the adopted mechanics, not a new seed search. Exact behavior still requires T02/T06 and authorized T10 acceptance after G1.

The JSON uses explicit `enabled=false` objects without unused numeric fields for inactive AMP/scaler/compile/clipping; these are intentionally inactive policies, not missing required numeric values. Future run IDs, selected LR, implementation source identity, and actual environment/acceptance records are stage-derived artifacts under the gates, not preselected or placeholder research values.

## Pre-G1 contract finalization instruction

On 2026-10-31 the user requested only these finalization corrections: express the chronological checkpoint rule as `current_metric - selected_metric > tau`, where `tau = 1e-6`; add UIT Slurm `account=uit` and `qos=mig2-35`; and place the final contract/checksum at `docs/E1/e1_numeric_contract_v1.json` and `docs/E1/e1_numeric_contract_v1.sha256`. All other reviewed numeric training decisions and deployment values remain unchanged. No contract checksum is embedded in the JSON itself; the separate sidecar hashes the exact final JSON bytes and is the digest to which later G1 approval will bind. Re-verify referenced sources after these corrections. G1 remains unapproved and implementation remains unauthorized.

## User instruction provenance

- Source attachment identifier: `7501cd9a-9b31-4e34-a237-49d6b36ff682/Pasted text.txt`.
- Original attachment byte SHA-256: `5f3c76c6e66aefa01a4bc7e527f3f7bd1b1b7684859bb11c70c60f9d12e4c3cb`.
- The complete instruction is reproduced below with LF line endings; no external attachment path is required to read the decisions.
- Numeric-value authority: the user's explicit decisions below. Pilot evidence supports only the stated feasibility/adoption rationale; the LR grid, seed IDs, walltime, trial rule, and reporting convention are user choices, not empirically optimized pilot results.

## User instruction (verbatim; line endings normalized)

G0 is approved.

I approve the E1 specification, implementation plan, and task ledger as the planning basis for T00. This approval does NOT authorize implementation yet. Implementation remains blocked until the complete numeric contract is prepared, reviewed, and G1 is explicitly approved.

Keep the implementation package paths lowercase as currently planned:

- `src/e1/`
- `tests/e1/`

Do not change them to `src/E1/` or `tests/E1/`.

Resolve T00 using the following decisions.

### R1 — Official LR grid

Freeze the common LR grid as:

```text
[2.5e-5, 5e-5, 1e-4, 2e-4]
```

Use the same four candidates for E1, E1-HR, and E2 when their official tuning stages are reached.

For E1:
- exactly four LR candidates
- exactly one tuning seed
- every successful LR trial must complete all 8 epochs and all 8 validations
- no fifth/replacement LR may be introduced
- no refinement of the grid based on observed validation results

LR-trial selection:
1. select the candidate with the highest selected-checkpoint supported validation Macro-F1;
2. if multiple LR candidates are within `1e-6` of the maximum, select the lower LR.

This `1e-6` tolerance is the LR-trial tie rule and is separate from the per-run checkpoint comparator, although they use the same numeric tolerance.

Operational interruption may resume the same declared trial under the existing resume contract. A candidate that cannot produce its required full-budget valid outcome must not be silently ignored or replaced; stop and return for user review rather than changing the search space.

### R2 — Official training seeds

Freeze:

```text
tuning_seed = 1001
final_seeds = [2001, 3001, 4001]
```

All four IDs are distinct.

Do not reuse:
- pilot seed `3406`
- split seed `406`

Split seed `406` remains only historical/fixed-split provenance and does not control official training randomness.

### R3 — Batch

Approve the pilot-proven batch policy for official E1:

```text
effective_batch = 2 original images
microbatch = 2
gradient_accumulation = 1
drop_last = false
```

Therefore:

```text
updates_per_epoch = 17499
max_epochs = 8
planned_max_updates = 139992
```

Tail handling must preserve the sample-mean loss semantics already specified.

This is the common effective-batch contract intended for E1/E1-HR/E2 unless a later architecture-specific feasibility issue is brought back for explicit review.

### R4 — Scheduler

Approve:

```text
max_epochs = 8
planned_updates = 139992

warmup_updates = 1000
warmup_start_factor = 0.1
cosine_floor_lr = 1e-6
scheduler_unit = optimizer updates
early_stopping = OFF
```

Use the proposed explicit schedule:

```text
Let:
u = zero-based optimizer update being executed
L = selected base LR
W = 1000
N = 139992
f = 1e-6

if u < W:
    LR(u) = L * [0.1 + 0.9 * u/W]

if u >= W:
    LR(u) = f + (L-f)/2 *
            [1 + cos(pi * (u-W)/(N-W-1))]
```

Required endpoints:
- update `0` -> `0.1L`
- update `1000` -> `L`
- final update -> exactly `1e-6`

The scheduler always retains the complete 8-epoch horizon. The selected best checkpoint epoch never shortens or changes the scheduler horizon.

### R5 — Already frozen

Keep unchanged:

```text
max_epochs = 8
early_stopping = OFF
full validation every epoch
checkpoint metric = supported validation Macro-F1
checkpoint tolerance tau = 1e-6
checkpoint tie-break = earlier
```

Do not introduce patience, min_delta, or validation-based stopping.

### R6 — AdamW / optimization numerics

Approve:

```text
optimizer = AdamW
betas = [0.9, 0.999]
eps = 1e-8
gradient_clipping = OFF
```

Keep the already-frozen decay policy:

```text
weight_decay = 0.05:
- Conv weights
- Linear weights

weight_decay = 0:
- biases
- LayerNorm parameters
- ConvNeXt layer-scale parameters
```

No LLRD.

Head LR multiplier remains `1`.

### R7 — Initialization and RNG

Approve the proposed deterministic policy.

New E1 classifier:

```text
Linear(768, 124, bias=True)
```

Initialization:

```text
weight: kaiming_uniform_(a=sqrt(5))
bias: uniform(-1/sqrt(768), +1/sqrt(768))
```

Bind classifier initialization to the official training seed for that run using an isolated CPU RNG context and restore the outer RNG afterward.

Adopt the pilot deterministic mechanics conceptually:
- official training seed controls the run
- deterministic derived seeds for epoch ordering, augmentation, workers/loaders, and related streams
- do not use unstable Python `hash()`
- brightness and contrast remain independent draws
- epoch indices for official E1 are `0..7`
- resume restores required Python/NumPy/CPU/CUDA/loader RNG state
- partial interrupted epochs replay from the last committed epoch boundary
- resume cannot extend the original 8-epoch budget

Do not hard-code pilot seed `3406`.

### R8 — Runtime / UIT deployment

Approve the deterministic FP32 runtime starting contract:

```text
FP32 = ON
AMP = OFF
GradScaler = OFF
TF32 = OFF
torch.compile = OFF
float32_matmul_precision = highest

cudnn.benchmark = false
cudnn.deterministic = true
torch deterministic algorithms = true
warn_only = false

num_workers = 2
pin_memory = true
persistent_workers = true
prefetch_factor = 2

OMP_NUM_THREADS = 2
MKL_NUM_THREADS = 2
CUBLAS_WORKSPACE_CONFIG = :4096:8
```

`PYTHONHASHSEED` must be deterministically bound to the official run seed before process startup rather than copied from pilot seed 3406.

Adopt the recorded package environment as the target software lock:

```text
Python 3.11.17
torch 2.8.0+cu128
torchvision 0.23.0+cu128
Pillow 11.3.0
NumPy 2.1.3
pandas 2.2.3
scikit-learn 1.6.1
CUDA 12.8
cuDNN 91002
```

Target compute:

```text
Slurm partition = normal
GRES = gpu:2g.35gb:1
CPUs = 4
RAM = 16 GiB
job walltime = 09:00:00
```

Expected target class remains NVIDIA H200 MIG 2g.35gb.

Do not require a specific MIG UUID.

Record the actual driver/device/allocation at execution time. If the target driver/device/software environment materially differs from the reviewed environment, do not silently proceed; require the T10/G2 acceptance checks to pass under the actual environment.

Use the existing paths:

```text
dataset:
/datastore/binhvp/BuiTrongTan/CS406/data/geoguessr

pretrained:
/datastore/binhvp/BuiTrongTan/CS406/pretrained/convnext_tiny-983f1562.pth

official E1 runs:
/datastore/binhvp/BuiTrongTan/CS406/runs/E1
```

T10 remains the target UIT GPU deployment/acceptance stage. It is not P1-v2 and is not an extra optimization experiment.

### R9 — Reporting

Approve:

```text
aggregate = arithmetic mean across the 3 final seeds
std = sample standard deviation, ddof=1
```

Keep full floating-point precision in machine-readable artifacts.

For human-readable reports:
- show classification metrics as percentages to 4 decimal places
- do not round before aggregate calculations

For deterministic Top-k ranking when logits are exactly tied:
- prefer ascending `label_id`

This ranking tie rule is only a deterministic numerical convention and must not alter logits.

Final test access remains locked and requires a later explicit user authorization after E1/E1-HR/E2 protocols/checkpoints are appropriately frozen.

### Next action

Using the decisions above:

1. Prepare the complete versioned E1 numeric training contract.
2. Include provenance/source references and its checksum.
3. Reconcile `plan.md` / task numerical oracles only where these reviewed T00 decisions require it.
4. Do not implement E1 yet.
5. Do not run tests, GPU jobs, tuning, or training yet unless they are strictly documentation/contract validation that does not constitute implementation.
6. Present the complete contract and any necessary planning diff for separate G1 review.

G1 remains NOT approved until I explicitly review the completed numeric contract.
