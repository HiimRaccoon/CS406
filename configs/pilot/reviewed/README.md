# Pre-P0 preparation examples

**STARTING P0 ATTEMPT — not frozen training recipe.** These files are preparation examples, not server-ready configs. No GPU pilot has run. See the complete [field review and blocker analysis](../../../docs/pilot_config_review.md).

Common first-pilot choices: seed/head seed **3406** (split stays 406), `linear_default` head initialization, AdamW betas `[0.9,0.999]`/eps `1e-8`/weight decay `.05`, P0 LR `1e-4`, P0 augmentation/clipping disabled, FP32 without AMP/scaler/TF32/compile, strict deterministic policy. Loader: 2 workers, 2 main threads, pin/persistent workers on, prefetch 2. These are declared policy choices, not measured performance claims or final official seeds/recipe.

Before CUDA process startup, common environment policy: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=3406`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. JSON does not set these environment variables. Confirm allocation ≥4 CPUs and target environment first. FP32 results do not predict AMP memory/throughput; changing policy requires common review and new comparable measurements.

| Files | Declared inputs | Meaning |
|---|---|---|
| `p0_exploratory_a_e1.example.json`, `p0_exploratory_a_e1hr.example.json`, `p0_exploratory_a_e2.example.json` | measurement_only; microbatch 2, accumulation 1; warmup 5 + measure 20; full_epoch=false | First finite diagnostic attempt; no feasible-maximum claim |
| `p0_exploratory_b_e1.example.json`, `p0_exploratory_b_e1hr.example.json`, `p0_exploratory_b_e2.example.json` | measurement_only; microbatch 1, accumulation 2; same windows | Manual next attempt **only after A GPU OOM**, with separate config/run/protocol ID |
| `p0_comparable_final_e1.example.json`, `p0_comparable_final_e1hr.example.json`, `p0_comparable_final_e2.example.json` | criteria; actual batch pending; proposal warmup 10 + measure 100; full_epoch=true | Fresh common-effective-batch evidence after measurement/review, not current final configurations |

E1-HR exploratory WH `[1344,576]` is **only a candidate**. A/B have a one-candidate ordered list, a matching per-attempt reviewed batch table, and no resolution failure predecessors. OOM A is not eligible evidence to lower resolution under B's different batch protocol. If B also OOMs, stop for user review; do not invent or automatically run a lower resolution.

Unfilled mandatory inputs:

- `paths.data_root`, `paths.weights`: actual server paths, or CLI/environment overrides.
- `expected_weight_sha256`: real file exists → full hash → user reviews → same digest in all P0/P1 runtime configs. Remains null here.
- `gates.conformance_report`: passing target-environment train/val integrity and train-only preprocessing conformance report.
- `feasibility` thresholds: remain null. Exploratory examples explicitly use `measurement_only`, so these nulls no longer block first diagnostics. Comparable-final explicitly uses `criteria` and cannot run until legitimate thresholds are reviewed/filled.
- Comparable examples additionally require the reviewed actual microbatch/accumulation, common effective batch, HR batch table and HR candidate confirmation after P0 evidence.

Inactive nulls are intentional: Linear init `std/bias`, clipping max norm, and scaler growth fields are N/A with their selected methods/flags. They do not require hardware input or block config check.

Copy the selected example to ignored `configs/pilot/local/` and preserve each attempt/failure under a distinct run ID. Do not launch examples directly or use unreviewed thresholds just to pass validation. A readiness check only, **not a job launch**:

```bash
python -m src.pilot.p0_feasibility --config configs/pilot/reviewed/p0_exploratory_a_e1.example.json --check-config
```

This command is **expected to fail now** on mandatory placeholders. Do not interpret an expected config failure as a training/numerical failure. Conformance uses the existing `configs/pilot/conformance.json` and does not require weights.

Measurement-only success is `measured`/`not_assessed`, never a pass; OOM/infrastructure/numerical failures stay fail. **All measurement-only reports have fallback_eligible=false**, including OOM, and cannot gate P1 or export a frozen recipe. Mode is bound into context/fingerprint; missing or changed mode evidence fails strict consumers.

Workflow: UIT conformance → measurement-only exploratory P0 → review actual memory/throughput/epoch constraints → freeze legitimate thresholds/common batch → fresh strict comparable P0 E1/E1-HR/E2 → only strict full-epoch passes may gate P1. Keep epoch timing separate from short-window throughput. Comparable-final windows must fit one train iterator: `10+100 <= ceil(ceil(34997/microbatch)/accumulation)`; otherwise review a common shorter finite window before those runs. Do not promote exploratory reports, manually change result status, auto-derive thresholds or carry trained weights into comparable models.

Original P0 templates now have explicit `feasibility.mode=null`; P1 templates are unchanged. Full P1 configs, official four-LR grid, final seed IDs, final horizon and E3 are deferred. This patch implements only measurement mode and strict evidence guards; no GPU job has run and real server paths/hash/conformance are still required.
