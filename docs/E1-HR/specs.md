# E1-HR — Controlled Full-Image Resolution Ablation

**DRAFT FOR HR-G0 REVIEW — specification only; no implementation or execution authorization.**

Revision date: **2026-11-02**. E1 train/validation is closed; E1-HR has no official results. The user's rewrite instruction fixes the inherited controls below for E1-HR. Approval of this written specification, its HR-specific implementation artifacts, and execution remain separate gates. This revision creates only this document.

## 1. Experimental question and source authority

**Does increasing full-image resolution from 896×384 to 1344×576 improve classification on the same fixed benchmark when all other scientific controls remain unchanged?** E1-HR is a controlled resolution ablation of E1, not a new representation architecture. Dimensions are width × height; the HR model input is `[N,3,576,1344]`.

| Aspect | E1 | E1-HR |
|---|---|---|
| Full-image input | One global view, 896×384 | One global view, **1344×576** |
| Architecture | ConvNeXt-Tiny; 768-d representation; Linear(768,124), bias=True | Identical |
| Scientific controls | Completed E1 protocol | Inherited unchanged under Sections 2–4 |
| Selected base LR | Determined by E1 tuning | Determined independently by the HR campaign, Section 5 |

Both dimensions increase by 1.5, giving 2.25 times the canvas pixels. This is a geometric calculation, not a measured compute or runtime ratio. E1-HR is not compute-matched to E2. Local views, fusion, alternative backbones, class balancing, geometry/augmentation searches, additional LR candidates and E3 feature caching are outside scope.

Sources, in order of applicable authority:

- The user's current rewrite decisions establish inheritance and override the previous draft's unresolved adoption proposals and narrower final-test gate.
- [Completed E1 specification](../E1/specs.md), [numeric contract](../E1/e1_numeric_contract_v1.json), [T00 decisions](../E1/t00_review.md) and [closeout review](../E1/closeout_review.md) define the inherited scientific controls and verified baseline. Contract reference SHA-256: `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143`. Older NEEDS REVIEW/preparation text is historical, not a reopening of completed decisions.
- [Master specification](../master_spec.md) and [design record](../grillme.md) establish the experiment's role, independent initialization and equal campaign protocol.
- [Frozen preprocessing specification](../preprocessing_v1.md), [configuration](../../configs/preprocessing/geoguessr_v1.json), [implementation](../../src/preprocessing/geoguessr_v1.py) and [split provenance](../../artifacts/baseline124/provenance.json) remain the deterministic input authorities.
- Archived HR [P0 configuration](../../reports/pilot/evidence/uit_2026-10-31/p0/e1hr/config_resolved.json) / [summary](../../reports/pilot/evidence/uit_2026-10-31/p0/e1hr/run_summary.json) and [P1 configuration](../../reports/pilot/evidence/uit_2026-10-31/p1/e1hr/config_resolved.json) / [history](../../reports/pilot/evidence/uit_2026-10-31/p1/e1hr/history.json) / [summary](../../reports/pilot/evidence/uit_2026-10-31/p1/e1hr/run_summary.json) support resolution feasibility and operational planning only. They are not official HR results, current acceptance or a convergence claim.

## 2. Inherited frozen scientific controls

These are **requirements, not proposals or unresolved adoption choices**. Only a **material HR target-acceptance failure that prevents the frozen E1-HR protocol from being executed correctly** may trigger explicit user review of an inherited control; a failure grants no automatic fallback. Minor operational failures do not reopen scientific controls. Implementation convenience, spare memory, elapsed walltime or observed validation scores cannot change them.

| Control | Fixed E1-HR requirement |
|---|---|
| Data / labels | Exact `baseline124-v1` split and ordered 124-label mapping; no regenerated membership |
| Model | TorchVision ConvNeXt-Tiny; preserve features, global average pooling and final LayerNorm; new `Linear(768,124,bias=True)`; 124 raw logits |
| Starting point | Fresh pinned ImageNet weights for every trial/final seed, with fresh optimizer/scheduler; no E1, pilot or tuning-checkpoint warm-start |
| Fine-tuning | Entire model trainable from epoch 1; no classifier-only phase or LLRD; head LR multiplier 1 |
| Stochastic depth | TorchVision 0.23.0 block-index linear schedule 0→0.1; disabled in evaluation |
| Sampling / loss | Natural shuffled train permutation without replacement; standard unweighted CE; label_smoothing=0; no balancing/class weights |
| Augmentation | Brightness/contrast amplitudes 0.10/0.10, independent factors in [0.9,1.1], brightness then contrast on valid scene pixels; validation augmentation OFF |
| Batch | **2 original images effective batch; microbatch=2; gradient accumulation=1; drop_last=False** |
| Epoch accounting | 34997 train images, 17499 microbatches/optimizer updates, retained one-image tail; all 7497 validation images in fixed manifest order without shuffle |
| Scientific budget | **8 training epochs, 8 full validations, 139992 optimizer updates; early stopping OFF** |
| AdamW | betas `[0.9,0.999]`, eps `1e-8`; clipping OFF; all groups share the candidate LR/schedule |
| Weight decay | 0.05 on Conv/Linear weights; 0 on biases, LayerNorm and ConvNeXt layer-scale parameters; every trainable tensor assigned exactly once |
| New-head initialization | `linear_default`: `kaiming_uniform_(a=sqrt(5))`; bias `U(-1/sqrt(768),+1/sqrt(768))`; isolated CPU RNG seeded by the official run seed, restoring the outer CPU state |
| Schedule | Update-based warmup/cosine specified below; same full horizon for every candidate and seed |
| Training IDs | Tuning seed **1001**; final seeds **2001/3001/4001**, all four distinct within HR; neither pilot 3406 nor split seed 406 is a training seed |
| Numerics / RNG / loaders | Exact deterministic FP32 policy and seed/worker/loader mechanics in Section 4 |
| Reporting | Unchanged metric definitions; arithmetic three-seed mean and sample std **ddof=1**, without pre-aggregation rounding |

Pinned weights: `torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, offline file `convnext_tiny-983f1562.pth`, SHA-256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`. Require the exact filename/full digest and strict loading; replace only the classifier. Do not download a substitute.

Using the same numeric seeds across architectures supports paired comparison; it does not reuse run state or imply bitwise-equivalent learning dynamics across resolutions. Artifact separation is defined in Section 7.

### Fixed schedule and loss accounting

For base LR L, zero-based optimizer update u, N=139992, W=1000 and floor f=1e-6:

```text
u < W:   LR(u) = L * (0.1 + 0.9*u/W)
u >= W:  LR(u) = f + (L-f)/2 * (1 + cos(pi*(u-W)/(N-W-1)))
```

Apply LR before update u; advance completed-update state only after the update. Update 0 uses 0.1L, update 1000 uses L, and update 139991 uses exactly 1e-6. Validation does not advance the schedule. A selected earlier checkpoint does not shorten training.

For an accumulation group with D actual images, divide summed CE by D, including the retained tail. With accumulation=1, the final one-image batch receives its true denominator. Report sample-mean train loss/validation NLL, not an unweighted mean of batch means. Nonfinite logits/loss/gradients or OOM fail the attempt and remain traceable; no silent sample/update skipping or scientific-policy substitution is permitted.

## 3. Fixed data and HR input construction

Reuse `artifacts/baseline124/{manifest.csv,summary.json,labels.json,class_support.csv}` byte-for-byte from source commit `33cb29d72974f1677efab0f56ed92360e5a9f712`. Label IDs are the ordered `labels.json` indices 0–123. Train has 34997 images/124 supported labels; validation has 7497/106; test metadata records 7502/111. Seed 406 never regenerates membership. Unknown layouts, missing/corrupt samples, label mismatches and root escapes fail rather than being skipped or replaced.

Use frozen preprocessing ID `geoguessr-ui-global-local-v1`; config SHA-256 `4f49dd41738168cfc65d4322fbe1b659ef5595229bb487ad86dd671a26a8903f`, module SHA-256 `4d1dea328f69c1c7e1fa4f333c7276de0c3bc81b322db40dc33530969357e666`.

Construct HR directly from the decoded raw 1536×662 or 1030×703 image through the existing `Preprocessor(experiment="e1hr", global_wh=(1344, 576))` mode. Preserve RGB conversion without EXIF transpose, footer/UI coordinates, valid-scene photometric formulas, float32 ImageNet-mean fill, CPU float32 tensor bilinear resize with antialias, `floor(value+0.5)` rounding, aspect-preserving centered letterbox and final ImageNet normalization. Use only the global output `[3,576,1344]`.

Do not upscale an already constructed E1 image/canvas, rescale UI templates, duplicate coordinate tables, requantize through uint8, re-mask resized edges or substitute `weights.transforms()`/PIL/System.Drawing resize. Resolution is the only input-policy change. The primary access restriction is Section 8.

## 4. RNG and deterministic runtime

Inherit E1's exact streams: `random.seed(seed)`, NumPy seed `seed % 2**32`, `torch.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`. Derived seeds hash UTF-8 pipe-separated `str(seed)` followed by key parts with SHA-256; interpret the first eight bytes little-endian modulo `2**63-1`. Do not use Python's built-in hash or add an architecture key that changes those streams.

| Stream | Fixed mechanics |
|---|---|
| Epoch order | Fresh CPU generator keyed `(seed, zero_based_epoch, "order")`; `torch.randperm(dataset_length)` |
| Jitter | Fresh CPU generator keyed `(seed, zero_based_epoch, manifest_relative_path, "jitter")`; two sequential scalar uniform draws, brightness then contrast |
| Loaders | Separate CPU generators keyed `(seed, "loader", "train")` / `(seed, "loader", "val")`; save/restore their states |
| Workers | Torch worker seed comes from the loader generator; seed worker Python/NumPy with `torch.initial_seed()%2**32`; carry epoch with sampled indices for persistent workers |

FP32 is mandatory; AMP, GradScaler, TF32 and torch.compile are OFF. Set matmul precision `highest`, cuDNN benchmark=false/deterministic=true, deterministic algorithms=true/warn_only=false. Use num_workers=2, pin_memory=true, persistent_workers=true, prefetch_factor=2 and CPU threads=2. Verify effective settings and reject TF32-enabling overrides.

Before Python startup, export `PYTHONHASHSEED=<official run seed>`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The inherited software policy is Python 3.11.17, Torch 2.8.0+cu128, TorchVision 0.23.0+cu128, Pillow 11.3.0, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, CUDA 12.8 and cuDNN 91002. Actual HR acceptance must demonstrate this policy on the current target; recording these versions is not evidence that acceptance has run.

## 5. Independent official LR campaign

Run exactly `[2.5e-5, 5e-5, 1e-4, 2e-4] × seed 1001`, each to eight epochs/eight full validations. HR's selected LR is unknown until this campaign completes; **do not inherit E1's selected LR**.

Within each run, select the first completed epoch initially, then replace it only when `current_metric - selected_metric > tau`, with `tau=1e-6` and supported-validation Macro-F1 on the [0,1] scale. Otherwise retain the earlier epoch. NLL and Top-k do not break ties. Every reported metric uses that same selected checkpoint.

Across all four valid completed trials, let M be the global maximum selected-checkpoint score. Eligible candidates satisfy `M-score <= 1e-6`; choose the lowest base LR among them. Trial order and sequential pairwise tolerance must not change the outcome. Publish all candidates, selected epochs/scores, checkpoint lineage and failures/retries. A failed or incomplete candidate cannot be dropped, replaced by a fifth LR, or counted as equal-budget success; if it cannot complete validly, stop for explicit user review.

**After LR selection, STOP and hand off the complete tuning ledger and locked HR selection for explicit final-seed authorization.** Only then execute the selected configuration with seeds 2001/3001/4001, each under the original eight-epoch budget. Do not cherry-pick a seed, add seeds or ensemble logits in place of the specified aggregate.

## 6. Slurm attempts, checkpointing and separate target acceptance

The deployment reference is UIT Slurm: partition `normal`, account `uit`, QoS `mig2-35`, GRES `gpu:2g.35gb:1`, one task, four CPUs, 16384 MiB RAM and **09:00:00 per attempt**. Use the reviewed Python environment at `/datastore/binhvp/BuiTrongTan/CS406/envs/cs406/bin/python` and target class NVIDIA H200 MIG 2g.35gb. Actual allocation-policy compatibility must be reviewed; this spec invents no longer walltime or specific MIG UUID.

Archived HR P0 measured **4473.22 seconds per full training epoch**, excluding validation/checkpoint I/O. Eight times that measurement is approximately **9.94 hours of training alone**—an extrapolation, not an observed official run. The three-epoch P1 evidence also supports planning for multiple attempts. **A nine-hour allocation is not a promise that one official HR run will finish.**

- One Slurm job attempt may execute only one declared run; one declared run may require multiple attempts. No architecture/LR/batch loop belongs inside an attempt.
- Walltime limits an operational attempt, not the scientific horizon. Resume uses the **same immutable declaration and its own `last.pt`**. Commit boundary zero and completed-epoch boundaries atomically with a single writer; failed writes preserve the previous boundary.
- The committed training boundary is the **completed-epoch boundary**, with boundary zero retained for initial recovery. If an attempt terminates during an epoch, discard that uncommitted epoch and replay it **from its beginning**, restoring the previous committed `last.pt`. Preserve model/optimizer/full scheduler state, inactive-scaler policy, Python/NumPy/CPU/all-CUDA RNG, loader generators, epoch/order, history and selected-checkpoint weights/state. Reconstruct persistent workers without advancing logical loader RNG beyond the uninterrupted reference.
- Never reset/extend the **8-epoch / 139992-update budget**, restart warmup, count replay as additional completed work or resume from reporting-only `best.pt`. Archive attempt identities, interruptions, failures and reviewed retries; do not overwrite another run or silently replace a failed trial.

**HR requires its own explicitly authorized target acceptance; E1 receipts, E1 smoke job 8198 and archived HR pilot evidence do not authorize HR.** Acceptance must bind actual HR source, runtime/software, device/driver/allocation, current dataset root/content, frozen fingerprints and real pinned weights. Verify full train/val integrity/conformance for actual root/packages before model allocation, then the real train-only preprocessing/forward/backward/resume path **`[2,3,576,1344] -> [2,124]`**, finite logits/loss/gradients and actual memory records. Interrupted-state reconstruction/replay must match uninterrupted model/optimizer/scheduler/RNG/loader progression on the accepted stack. This is acceptance, not another optimization trial or pilot rerun.

## 7. HR identity and evidence contract

Every HR declaration, resolved configuration, checkpoint and report must explicitly bind:

```text
experiment = E1-HR; resolution_wh = [1344,576]
stage; seed; base LR; original epoch/update budget
HR contract hash; accepted source identity; actual software/runtime/device identity
fixed data, preprocessing and pretrained fingerprints
declaration/config, checkpoint, history and selection lineage
```

Run IDs must identify HR explicitly, for example `e1hr-final-lr0.00010000-seed2001`; do not reuse E1 run IDs or rename its artifacts into HR evidence. Keep separate HR receipts/output roots. A reviewed HR versioned contract and exact-byte checksum sidecar must bind the inherited values without modifying the frozen E1 contract or embedding the contract's own checksum inside itself.

Preserve immutable declarations/configs, complete run summaries/eight-epoch histories, `best.pt`/`last.pt`, operation/attempt records, all four trial outcomes, locked selection, selected analyses, three-seed report and scoped checksum inventory. Verify declaration ↔ summary ↔ history ↔ selected checkpoint ↔ selection ↔ report consistency, and record `test_images_opened=0`.

HR entrypoint/schema/package and bounded reuse of E1 mechanics belong in the later implementation plan. Existing E1 validators pin the E1 contract: do not bypass them or rewrite E1 scientific code/frozen artifacts to accept HR. Do not create a generic permanent training framework or introduce a runtime dependency on pilot code.

## 8. Final-test gate

**Final test access is LOCKED / NOT AUTHORIZED throughout E1-HR specification, implementation, acceptance, tuning, final training and validation reporting.** Test membership/counts may be read as metadata; reject test-image access before opening files, and add no test evaluator in this stage. Do not use test information to choose resolution, LR, epochs, augmentation or subsequent experiments.

Final-test evaluation may be considered only after **all experiments intended for the final comparison—including E3 if it remains in the study—have frozen protocols/checkpoints and closed their intended training/validation work**, followed by separate explicit user approval of the final evaluation. HR closeout or any HR training gate cannot satisfy that approval.

## 9. Reporting and comparison

Use the same definitions as E1: Top-1/3/5 rank all 124 outputs, with ascending label ID for exactly tied unmodified logits; supported Macro-F1 averages full-split per-class F1 over 106 GT-supported validation classes; unsupported F1 is null; supported classes with no correct predictions receive zero. Weighted-F1 uses evaluation GT support, and NLL is diagnostic sample-mean CE.

Per-class metrics and bucket F1 retain false positives from the entire split, including other buckets. Buckets use unique fixed-split n_train: 1–19, 20–99, 100–499, ≥500 and the 1–2 subset. Bucket Top-k is GT-conditioned but ranks all outputs; zero-support metrics are null. Export overall/per-class/bucket tables, manifest-keyed predictions, confusion/most-confused pairs, histories/learning curves and selected-checkpoint lineage.

Report all three final seeds with arithmetic **mean ± sample std (`ddof=1`)**, keeping full machine precision and displaying percentages to four decimals without pre-aggregation rounding. Compare against the [official E1 three-seed aggregate](../../src/e1/e1_three_seed_validation_report_v1.json), not pilot results or the best E1 seed. Optional paired per-seed E1-versus-HR deltas use the matching IDs; three seeds alone do not establish significance.

Record geographic/panorama dependence limitations, extreme-tail sample scarcity, seed/sampling uncertainty and the validation-only scope. Efficiency is secondary measured information: label train-only versus validation/checkpoint I/O and full-job scopes, throughput in original images, and peak allocated/reserved memory. Do not infer compute matching from resolution or pilot timing. Official HR result fields remain `-` until verified outcomes exist.

## 10. Stage-specific work and approval gates

The scientific controls above are settled. What remains stage-specific is their HR contract serialization/provenance and hash, implementation isolation/entrypoint/schema, distinct run/output identities, actual target/allocation acceptance, multi-attempt operational evidence, and approvals for each execution stage. HR's selected LR and result hashes are campaign outputs, not missing defaults. A longer allocation is not assumed; any allocation-policy change requires review without changing the scientific budget.

| Gate | Required review / authorization | Current state |
|---|---|---|
| HR-G0 | Review this specification; authorize preparation of the HR implementation plan/task ledger and explicit contract binding | **OPEN — ready for review** |
| HR-G1 | Approve those written artifacts, freeze exact HR contract bytes/sidecar, select implementation method and explicitly authorize implementation | Not approved |
| HR target-work authorization | Explicitly permit HR GPU deployment/acceptance under Section 6 | Not granted |
| HR-G2 | Accept CPU/regression and authorized HR target evidence, actual environment identity and concrete four-trial declarations | Not started |
| HR-G3a | Explicitly authorize only the four HR LR trials after HR-G2 | Not granted |
| HR-G3b | Review the completed tuning handoff/selection, then explicitly authorize three HR final seeds | Not granted |
| Final evaluation | Separate study-wide authorization under Section 8 | Not authorized |

No scientific decision or missing local evidence blocks **HR-G0 specification review**. Later gates still need their stage artifacts and actual acceptance evidence. This document prepares no plan/tasks, runnable contract/config, implementation or job, and asserts no HR acceptance/result. Only a **material HR target-acceptance failure that prevents the frozen E1-HR protocol from being executed correctly** may trigger explicit user review of an inherited control; it cannot authorize unilateral changes. Minor operational failures do not reopen scientific controls.
