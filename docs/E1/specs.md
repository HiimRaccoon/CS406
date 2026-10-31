# E1 - ConvNeXt-Tiny Full-Image Baseline Specification

**DRAFT FOR USER REVIEW - not frozen; does not authorize implementation or official runs.**

Revision date: **2026-10-31**. This document completely replaces the previous E1 draft at repository commit **`7f7c37782a49a804c88ffd07d5334ac6ee2115a5`**. It incorporates the user's reviewed E1 decisions: a fixed eight-epoch training budget with early stopping disabled, explicit no-decay ConvNeXt layer-scale parameters, and disjoint tuning/final training seeds. Other unresolved official values remain **NEEDS REVIEW**; an implementation must not supply them through defaults.

This is a specification revision only. It creates no implementation code, runnable configuration, training recipe, Slurm job, or experiment. The rest of the repository remains unchanged.

## 1. Purpose, Source Authority, and Scope

E1 is the **124-label country/territory classification baseline**, using one full-image view at **896 x 384**. It provides the reference for E1-HR/E2 and, later, an E1 representation source for E3. Its research objective is to benchmark classification performance on the **existing exact fixed split** and analyze overall-accuracy versus long-tail-recognition trade-offs. There is no single winner metric for comparing methods.

Sources have distinct roles:

| Source | Authority for E1 |
|---|---|
| [Master specification](../master_spec.md), [design review record](../grillme.md) | Established research and architecture decisions; the master specification remains a draft |
| [Frozen preprocessing configuration](../../configs/preprocessing/geoguessr_v1.json), [preprocessing specification](../preprocessing_v1.md) | The deterministic preprocessing v1 contract; implementation convenience cannot change it |
| [Split provenance](../../artifacts/baseline124/provenance.json) and the four fixed artifacts | Exact split membership, label IDs, counts, and fingerprints |
| [Pilot plan](../pilot_plan.md), [post-execution configuration review](../pilot_config_review.md), [pilot report](../../reports/pilot/pilot_results.md) | Completed pilot gates, recorded measurements, and unresolved official-training adoption decisions |
| [Server runbook](../server_pilot_runbook.md), archived pilot configurations/history, and `src/pilot/` | Execution evidence and reference mechanics; not an automatically approved official E1 framework |

**Precedence for this revision:** the user's three newly reviewed E1 decisions override older statements in the referenced documents that permit performance-based stopping, leave layer-scale decay undecided, or leave tuning/final seed overlap undecided. Those historical documents are not modified by this task. All other fixed research, dataset, and preprocessing decisions remain in force.

Historical statements that the pilot was deferred, UIT execution had not occurred, or the E1-HR resolution was undecided describe earlier preparation stages. Current post-execution evidence records target conformance PASS, strict P0 PASS, and bounded P1 COMPLETE. No P1-v2 is required. Updating execution status does not change frozen preprocessing v1 or turn diagnostic pilot scores into official results.

**Outside E1 scope:** local views or fusion, E1-HR training, cRT/head retraining/feature caching, class balancing, LLRD, pretrained-checkpoint comparisons, strong augmentation, geometry searches, test-driven tuning, and robustness/unseen-location studies. This specification adds no experiment or ablation branch.

## 2. Decision Status

| Component | Status |
|---|---|
| Exact split, 124-label mapping, preprocessing v1 | FROZEN at their authoritative sources |
| E1 architecture, pretrained starting point, natural unweighted CE, maximum stochastic depth 0.1, head LR multiplier 1 | Established design decisions, documented in this draft for review |
| Optimizer decay groups | FIXED: decay 0.05 for Conv/Linear weights; zero decay for biases, LayerNorm parameters, and ConvNeXt layer-scale parameters |
| Brightness/contrast amplitudes 0.10/0.10 | REVIEWED / ACCEPTED through train-only visual review; used in P1 |
| Official training budget | FIXED: `max_epochs = 8`, full validation after every completed epoch, `early_stopping = OFF`; successful tuning and final runs complete all eight epochs |
| Checkpoint selection | FIXED: supported validation Macro-F1 over epochs 1-8, `tau = 1e-6`, tie-break `earlier`; independent of training termination |
| LR tuning/final protocol | Four LR candidates with one tuning seed, then the selected LR with three final training seeds |
| Seed-set relationship | FIXED: one tuning seed and three distinct final seeds; all four IDs are distinct, and the tuning seed is excluded from the final set |
| Exact four-LR values and exact numeric training seed IDs | NEEDS REVIEW; no values are selected in this revision |
| Official batch, runtime, initialization/RNG binding, AdamW betas/eps, clipping, and warmup/cosine details | NEEDS REVIEW as listed in Section 12; pilot values are references, not implicit official defaults |
| Specification approval, official implementation, smoke, tuning, final runs, and test evaluation | Not completed by this document |

## 3. Dataset, Labels, and Split Access

Reuse `artifacts/baseline124/{manifest.csv,summary.json,labels.json,class_support.csv}` byte for byte. Their original repository is `HiimRaccoon/cs406-geoguessr`, source commit **`33cb29d72974f1677efab0f56ed92360e5a9f712`**. Do not regenerate membership from split seed **406**, re-stratify, deduplicate again to alter membership, or reorder the label mapping.

| Split | Unique images | GT-supported labels | Access during E1 development and tuning |
|---|---:|---:|---|
| train | 34997 | 124 | Training under natural sampling; train-only diagnostic samples |
| val | 7497 | 106 | Full deterministic validation; checkpoint and LR selection |
| test | 7502 | 111 | Metadata only; image access remains locked until separately authorized final evaluation |
| Total | 49996 | 124 outputs | Exact fixed membership |

The raw dataset contains 49997 images; **one byte duplicate** was removed. Original grouping was based only on SHA-256. Geographic independence and panorama-disjoint membership are not established. Do not remove extreme-tail classes from training or evaluation. E1 always produces **124 logits**, including labels without ground-truth support in the validation or test split.

`labels.json` is an ordered list: its **index is the label ID**, from 0 through 123. Mapping checks include `0=Aland`, `120=United States`, and `123=Vietnam`. Preserve the country/territory names exactly. The runtime data root directly contains the label folders; manifest-relative paths remain unchanged. The summary's `data_root_at_preparation` records a historical path, not a required UIT runtime location.

Before allocating the model, verify:

- The four artifacts' full SHA-256 digests and byte counts against provenance and the frozen configuration. Summary fingerprints, split counts, class coverage, label IDs/names, class support, unique paths, and unique image hashes must agree.
- Dataset-root resolution and path containment. Missing/corrupt files, invalid decoded layouts, or label mismatches must produce a recorded failure; never silently skip or replace a sample.
- A valid target conformance/hash-audit record for the actual dataset root and package environment, covering **34997 train plus 7497 validation images** without reading test bytes. Keep the dataset immutable after the audit; repeat the audit when root/content changes, not every epoch.
- Train and validation loaders reject `split=test` before opening a file. Reading test counts or membership from metadata does not authorize opening test images. A final test evaluator belongs to a later, separately authorized phase.

The full artifact hashes are maintained in [provenance](../../artifacts/baseline124/provenance.json) and `source_split.sha256` in the frozen preprocessing configuration, rather than duplicated as a second independent hash table here.

Split seed **406** controls the already-fixed dataset membership and is unrelated to training-seed selection. It must not be used as an implicit training-seed default or as a reason to create new membership.

## 4. E1 Input and Preprocessing

Preprocessing ID: **`geoguessr-ui-global-local-v1`**. Configuration SHA-256:

`4f49dd41738168cfc65d4322fbe1b659ef5595229bb487ad86dd671a26a8903f`

E1 uses only the **global view** from the [reusable preprocessing module](../../src/preprocessing/geoguessr_v1.py). It does not construct local views or use preview PNGs as model inputs. Do not use `weights.transforms()`, a 224-pixel center crop, aspect-ratio distortion, or an alternative resize backend.

The per-image sequence is:

1. Decode RGB using Pillow without EXIF transposition. Accept only decoded raw layouts `1536 x 662` and `1030 x 703`.
2. Convert through `pil_to_tensor` to CPU RGB CHW **float32 in [0,1]**. No intermediate uint8 requantization or preprocessing AMP is allowed.
3. Remove the bottom **14-pixel** footer. Construct the valid-scene mask from the frozen UI rectangles. Intentionally retain navigation arrows under the reviewed exception.
4. During training, apply one brightness/contrast draw to valid scene pixels, with brightness before contrast. Exclude UI pixels from photometric statistics and fill UI **after** jitter with the exact mean. Validation and final test preprocessing have no jitter.
5. Preserve aspect ratio and letterbox to **WH=[896,384]**, using CPU float32 tensor bilinear resize with `antialias=True`. Use half-up rounding `floor(value+0.5)`, centered placement, and any extra padding pixel on the right/bottom.
6. Normalize after resize/padding using ImageNet mean **[0.485,0.456,0.406]** and standard deviation **[0.229,0.224,0.225]**, with `inplace=False`.

UI and letterbox fills use the exact float32 mean, never a rounded uint8 color. Do not reapply a hard mask after resize. Constant mask interiors and padding are approximately zero after normalization; antialiasing may blend scene content at boundaries.

| Raw layout | Frame after footer removal | Resized content | Padding L/T/R/B | Output CHW |
|---|---|---|---|---|
| 1536 x 662 | 1536 x 648 | 896 x 378 | 0/3/0/3 | [3,384,896] |
| 1030 x 703 | 1030 x 689 | 574 x 384 | 161/0/161/0 | [3,384,896] |

Use the frozen configuration for mask coordinates and photometric formulas; E1 must not introduce a separately maintained coordinate table. Navigation arrows and some watermarks intentionally remain. Do not describe preprocessing as removing every UI overlay.

### Augmentation

Training brightness and contrast amplitudes are **0.10/0.10**, placing each factor in **[0.90,1.10]**. Sample factors per original image. Retaining the pilot's two independent uniform draws is proposed for official adoption under R7; the official RNG binding remains subject to review. Use the frozen valid-scene brightness/contrast formulas, not an implicit `ColorJitter` implementation with different ordering or statistics.

Validation/test preprocessing is deterministic, without test-time augmentation. The main E1 recipe excludes flips, random crops, RandAugment, Random Erasing, Mixup/CutMix, and label smoothing.

Reusing the pilot RNG key `(training_seed, epoch, relative_path, "jitter")` and epoch-order seeding is proposed so that jitter does not depend on persistent-worker scheduling. **Official RNG adoption remains NEEDS REVIEW in Section 12.** Do not derive sample seeds from Python's process-dependent built-in hash.

## 5. Model Contract

```mermaid
flowchart LR
    A[Raw RGB screenshot] --> B[Frozen UI preprocessing + E1 global 896 x 384]
    B --> C[ConvNeXt-Tiny features]
    C --> D[Global average pooling + pretrained final LayerNorm]
    D --> E[Flatten: 768-dimensional representation]
    E --> F[New Linear 768 to 124, with bias]
    F --> G[124 raw logits]
```

| Boundary | Contract |
|---|---|
| Model input | Batched floating tensor [N,3,384,896]; one view per original image |
| Feature extractor | Pretrained ConvNeXt-Tiny features; no block/depth/channel redesign |
| Representation | Retain pretrained global average pooling and final LayerNorm, then flatten to [N,768] |
| Classifier | New Linear(768,124), bias=True; do not retain the ImageNet 1000-class linear head |
| Forward output | [N,124] raw logits; do not apply softmax before CE |
| Trainable parameters | The entire representation and new classifier from epoch 1 |
| Stochastic depth | Explicit maximum **0.1**, using the selected TorchVision block-index linear schedule from 0 to 0.1; no separate rate/schedule tuning |
| Evaluation mode | Disable stochastic depth and training randomness for validation/final inference |

The common starting point is **`torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`**, filename **`convnext_tiny-983f1562.pth`**, with the reviewed full SHA-256:

`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`

Check the exact filename, file existence, and full digest before loading. The `983f1562` prefix or a default weight alias is not sufficient. Do not download a different checkpoint when the file is unavailable. Strict-load the pretrained state, preserve the pretrained representation/final LayerNorm, and replace only the classification linear head.

Every tuning trial and final run starts independently from the same ImageNet file with fresh optimizer and scheduler state. Do not warm-start from P0/P1, another LR trial, or an older GeoGuessr checkpoint.

New-head initialization must be explicit, deterministic, and recorded with its exact rule, framework version, and seed behavior. The pilot used `linear_default`: `kaiming_uniform_(a=sqrt(5))`, bias `U(-1/sqrt(768),+1/sqrt(768))`, and an isolated CPU RNG whose outer state is restored. **Keeping this rule is proposed; official adoption and seed binding remain NEEDS REVIEW.** Do not hard-code the pilot initialization seed 3406.

## 6. Loss and Optimization Invariants

- Use **standard unweighted CE** on the natural training distribution, with `label_smoothing=0`. Do not use class weighting, `sqrt_inverse`, oversampling, or balanced sampling. Long-tail treatment is intentionally isolated to E3.
- Every completed training epoch visits all **34997 unique train images exactly once**, shuffled by the reviewed training-seed/epoch policy, without replacement. Retain the tail batch with `drop_last=False`. Validation covers all 7497 images without shuffle or augmentation.
- Use **AdamW** with the fixed decay groups below. Every trainable parameter must belong to exactly one explicitly enumerated optimizer group.
- Use `LR_head = LR_backbone`, multiplier **1**, with no LLRD and no classifier-only warmup phase. Tune only one base LR per architecture; all parameter groups share its scheduler trajectory.
- Use **LR warmup followed by cosine decay**, scheduled by optimizer updates. AdamW betas/eps, clipping/numerics, and the unresolved warmup/cosine details must not come from unreviewed library defaults or silent pilot copying.

### Fixed weight-decay groups

| Parameter category | Weight decay | Status |
|---|---:|---|
| Trainable Conv weights | 0.05 | FIXED |
| Trainable Linear weights, including the new classifier weight | 0.05 | FIXED |
| Biases | 0.0 | FIXED |
| LayerNorm parameters | 0.0 | FIXED |
| ConvNeXt layer-scale parameters | 0.0 | FIXED |

ConvNeXt layer-scale parameters remain trainable under full fine-tuning but **always belong to the no-decay group**. This is an E1 optimization invariant, not a tunable option or unresolved adoption decision. Record the actual parameter-name enumeration and verify that no parameter is duplicated or omitted when module names differ from the pilot implementation. Any unexpected parameter category must be reported rather than assigned through an implicit fallback policy.

Training CE and validation NLL are sample means, not unweighted averages of batch means. Nonfinite loss/logits/gradients or OOM must fail and be reported; never silently skip an update, omit a batch, or change microbatch/resolution/LR while reporting the same run as successful.

### Batch and optimizer-update accounting

For one GPU, microbatch `m` and accumulation `a` give nominal effective batch **B=m*a original images**. An accumulation group with `D` actual images uses summed CE divided by **D**, including short tail microbatches/groups. Perform optimizer/scheduler steps only after the group is complete. Validation does not advance the scheduler.

For a natural epoch with retained tails:

```text
microbatches_per_epoch = ceil(34997 / m)
updates_per_epoch = ceil(microbatches_per_epoch / a)
planned_total_updates = updates_per_epoch * 8
```

The **pilot** used `m=2,a=1,B=2`. **If** official review adopts that batch, the count is 17499 updates per epoch and **139992** updates for a successful eight-epoch run. This is a conditional calculation, **not a newly frozen official batch**. The P1 horizon of **52497** updates must not be reused as an official eight-epoch horizon.

Future E1-HR/E2 comparisons must retain the reviewed common effective batch, LR-trial budget, and compatible training protocol through their own specifications. This E1 revision does not implement those methods or authorize silently changing their settings.

## 7. Official Run Protocol

### 7.1 Preflight and smoke

Before official execution, the specification must be reviewed, the required numeric contract must be frozen/versioned/hashed, dataset/preprocessing/pretrained/environment identity must be verified, and implementation/smoke acceptance must be reviewed separately. Smoke uses synthetic inputs or train-only samples to check wiring, numerics, and resume behavior. It has no test access and is neither a final seed nor a winning tuning trial.

Completed pilot execution does not require P1-v2. If an implementation/environment change requires an acceptance check, verify the affected contract and record the evidence without opening a new architecture or optimization search.

### 7.2 Fixed budget and validation cadence

**Official E1 uses `max_epochs = 8`, full validation after every completed epoch, and `early_stopping = OFF`.** Every LR trial and every final training run that is reported as complete must finish all eight training epochs and all eight validation passes.

All four LR trials receive the same full eight-epoch budget. Validation performance cannot terminate a run, shorten a trial, extend training, or change its remaining allocation. A checkpoint selected from an earlier epoch does not reduce the completed training budget.

A numerical/infrastructure failure must be recorded as a failure. A job interruption may be resumed from its saved boundary, but an interrupted prefix is not a completed trial or final run. Such operational conditions do not authorize a validation-based stopping rule or a shorter success definition.

### 7.3 LR tuning

Run **four distinct positive base LR candidates with one official tuning seed**. All trials share the fixed eight-epoch budget, reviewed warmup/cosine details, checkpoint rule, effective batch, optimizer/runtime policy, and initialization/RNG contract. The exact grid and numeric tuning seed remain NEEDS REVIEW.

Freeze the common four-value grid for E1/E1-HR/E2 before official search. Do not add an E1 trial because the observed score is unsatisfactory. Every trial starts independently from the pinned ImageNet weights.

For each completed trial, select its checkpoint using Section 8, then select the LR using **supported validation Macro-F1 at that selected checkpoint**. Publish all four LR values, outcomes, selected epoch/metrics, and full histories, including unsuccessful or unselected trials. The **LR-trial tie rule remains NEEDS REVIEW** and is separate from the already-fixed checkpoint tie-break.

Record failures/retries with their causes. Do not erase failures from the ledger, silently substitute a fifth LR, or present incomplete trials as equal-budget completed comparisons.

### 7.4 Disjoint training seeds and final training

Freeze the numeric IDs before official runs under this **fixed seed-set rule**:

- Exactly **one official tuning seed**.
- Exactly **three distinct official final-training seeds**.
- **All four seed IDs are distinct**: the tuning seed must not appear in the final-seed set.
- Pilot seed **3406 is not automatically reused**; using any pilot ID would require an explicit reviewed numeric selection under the same disjointness rule.
- Split seed **406 remains unrelated to training seeds** and never regenerates or changes membership.

The exact numeric IDs remain **NEEDS REVIEW**. Their disjointness is already frozen and must not be presented as an open choice.

After LR selection, freeze the selected configuration and train it from scratch under the three final seeds. The protocol is **four LR candidates x one tuning seed, followed by selected LR x three final seeds**, not four LRs x three seeds. Final runs start from the same ImageNet file with fresh optimizer/scheduler state, never from the winning tuning trial or a pilot checkpoint.

Final seeds do not reopen LR/grid/augmentation decisions. They change only initialization/optimization/batch-order/augmentation randomness under the reviewed seed contract. Do not cherry-pick the best seed using test results or count the pilot as a final run. Each successful final run completes the full eight-epoch budget even if its selected checkpoint is earlier.

Test images remain locked throughout specification work, implementation, smoke, tuning, and training/validation. Final test evaluation is a separate authorized phase after protocol/checkpoints are locked. Do not refit on train plus validation because that would change the fixed benchmark protocol.

## 8. Checkpoint Selection and the Fixed Scheduler Horizon

**Training termination and checkpoint selection are independent.** Every successful official run completes epochs **1-8**. Full validation follows each completed epoch. The reported model is `best.pt`, selected by **Macro-F1 over ground-truth-supported validation classes**, using **`tau = 1e-6`** and **tie-break `earlier`**.

| Rule | Reviewed requirement |
|---|---|
| Training budget | Eight completed training epochs and eight full validation passes |
| Performance-based early stopping | OFF; validation scores never stop training |
| Checkpoint metric | Supported validation Macro-F1 |
| Checkpoint tolerance | `tau = 1e-6` on the [0,1] metric scale |
| Checkpoint tie-break | `earlier`; validation NLL does not break checkpoint ties |
| Candidate checkpoints | Completed epochs 1 through 8, all considered under the same rule |
| Scheduler horizon | The full reviewed eight-epoch optimizer-update budget, independent of the selected epoch |
| LR-trial tie rule | NEEDS REVIEW; do not infer it from the checkpoint tie-break |

Use the chronological tolerance comparator consistently: select the first completed epoch initially; at each later epoch, replace the selected checkpoint only when `current_f1 - selected_f1 > 1e-6`. Within tolerance, retain the already-selected, earlier epoch. A lower F1 never replaces the selected checkpoint. Do not use NLL, Top-1, training loss, or test results to override this rule.

Here, "best" means the checkpoint selected under this fixed tolerance/earlier rule. An individual raw F1 peak within tolerance can be higher than the selected score; report the selected epoch/score and the full history rather than mixing metrics from different epochs.

All reported Top-1/Top-3/Top-5, Macro-F1, Weighted-F1, per-class, and bucket metrics for a run must come from **that same selected checkpoint**. `last.pt` exists for resume and is not automatically the reported model. Both checkpoints remain traceable even when they refer to the same final epoch.

Compute the scheduler's planned total updates from the reviewed batch accounting in Section 6 and the full **eight-epoch** budget before the run starts. Do not shorten the horizon to the selected best epoch, reuse the three-epoch P1 horizon, restart warmup after an interruption, or extend the budget through resume. Official warmup duration/start factor and cosine formula/floor details remain NEEDS REVIEW; the full eight-epoch horizon requirement itself is fixed.

Record planned/completed epochs and updates, the selected checkpoint epoch/score, full validation history, and the termination reason. A successful run terminates because the eight-epoch budget is complete, not because validation stopped improving.

## 9. Metrics and Error Analysis

Let `C[i,j]` be the count with ground-truth label `i` and predicted label `j` over the **entire evaluated split**. Compute each class's F1 from its complete TP/FP/FN counts, including false positives contributed by ground-truth samples in other classes or frequency buckets.

A GT-supported class with no correct predictions receives F1=0. A class with zero GT support has F1 recorded as **null/not evaluated** and is excluded from supported Macro-F1, rather than being inserted into that average as zero.

| Output | Definition |
|---|---|
| Top-1 / Top-3 / Top-5 | Sample accuracy ranking all **124 outputs**; never mask labels because the evaluation split lacks their GT support |
| Supported Macro-F1 | Mean per-class F1 over GT-supported classes in the split: **106** for validation and **111** for final test according to fixed metadata |
| Weighted-F1 | Per-class F1 weighted by **evaluation GT support**, not training frequencies |
| NLL | Unweighted mean evaluation CE; reported as a diagnostic, never used to select checkpoints or break their ties under the fixed F1/earlier rule |
| Per-class CSV | Label ID/name, unique train count/bucket, GT support, predicted count, precision, recall, and F1, explicitly identifying unsupported classes |

Report the main classification metrics together. Do not declare a single method winner from one scalar, average minibatch Macro-F1 values, or reduce output rankings to 106/111 classes. The LR-trial tie rule, if it uses any secondary information, must be separately reviewed before tuning.

### Frequency-bucket contract

Class membership is based on **unique n_train after deduplication and the fixed split**, never raw counts, validation/test support, or resampling frequency. The [class-support artifact](../../artifacts/baseline124/class_support.csv) establishes:

| n_train bucket | Total classes | Unique train images | Validation-supported classes | Validation samples |
|---|---:|---:|---:|---:|
| 1-19 | 42 | 243 | 24 | 49 |
| 20-99 | 37 | 2077 | 37 | 445 |
| 100-499 | 29 | 7049 | 29 | 1513 |
| >=500 | 16 | 25628 | 16 | 5490 |
| 1-2, a subset of 1-19 | 20 | 21 | 2 | 2 |

For each evaluated bucket, report total classes, GT-supported classes, evaluation samples, Macro-F1, and Top-1/Top-3/Top-5. Compute bucket Macro-F1 by averaging the supported bucket classes' per-class F1 values **already computed on the entire evaluation split**. Do not first filter GT rows and discard cross-bucket false positives.

Bucket Top-k uses samples whose GT belongs to that bucket, while ranking all 124 outputs. A bucket without GT support has null metrics, not fabricated zero scores. Keep membership fixed across methods and seeds.

The final E1 report includes each of the **three distinct final seeds** and their mean plus/minus standard deviation. The standard-deviation convention remains NEEDS REVIEW and must be stated before aggregation. Do not choose the best seed or replace individual-model score aggregation with an averaged-logit ensemble. Retain full validation histories/learning curves, confusion matrices, and most-confused pairs. Interpret results from the 1-2-image extreme tail cautiously.

## 10. Runtime, RNG, Resume, and Output Contract

### Runtime and reproducibility

The target is single-GPU UIT/Slurm execution, with the expected partition `normal` and GRES `gpu:2g.35gb:1`; GPU work belongs inside an allocation. Pilot records identify an H200 MIG 2g.35gb, Python 3.11.17, torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, and cuDNN 91002. Official runtime/environment adoption remains subject to review.

Record resolved package versions, driver/device/allocation metadata, process environment, and actual Torch settings, not merely the requested configuration. Keeping the pilot's deterministic FP32 runtime/loader policy is proposed in Section 12 as a single common adoption decision, not a runtime ablation. Do not change AMP, TF32, worker counts, or compile policy independently for one E1 trial.

Dataset/preprocessing conformance must match the actual target packages and dataset root. An archived pass does not replace the required check when those conditions change.

The official RNG contract covers Python, NumPy, CPU/CUDA RNGs, classifier initialization, epoch order, augmentation, and worker/DataLoader generators. Reusing the deterministic derived-seed mechanics from the [pilot dataset](../../src/pilot/dataset.py) is proposed, but the binding must use reviewed official IDs satisfying the **four-distinct-ID tuning/final rule**. Do not hard-code pilot seed 3406 or its `PYTHONHASHSEED` value. Persistent workers must receive the correct epoch; restarting a job must not repeat epoch-zero ordering or jitter.

### Checkpoint and resume

An official checkpoint has its own identity/type. Do not accept `kind=bounded_pilot_checkpoint` as official resume input. Write `last.pt` atomically at a completed epoch boundary, including model, optimizer, scheduler, optional AMP/scaler state, completed epoch/update counters, the fixed eight-epoch planned budget, full resolved configuration/hash, data/label/preprocessing/pretrained fingerprints, official source/environment identity, RNG/loader state, selected-checkpoint information, and full history.

Resume the same run from `last.pt` with unchanged seed/configuration/source/runtime identity and the **original eight-epoch budget**. Reject identity or budget mismatches. Do not resume from `best.pt` to roll back optimization and add budget, or warm-start from a pilot or another LR trial.

Replay an interrupted partial epoch from the last committed boundary. Log replayed operational work separately; it neither counts as an additional completed epoch nor authorizes training past epoch eight. Keep an initial boundary checkpoint for recovery before epoch one completes. Do not overwrite an existing run with a new trial. Enforce a single writer for each run folder.

Interruption is an operational state, not validation-based early stopping. Resume must complete the remaining part of the original full budget; it cannot change the selected-checkpoint rule, shrink the scheduler horizon, reset warmup, or extend training. Successful completion is possible only after all eight epochs and their full validations are recorded.

The supplied UIT bundle does not contain a separate interrupted/resumed diagnostic execution. Pilot checkpoint code and synthetic CPU tests provide references, not automatic evidence of official CUDA resume equivalence. Validate the reviewed runtime's behavior during authorized implementation acceptance without requiring P1-v2 or a new convergence study.

Pilot `identity()`/`validate_gates()` and checkpoint helpers are references. **Do not require the official source fingerprint to equal the pilot source fingerprint**: official implementation will have its own source identity. Keep pilot evidence references and official implementation identity separate, while validating the applicable data/preprocessing/pretrained/runtime contracts. This document does not approve an unsolicited permanent `src/data/`, `src/models/`, or `src/training/` refactor.

### Required artifacts after implementation is authorized

These are output requirements for review, not official run files created by this revision:

| Artifact | Required contents |
|---|---|
| Resolved configuration and recipe/spec reference/hash | Tuning/final stage, run/trial ID, official seed and its role, base LR, every numeric/runtime policy, fixed eight-epoch budget, early stopping OFF, and source/data/preprocessing/pretrained identity |
| Environment/model/optimizer metadata | Versions, GPU/allocation/environment, architecture/head/drop-path/initialization rules, enumerated decay/no-decay groups including no-decay layer-scale parameters, and parameter counts |
| `history.json`, `metrics.csv`, learning curves | Every completed epoch: train loss/sample count, updates, LR endpoints, validation metrics, timing, and memory; a complete run contains all eight epochs |
| `best.pt`, `last.pt` | Separate selection/reporting and resume roles; full state needed for their contracts |
| `run_summary.json` | Complete/failure/interrupted status, planned/completed budget, selected epoch/score, termination/failure reason, and test-access status; no validation-based shortened-success status |
| Tuning ledger | All four reviewed LR candidates and their outcomes, full-budget completion evidence, selected LR, and the reviewed trial-selection/tie rationale |
| Selected-checkpoint analysis | Validation overall/per-class/bucket metrics, predictions keyed by manifest path/label ID with Top-k, and confusion matrix; test outputs only during final evaluation |
| Three-seed report | Each disjoint final-seed run and mean plus/minus standard deviation, coverage, and limitations; pilot scores remain diagnostic |

Clearly distinguish train-only timing, training plus validation, checkpoint/report I/O, and full-job walltime. Report throughput in original images and peak allocated/reserved memory separately. P0 efficiency measurements are not official E1 score requirements; efficiency remains secondary information.

Later E3 work may use the pre-classifier representation from each **selected official final-seed E1 checkpoint**. Preserve identity/checkpoint lineage for that handoff. Do not cache features or implement E3 within this E1 revision.

## 11. Pilot Evidence and Its Interpretation

| E1 evidence | Recorded result | Limit |
|---|---|---|
| [Strict P0 summary](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/run_summary.json), [configuration](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/config_resolved.json) | pass, assessed, criteria, full epoch; microbatch/accumulation 2/1; train epoch 2080.53 s; steady-state 16.9549 img/s and full-epoch 16.8212 img/s; reserved peak 1.99023 GiB in the measured window and 1.99414 GiB over the epoch | Feasibility of the declared workload, not convergence or maximum feasible batch |
| [Bounded P1 configuration](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/config_resolved.json), [summary](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/run_summary.json) | complete 3/3 epochs, 52497 updates, diagnostic seed 3406 and LR 1e-4; official_experiment=false | Bounded diagnostic execution, not an official run or eight-epoch result |

P0 establishes feasibility only under the measured policy; it does not establish an optimal batch. P1 demonstrates bounded optimization behavior, not convergence or optimality of an eight-epoch budget. Both E1 summaries record `test_images_opened=0`; target conformance and all pilot profiles passed according to the [pilot report](../../reports/pilot/pilot_results.md). The evidence manifest excludes its former self-entry, and all **47 payload hashes PASS**.

The [P1 history](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/history.json) and [metrics CSV](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/metrics.csv) agree:

| Epoch | Train CE | Validation NLL | Validation Top-1 | Supported validation Macro-F1 |
|---|---:|---:|---:|---:|
| 1 | 1.956166 | 1.430387 | 0.615179 | 0.191439 |
| 2 | 0.990609 | 0.977231 | 0.727758 | 0.347943 |
| 3 | 0.378864 | 0.856378 | 0.761238 | 0.395442 |

The best diagnostic checkpoint is epoch three, with Top-3=0.898093, Top-5=0.936508, and Weighted-F1=0.747391 from the same epoch. Reserved memory stayed at 2141192192 bytes across all three epochs. E1 P1 data-wait fractions were approximately 16.4-19.2%; this is not evidence of optimal loader settings or authorization for a worker search.

Do not impose an implementation acceptance threshold requiring official E1 to exceed the pilot's 0.395442 Macro-F1. Correctness/protocol compliance and measured model quality are different questions. The diagnostic LR **1e-4** and pilot seed **3406** are not silently promoted to official values. No P1-v2 is required for this specification review.

## 12. Review Register Before Official Execution

The remaining proposals complete **one common recipe**; they add no experiment branch. No `configs/training/training_recipe_v1` is created, and a partial recipe must not be exported while required official fields are unresolved.

| ID | Item | Evidence or proposal for review | Status |
|---|---|---|---|
| R1 | Exact four distinct base LR values; LR-trial tie/failure accounting | Only diagnostic LR 1e-4 is established by the pilot. Do not invent a grid or refine it using pilot validation scores. The LR-trial tie rule is not the checkpoint tie rule. | NEEDS REVIEW |
| R2 | Exact one tuning ID and three final-training IDs | **All four IDs must be distinct**, and the tuning ID is excluded from the final set. This relationship is fixed; only the numeric IDs remain unresolved. Pilot 3406 is not automatically reused; split seed 406 is unrelated. | NEEDS REVIEW: numeric IDs only |
| R3 | Official common effective batch and E1 microbatch/accumulation | Retaining pilot B=2, m=2, a=1 is proposed because all three profiles passed strict feasibility at that batch. It is not automatically an official adoption. | NEEDS REVIEW |
| R4 | Warmup duration/start factor and cosine formula/floor details | Pilot values were 1000 warmup updates, start factor 0.1, floor 1e-6, and a three-epoch horizon. Review official details with the LR grid and batch. The official horizon must cover the full eight epochs and cannot follow the selected epoch. | NEEDS REVIEW: schedule details only |
| R5 | Full training budget and checkpoint rule | Eight epochs, full validation after every epoch, early stopping OFF; supported validation Macro-F1 over epochs 1-8, tau=1e-6, earlier checkpoint tie-break. These are reviewed invariants, not unresolved choices. | FIXED; verify implementation compliance |
| R6 | AdamW betas, AdamW eps, and gradient clipping | Pilot betas [0.9,0.999], eps 1e-8, and clipping OFF are proposals. Weight decay 0.05 for Conv/Linear weights and zero decay for biases/LayerNorm/ConvNeXt layer-scale are already fixed and excluded from this review. | NEEDS REVIEW: betas/eps/clipping only |
| R7 | New-classifier initialization and official RNG/worker/resume binding | Retaining linear_default with isolated CPU RNG and sample/epoch-keyed jitter/order is proposed. Bind only to the reviewed, disjoint official IDs and verify full-budget resume behavior. | NEEDS REVIEW |
| R8 | Official runtime/environment and Slurm/output adoption | Proposed pilot policy: FP32; AMP/scaler/TF32/compile OFF; highest matmul precision; cuDNN benchmark OFF; deterministic algorithms ON, warn_only=false; workers=2, pin/persistent=true, prefetch=2, threads=2; recorded target packages. Review environment variables/allocation/walltime without copying the pilot seed. | NEEDS REVIEW |
| R9 | Reporting convention and final-evaluation handoff | Review the standard-deviation convention, any remaining reporting conventions, and final test-unlock ownership. Checkpoint tau/earlier and seed disjointness are fixed; LR-trial ties remain under R1. Test stays locked during training. | NEEDS REVIEW |

**Resolved decisions must not be reopened as implicit searches:** ConvNeXt-Tiny full-image E1, 896 x 384 input, exact split/labels, frozen preprocessing v1, exact ImageNet checkpoint/full hash, full fine-tuning from epoch one, natural unweighted CE, maximum stochastic depth 0.1, the explicit decay/no-decay groups, head LR multiplier 1, brightness/contrast 0.10/0.10, full eight-epoch budget with early stopping OFF, checkpoint tau=1e-6/earlier, and disjoint one-tuning/three-final seed sets.

Architecture and functional requirements may be reviewed before every numeric value is chosen. Official LR search/final runs still require the applicable open items R1-R4 and R6-R8 to be resolved into a complete versioned numeric contract, with relevant R9 conventions established before aggregation/evaluation. **R5 is already fixed.** Do not let `null`, "use default," or pilot fallbacks resolve required official fields.

## 13. Implementation Acceptance After Separate Authorization

| Acceptance check | Required evidence |
|---|---|
| Dataset isolation | Exact fingerprints/counts/label mapping; complete train/validation coverage; test access rejected before file opening |
| Preprocessing | Both raw layouts produce [3,384,896] with v1 fill/padding/rounding/backend; jitter preserves mask/padding; evaluation is deterministic |
| Architecture and loading | Mandatory full pretrained digest; mismatch failure; retained pretrained representation/LayerNorm; logits [N,124]; full fine-tuning and correct stochastic-depth/evaluation behavior |
| Fixed optimizer groups | Conv/Linear weights decay at 0.05; biases, LayerNorm, and **all ConvNeXt layer-scale parameters are no-decay**; every trainable parameter appears exactly once |
| Sampling, loss, and updates | Complete natural epochs, unweighted CE, exact sample-mean accumulation including short tails, and correct optimizer/scheduler step counts |
| Full-budget execution | Every successful tuning/final run completes eight epochs and eight validations; worsening or flat validation scores do not terminate training |
| Scheduler and checkpoint selection | The full reviewed eight-epoch horizon is preserved even when best.pt is earlier; tau=1e-6 and earlier tie-break are tested, and NLL cannot override checkpoint selection |
| Resume | Interrupted/resumed behavior preserves order/jitter/RNG, LR, optimizer counters, selected-checkpoint state, history, and the original eight-epoch budget; wrong identities/pilot checkpoints are rejected |
| Seed separation and trial/final isolation | Exactly four distinct official seed IDs; one tuning seed absent from the three final IDs; all four LR trials use that one tuning seed, and final runs initialize independently |
| Metrics | Supported Macro-F1/Weighted-F1 from full-split counts, unsupported F1=null, Top-k across 124 outputs, retained cross-bucket false positives, and reports from the selected checkpoint |
| Artifacts and failures | Atomic checkpoints, explicit nonfinite/OOM/interruption outcomes, no overwritten trial histories, and traceability from the report to the selected checkpoint |

Tests and smoke must check behavior rather than treating a class name or pilot file layout as proof of official implementation. Use synthetic/train-only inputs for correctness checks. This document does not run those future implementation checks, train a model, or create an official execution framework.

## 14. Review and Handoff Checklist

- [x] Ground the draft in the 26 requested sources, covering E1 architecture, data, preprocessing, optimization, and reporting.
- [x] Separate fixed research decisions, recorded pilot behavior, and unresolved official adoption values.
- [x] Apply the reviewed full eight-epoch budget with early stopping OFF and fixed checkpoint tau=1e-6/earlier throughout the document.
- [x] Treat ConvNeXt layer-scale parameters as an unconditional no-decay category.
- [x] Require one tuning seed and three final seeds with all four IDs distinct, while leaving the actual numeric IDs unresolved.
- [x] Preserve fixed membership/preprocessing and diagnostic-only pilot status without adding experiments or promoting pilot LR/seed values.
- [ ] Obtain user review/approval of `docs/E1/specs.md` and resolve the applicable open register items; R5 is already fixed.
- [ ] Review/freeze a complete versioned numeric training recipe with its hash and authoritative references.
- [ ] Separately review an implementation plan and obtain implementation authorization.
- [ ] Implement E1, perform acceptance/smoke checks, run official LR tuning, and train the three disjoint final seeds.
- [ ] Unlock final test evaluation only in its separate phase after protocol/checkpoints are locked.

Creating or committing this revision does not freeze the master specification, mark official E1 implementation as complete, or authorize code/job execution. The next step is **reviewing this specification and the remaining open values**, not implementing E1 automatically.

## 15. Source Index - 26 Files Read

| # | File | Role |
|---|---|---|
| 1 | [docs/master_spec.md](../master_spec.md) | Main design, execution status, and official freeze boundaries; newer E1 decisions take precedence as stated in Section 1 |
| 2 | [docs/grillme.md](../grillme.md) | Research decisions, historical review, and scope controls |
| 3 | [docs/preprocessing_v1.md](../preprocessing_v1.md) | Frozen input/UI/resize contract |
| 4 | [docs/pilot_plan.md](../pilot_plan.md) | Completed pilot gates and pending official recipe |
| 5 | [docs/pilot_config_review.md](../pilot_config_review.md) | Historical proposals versus post-execution resolution |
| 6 | [docs/server_pilot_runbook.md](../server_pilot_runbook.md) | Runtime/checkpoint/Slurm references, not an official launcher |
| 7 | [reports/pilot/pilot_results.md](../../reports/pilot/pilot_results.md) | Verified diagnostic measurements, limitations, and post-P1 review |
| 8 | [P1 E1 config_resolved.json](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/config_resolved.json) | Actual bounded pilot configuration |
| 9 | [P1 E1 run_summary.json](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/run_summary.json) | Completion, identity, and selected diagnostic checkpoint |
| 10 | [P1 E1 history.json](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/history.json) | Three-epoch curves, counts, and 124-class per-epoch metrics |
| 11 | [P1 E1 metrics.csv](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/metrics.csv) | Scalar metrics, LR/update counts, and timing cross-checks |
| 12 | [P0 E1 run_summary.json](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/run_summary.json) | Strict full-epoch feasibility evidence |
| 13 | [P0 E1 config_resolved.json](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/config_resolved.json) | Declared batch/runtime/criteria/windows |
| 14 | [configs/preprocessing/geoguessr_v1.json](../../configs/preprocessing/geoguessr_v1.json) | Exact frozen geometry/UI/photometric contract |
| 15 | [artifacts/baseline124/summary.json](../../artifacts/baseline124/summary.json) | Counts, deduplication, and SHA-only grouping limitation |
| 16 | [artifacts/baseline124/labels.json](../../artifacts/baseline124/labels.json) | Exact 124-label index mapping |
| 17 | [artifacts/baseline124/class_support.csv](../../artifacts/baseline124/class_support.csv) | Train-frequency buckets and evaluation support |
| 18 | [artifacts/baseline124/provenance.json](../../artifacts/baseline124/provenance.json) | Source commit, artifact hashes, and byte counts |
| 19 | [src/pilot/p1_train.py](../../src/pilot/p1_train.py) | Bounded diagnostic runner; not an official eight-epoch/search implementation |
| 20 | [src/pilot/engine.py](../../src/pilot/engine.py) | Sample-weighted CE, accumulation, update-based scheduling, and tolerance-comparator reference |
| 21 | [src/pilot/models.py](../../src/pilot/models.py) | Backbone/head/loading/initialization and optimizer reference |
| 22 | [src/pilot/dataset.py](../../src/pilot/dataset.py) | Manifest integrity, split isolation, natural sampler, and seed/worker behavior |
| 23 | [src/pilot/checkpoint.py](../../src/pilot/checkpoint.py) | Atomic full-state epoch-boundary resume reference |
| 24 | [src/pilot/runtime.py](../../src/pilot/runtime.py) | Runtime/gates/identity and source fingerprints |
| 25 | [src/pilot/config.py](../../src/pilot/config.py) | Explicit unresolved-value validation; a pilot-only schema |
| 26 | [src/preprocessing/geoguessr_v1.py](../../src/preprocessing/geoguessr_v1.py) | Reusable frozen preprocessing implementation |
