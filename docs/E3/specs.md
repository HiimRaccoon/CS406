# E3 — Frozen-Representation Long-Tail Classifier Study

**DRAFT FOR USER REVIEW. E3 implementation, cache extraction, pilot execution and official head training remain NOT AUTHORIZED. Final test remains LOCKED / NOT AUTHORIZED.**

Revision date: **2026-11-11 (Asia/Saigon)**. The E3 protocol is not frozen. E3-specific choices without a reviewed value are explicitly **PENDING** and require resolution through the gates in Section 13; library defaults and completed full-model recipes cannot resolve them.

**E3-G0 is PASS for documentation/design preparation only: plan/tasks and E3-T00b preparation. E3-G1, implementation, cache extraction, pilot execution, official head runs and test access remain NOT AUTHORIZED.**

## 1. Research question, scope and authority

**On each completed E1/E2 representation, what changes when a reset linear classifier is trained with class-uniform sampling instead of natural sampling?** The principal controlled contrast is **cRT versus natural head retraining on the same frozen representation and cache**. Original heads remain references for the effect of resetting/retraining the classifier and removing training augmentation.

| Comparison level | Contrast | Interpretation |
|---|---|---|
| **PRIMARY** | `cRT - natural head retraining` within the same source family, source seed, accepted cache and paired initialization | Effect of changing the sampling policy under an otherwise identical head-training protocol |
| **SECONDARY** | `natural head retraining - original head` within the same source family and source seed | Combined effect of resetting/retraining the classifier and the deterministic training-input policy |
| **EXPLORATORY** | Compare `(E1-cRT - E1-N)` versus `(E2-cRT - E2-N)`, joining matching source seeds | Compare balancing effects across the two already completed representation families |

E3 uses the completed E1/E2 representations as fixed inputs. **E1-versus-E2 is not a new controlled architecture experiment in E3.** The exploratory contrast does not isolate architecture, feature dimension, local detail or source-training effects.

The study benchmarks country/territory classification on the exact existing fixed split. It examines overall accuracy and long-tail recognition together; it introduces no single winner metric, minimum gain requirement or new generalization claim.

| Source | Authority and interpretation for E3 |
|---|---|
| Current user instruction; [master specification](../master_spec.md), Sections B/C/H | Six-arm main comparison, frozen E1/E2 representations, deterministic single-feature caches, paired natural/cRT controls and bounded head-pilot approach |
| [Design review record](../grillme.md), Questions 9/10/15–19/29 | Natural retraining is necessary; full class-uniform cRT, no jittered feature bank, no full separate E3 LR-search branch |
| [E1 specification](../E1/specs.md), [task ledger](../E1/tasks.md), [closeout review](../E1/closeout_review.md) | Completed E1 method, final source seeds/checkpoints, input/metric conventions and verified supplied-package closeout scope |
| [E1-HR specification](../E1-HR/specs.md), [task ledger](../E1-HR/tasks.md) | Resolution-control context, stage isolation and authorization conventions; not an E3 source family |
| [E2 specification](../E2/specs.md), [task ledger](../E2/tasks.md), [plan](../E2/plan.md), [review](../E2/final_review.md) | Exact shared representation/fusion, identity/recovery/report-validation requirements and completed-stage handoff |
| [E2 server closeout record](../E2/server_closeout_2026-11-10.md) | E2 closed on the user's server-completion confirmation; server receipts/full lineage were not independently audited in this checkout |
| [E1 model](../../src/e1/model.py), [E2 model](../../src/e2/model.py) | Exact implemented pre-classifier feature boundaries; Section 5 reproduces their definitions |
| [E1 report](../../src/e1/report.py), [E2 report](../../src/e2/report.py), [metric implementation](../../src/e1/metrics.py) | Selected-checkpoint lineage, complete validation exports, metric semantics, sample-standard-deviation aggregation and actual schema differences |
| Frozen [split provenance](../../artifacts/baseline124/provenance.json), preprocessing configuration/module and completed numeric contracts | Exact immutable dataset, labels, preprocessing and inherited execution/reporting controls |

Older draft/status wording does not reopen completed decisions. In particular, master/E1/HR documents retain historical preparation language, and the E2 native review predates server execution. The current user identifies E1/E1-HR/E2 as completed; the later E2 ledger records closure explicitly. E3 respects that handoff without claiming to have independently inspected unavailable server evidence. Materializing the selected source binaries and their bindings is a prerequisite for E3 execution, not a request to rerun or reopen those experiments.

**Outside the main E3 scope:** E1-HR head retraining/cRT, StreetCLIP, new backbones, end-to-end fine-tuning, LayerNorm/layer-scale adaptation, feature transformations, new crop/fusion designs, multiple augmented features per image, auxiliary heads, class-weighted CE, logit adjustment, alternative balancing strengths, additional final seeds, ensembles, split regeneration and test evaluation. StreetCLIP belongs to a separate later experiment. No current source requires E1-HR in the main E3 matrix.

## 2. Main comparison and decision status

| Report arm | Source representation | Classifier state at start | Training distribution | Trainable parameters |
|---|---|---|---|---|
| E1 original head | Selected E1, 768-d | Unchanged selected source head | No E3 training | None |
| E1 natural head retraining (`E3-E1-N`) | Same selected E1, 768-d | New shared paired initialization | Natural | New linear weight and bias only |
| E1 cRT (`E3-E1-cRT`) | Same selected E1, 768-d | Identical initial state to E3-E1-N | Class-uniform, then image-uniform within class | New linear weight and bias only |
| E2 original head | Selected E2, 1536-d | Unchanged selected source head | No E3 training | None |
| E2 natural head retraining (`E3-E2-N`) | Same selected E2, 1536-d | New shared paired initialization | Natural | New linear weight and bias only |
| E2 cRT (`E3-E2-cRT`) | Same selected E2, 1536-d | Identical initial state to E3-E2-N | Class-uniform, then image-uniform within class | New linear weight and bias only |

Each arm reports **source seeds 2001, 3001 and 4001**, joined by matching ID. This gives **six original source-model references and twelve new head-training runs**, not eighteen new training runs. The E3 final run's root seed is its corresponding source seed. Do not choose a winning source seed, combine features across seeds, average source weights or add a head-seed sweep.

| Item | Status in this draft |
|---|---|
| Source families, six report arms and source-seed set | INHERITED from the intended study and completed final runs |
| Exact feature definitions, data/labels, no augmentation and evaluation-mode extraction | INHERITED; representation remains frozen throughout E3 |
| Natural retraining as the direct cRT control; only sampling differs within each pair | INHERITED and mandatory |
| Official head protocol across E3-E1-N, E3-E1-cRT, E3-E2-N and E3-E2-cRT | FIXED: one common protocol across all four arms and all three source seeds; numeric choices remain PENDING |
| Source-head references | Unmodified original heads; no E3 optimization or reselection |
| Metrics, frequency buckets, three-seed arithmetic mean/sample std and display conventions | INHERITED unchanged |
| Head checkpoint comparator | Adopt the completed supported-Macro-F1 chronological comparator, `tau=1e-6`, earlier ties; cadence and horizon remain PENDING |
| Head termination policy | Fixed budget; early stopping OFF. The numeric head budget is PENDING, not the source eight-epoch budget |
| Head optimization, initialization and small-pilot envelope | PENDING under Section 14; require explicit review/freeze |
| Cache storage/extraction operational choices and numerical acceptance tolerances | PENDING under Section 14; cannot change the feature definition |
| E3-G0 | PASS for documentation/design preparation only: plan/tasks and E3-T00b preparation |
| E3-G1 and all implementation/execution gates | NOT AUTHORIZED |
| Final test | LOCKED / NOT AUTHORIZED |

The head comparator and fixed-budget policy are explicit E3 design adoptions of completed conventions for review. They do not imply that an E3 protocol has already been approved or executed.

## 3. Selected source checkpoints and lineage handoff

Use the **selected official final-seed checkpoint** for each source, including its trained original classifier. Do not substitute `last.pt`, a tuning/pilot checkpoint, fresh ImageNet features or an earlier convenient binary. E1 seed 3001 specifically uses selected **epoch 7**, although its completed source run trained eight epochs.

| Source | Completed selected LR | Seed | Selected epoch | Recorded selected `best.pt` SHA-256 |
|---|---:|---:|---:|---|
| E1 | 1e-4 | 2001 | 8 | `e9c6202568715d81cf0949f4a7cfcd2546839b9cb12f577cdfc2a603de910a4f` |
| E1 | 1e-4 | 3001 | 7 | `2855d17ac932c3986a64363259d12334aa0a5af6f8f60f3fb9258a8be9544fdf` |
| E1 | 1e-4 | 4001 | 8 | `eb74126015b3854cad78fb9c2dff7a59a2352b3299a8998ef963ca7f72c68ef2` |
| E2 | 5e-5 | 2001 | 8 | `d54dc636908085b6396766c9f7c8962ee707bf4862dab7891c18c3363029902c` |
| E2 | 5e-5 | 3001 | 8 | `a8f2652b519cf55cb5281cc0c6cd993acc1eeb569102acbddff2e53695236e7d` |
| E2 | 5e-5 | 4001 | 8 | `f759e266f04dd2d6f9693747468a1ea5ccc053a35e8cbf332d21ac0b7a818a5e` |

These are **recorded report/checksum references**; actual source binaries require independent verification at source-byte acceptance. The LR column describes completed representation training; it supplies **no E3 head LR**.

| External/materialized source-report artifact | Recorded exact file-byte SHA-256 | Source contract SHA-256 |
|---|---|---|
| E1 official aggregate JSON, logical filename `e1_three_seed_validation_report_v1.json` | `3387477c40a97388e967f10fd74d313664946d0b7ec1ae963ad0e49a98b3db9e` | `c9153c53534e95099c374e8ab7b54bc347588f11b200adcc92b3c23ad6ac4143` |
| E2 official aggregate JSON, logical filename `e2_three_seed_validation_report_v1.json` | `75bdba59bccc8c8b7db498363d5c411cbde5165e0f9791bd925f41be6e5a82b9` | `ef8767460bbd54454baa22ce289ed4222742954bcdefa8024a8972f5dc958f72` |

Neither report is present as a repository-relative file on branch `e3`. These logical filenames identify external/materialized source artifacts, not repository links. At **E3-T00b/P01**, supply each report's actual existing path, verify its exact bytes against the recorded SHA-256 above, and record that path/digest in the candidate source manifest. Do not fall back to an assumed `src/e1/` or `src/e2/` report path. The E1 aggregate digest also matches the completed E2 comparator pin; the recorded digests do not imply E3 acceptance or approval.

T00b must distinguish these path and binding classes:

| Class | T00b treatment |
|---|---|
| Actual existing source-artifact inputs | Inventory actual supplied report/binary/analysis paths and available historical source evidence; verify report bytes at P01 and retain the source-byte acceptance requirements below. Unmaterialized inputs remain explicitly PENDING, not accepted merely because a prospective path is named |
| Planned cache/output roots | Declare intended roles, locations and containment separately; these roots and payloads may not exist yet. A planned root is not an existing source input or an accepted cache |
| Candidate schema/receipt bindings | Specify schemas, gate roles, prospective receipt paths and required binding fields for review. Future E3 accepted-cache indices and approved receipts are not prerequisites before G1; do not invent their hashes, approval status or contents. Existing historical source evidence is recorded separately from future E3 approvals |

Before source-byte acceptance, obtain the actual six selected binaries and applicable declarations/resolved configurations, selected analyses/predictions, histories/summaries, selection bindings and recorded source contract identities. Hash the actual files, strict-load compatible complete state dictionaries and reconcile source family/seed/selected epoch/head shape with the registry. The current specification does not presume server checkpoint paths, receipt paths or receipt contents; those are **PENDING materialization**. Missing/mismatched source bytes block E3 use of that source, without silently replacing it or treating the other two seeds as a complete aggregate.

E1's older report schema puts seeds in `per_seed` and does not carry every newer E2 top-level field. Validate the actual reviewed source schema and obtain referenced semantics from its frozen contract. Do not require an invented E2-style E1 envelope, rewrite original reports or demand equal scientific/runtime identities across the two different architectures.

## 4. Exact dataset and split isolation

Reuse `artifacts/baseline124/{manifest.csv,summary.json,labels.json,class_support.csv}` byte-for-byte, source commit `33cb29d72974f1677efab0f56ed92360e5a9f712`. Provenance ID is `baseline124-v1`; historical split seed **406** never regenerates membership or becomes a head-training seed.

| Split | Unique images | GT-supported labels | E3 access |
|---|---:|---:|---|
| train | 34997 | 124 | Authorized deterministic caching and head training only |
| val | 7497 | 106 | Authorized deterministic caching, protocol/checkpoint selection and validation reports |
| test | 7502 | 111 | Metadata only; image/feature/prediction access prohibited |

There are 49996 unique fixed-split images and 124 output labels. `labels.json` is an ordered list; its index is the label ID 0–123. Preserve names/order, including `0=Aland`, `120=United States`, `123=Vietnam`. Do not filter extreme-tail classes, refit on train plus validation or select a new partition.

| Fixed artifact | Exact bytes | SHA-256 |
|---|---:|---|
| `manifest.csv` | 7519661 | `9cbd8b70e8522b5aa4a20fba4fa7b2ecea855cf377f98e41ebb162a07f8a8040` |
| `summary.json` | 994 | `cf3dcf909aefd17c531cd115d6c30e54bbeb3dbd13962f974317ccde8a2f3e4b` |
| `labels.json` | 1927 | `5f7e8f5810e9ac8c43c1c22b2e4febe2e7a6174629b9031d3e943fce511832aa` |
| `class_support.csv` | 6308 | `ff46f4074464644f573b4ca64eb61320a61a8cc3daed3a3886a350cbc3c6fb75` |

Before extraction/model allocation, verify metadata integrity/coverage and the current root-contained train/validation image audit/conformance. Reject missing/corrupt files, unsafe paths/symlink escapes, wrong layouts, duplicate/missing cache keys and mismatched labels; never skip/replace rows. An unchanged valid audit may be reused only with its actual current root/content/software bindings verified.

Reject `split=test` before image opening, image-byte hashing, preprocessing or feature loading. The restriction also excludes precomputed test features, previews, predictions and test-based selection signals. Frozen manifest metadata can be read without opening test images. The historical pipeline previously inspected/evaluated test; the present lock does not justify claiming test was never seen.

## 5. Frozen representation and deterministic input contract

### 5.1 Exact implemented feature boundaries

For E1, use `E1Model.encode` unchanged in meaning:

```text
z_E1(x) = flatten(final_norm(avgpool(features(global(x)))))
shape   = [B,768]
```

For E2, use `E2Model.encode_view` and `E2Model.encode` unchanged in meaning:

```text
f(v)       = flatten(final_norm(avgpool(features(v))))
G          = f(global)                                      # [B,768]
L          = f(local.flatten(0,1)).reshape(B,4,768)            # [B,4,768]
z_E2(x)    = concatenate([G, L.mean(dim=1)], dim=1)           # [B,1536]
```

E2 has one shared ConvNeXt-Tiny parameter set. Apply the source final LayerNorm **per view before averaging**, execute the global call first, then one call on batch-major flattened locals, and preserve configuration crop order `left, middle_left, middle_right, right_raised`. Do not average pre-normalization activations, concatenate four separate locals, invert fusion order or add post-fusion normalization/projection. Source representations include the completed trained final LayerNorm and layer-scale values.

| Source | Deterministic model input | Cache feature | Original/new linear-head shape |
|---|---|---|---|
| E1 | `[B,3,384,896]` global | `[B,768]` | `Linear(768,124,bias=True)` |
| E2 | global `[B,3,384,896]`, local `[B,4,3,320,320]` | `[B,1536]` | `Linear(1536,124,bias=True)` |

Set every source representation parameter `requires_grad=False`, including final LayerNorm and ConvNeXt layer-scale; preserve buffers/state. Keep representation modules in **evaluation mode** throughout extraction. Use no gradient recording for extraction; stochastic depth is OFF. A source builder's default full trainability is not evidence of freezing: strict source-state loading must be followed by explicit freeze/eval assertions. Calling a parent `.train()` must not reactivate the encoder later; cached-head training needs only the standalone classifier.

### 5.2 Input preprocessing and augmentation removal

Use frozen preprocessing ID `geoguessr-ui-global-local-v1`, configuration SHA-256 `4f49dd41738168cfc65d4322fbe1b659ef5595229bb487ad86dd671a26a8903f` and module SHA-256 `4d1dea328f69c1c7e1fa4f333c7276de0c3bc81b322db40dc33530969357e666`, together with the runtime-read checksum sidecar.

Construct E1 global **896×384**, and E2 global **896×384 plus four local 320×320**, from the decoded raw image through the existing experiment modes. Preserve RGB decode without EXIF transpose, accepted layouts 1536×662 / 1030×703, 14-pixel footer removal, frozen UI masks and retained navigation arrows. Preserve exact float32 ImageNet-mean fill, aspect-preserving centered global letterbox, tensor bilinear resize with antialias/half-up rounding, and ImageNet normalization `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`.

E2 locals come directly from the same UI-processed raw tensor using authoritative integer `local_crops_xywh`; never crop a resized global/padded canvas. Do not duplicate coordinate tables or substitute `weights.transforms()`/another resize backend.

**Both train and validation extraction disable brightness/contrast jitter.** Each original image has exactly one deterministic feature per source checkpoint. There is no epoch-dependent extraction, online augmentation, jittered cache bank or test-time augmentation. This is the master E3 design and intentionally differs from augmented full-model source training; natural retraining controls that change when interpreting cRT.

## 6. Feature-cache construction, identity and acceptance

Maintain **six distinct source-cache bundles**, each containing one train and one validation split: `2 families × 3 source seeds × 2 splits = 12 split payloads`. Every bundle contains 34997 train rows and 7497 validation rows in fixed manifest order. Caches are immutable after acceptance and shared by reference, not regenerated independently for natural and cRT.

| Cache boundary | Required contents/checks |
|---|---|
| Source identity | Family, source seed/run ID, selected epoch, actual selected-binary SHA-256, original-head state identity, source contract/report/selection bindings |
| Feature identity | Exact Section 5 definition, dimension, view shapes/order, fusion, final LayerNorm position, evaluation mode, jitter OFF |
| Input identity | Split, manifest-relative paths in order, label IDs, row counts, ordered labels/support and fixed artifact/preprocessing fingerprints; current data-root/audit binding |
| Extraction identity | Consumed source hashes, actual software/device/runtime, batch/chunk policy, compute dtype, storage dtype/layout and extraction acceptance binding |
| Payload identity | Exact files/chunks, shapes, byte counts/checksums, ordered key/label index and completion index; reject missing/extra/reordered/partial chunks |

Extraction arithmetic inherits deterministic **FP32**, with AMP/scaler/TF32/compile OFF. Exact on-disk dtype/container/chunk size and extraction batch/worker policy are **PENDING**. Storage must preserve extracted FP32 feature values losslessly; FP16/quantization, feature centering/standardization, whitening, L2 normalization, PCA and learned transforms are not implicit storage optimizations. Any proposal changing values requires design review rather than promotion under the same feature identity.

Require finite features, exact dimension/coverage and a one-to-one `(source family, source seed, split, relative_path, label_id)` index. Validate hashes and bindings before use, not just matching row counts. Train labels must not be taken from validation support or inferred from cache order without checking the index.

Extraction may publish deterministic chunks atomically, but a cache is usable only after its complete index is committed and validated. Recovery granularity is **PENDING**; partial caches, stale indices and mixed-checkpoint chunks cannot become accepted input. Existing accepted payloads are never overwritten by a different source/runtime extraction.

### Original-head replay is a cache acceptance check

For each source, retain its unchanged original head and verify:

```text
source.forward(deterministic_input)
          versus
source_original_head(accepted_preclassifier_feature)
```

Use train-only fixtures for bounded acceptance. Full validation replay and full train/validation caching need their explicit access/execution authorization. Verify ordered predictions/Top-5, full-split/bucket metrics and selected-source lineage against the original supplied validation analysis; do not accept equal aggregate scores as sufficient evidence of equal predictions.

Discrete membership/GT/Top-k/confusion/coverage must agree with the authoritative selected-source outputs. Float-logit and NLL comparison tolerances are **PENDING** and must be frozen before cache acceptance; no tolerance may silently excuse wrong feature order, missing samples, label remapping or changed predictions. Batch-shape/runtime differences can affect floating-point replay even under deterministic execution; investigate mismatches and stop for review rather than silently replacing the original reference or changing the selected checkpoint.

The original-head rows in E3 retain the verified original selected models/results. Cache replay is equivalence evidence, not a newly tuned baseline or permission to reselect original weights. Missing binaries/selected analyses or failed replay block the affected cache and the complete E3 handoff.

## 7. Classifier-only training and paired controls

E3 trains a new **single biased linear layer** on accepted cached features. No original source-head weights are used to initialize either retraining arm. Preserve the original trained head separately for its reference arm; resetting it in place must not destroy source evidence.

| Head family | Trainable tensors | Exact parameter-count oracle |
|---|---|---:|
| E1 | New weight `[124,768]` and bias `[124]` | **95,356** = `124*768+124` |
| E2 | New weight `[124,1536]` and bias `[124]` | **190,588** = `124*1536+124` |

These counts follow from inherited dimensions; they are not selected hyperparameters or observed training results. Representation trainable count is **zero**. Optimizer enumeration must contain exactly the new head tensors once each, with no encoder/final-normalization/layer-scale parameters or retained optimizer state from E1/E2. Existing source `build_optimizer` functions define full-model recipes; they must not be used unchanged as an E3 authorization/configuration shortcut.

**One common official head-training protocol is required for E3-E1-N, E3-E1-cRT, E3-E2-N and E3-E2-cRT, across all three source seeds.** All four arms use the same optimizer family/configuration, LR/schedule, weight-decay policy, batch/accumulation policy, presentation/update budget, validation cadence, checkpoint-selection rule and numerical policy. Independent E1/E2 final-protocol tuning is excluded.

Only differences mathematically required by feature dimension are allowed: `Linear(768,124)` versus `Linear(1536,124)`, their parameter counts and fan-in-dependent bounds under one shared initialization rule. These dimensional consequences do not permit separately chosen learning rates, regularization, schedules or budgets. Actual initialized tensors must match within every N/cRT pair; cross-family tensor equality is not required for different shapes.

For every `(family, source_seed)` natural/cRT pair, require identical:

- Selected source checkpoint and accepted train/validation cache bytes/indices.
- New head architecture, initialization rule, actual initial weight/bias tensor values and initialization-state hash.
- Optimizer family/hyperparameters, regularization groups, LR trajectory, batch/accumulation policy and clipping/numerics.
- Total feature presentations, optimizer updates, validation schedule, fixed termination budget and checkpoint-selection rule.
- Loss definition, metric/export code and full validation membership/order.

Only the sampling policy changes. Create the paired head state once under the reviewed deterministic initializer and clone it before either branch trains. Fresh optimizer/scheduler states start each branch; they are not copied from the trained natural arm into cRT. RNG consumption by one branch must not change the other's initialization or loader state.

The completed source initializer is `linear_default` with isolated CPU RNG and fan-in-specific bounds. **E3 adoption of that rule, its exact stream binding and any other initializer remains PENDING for the small pilot.** Do not silently retain it, use zeros, warm-start from the original head or hard-code a library default. E1/E2 head tensors have different shapes; equality is required within a pair, not across feature dimensions.

Use raw `[B,124]` logits and **standard unweighted CE with label_smoothing=0** for both retraining arms. No class weights, balancing loss, bias/logit prior correction, extra regularizer or auxiliary view loss may be added only to cRT. Optimizer family, LR, betas/eps or momentum, head weight decay/bias-decay policy, schedule and clipping are **PENDING**, not inherited automatically from full-model AdamW.

## 8. Natural versus class-uniform sampling and budget accounting

Let `N=34997` and `n_c` be unique fixed-split train support of label `c`. All 124 labels have positive train support.

| Arm | Sampling contract |
|---|---|
| Natural | Inherit a shuffled image permutation without replacement within a natural pass; each training image appears once per complete pass. Marginal `P(c)=n_c/N`, `P(i given c)=1/n_c` |
| cRT | For every draw, choose a label uniformly among all 124, then an image uniformly within that label: `P(c)=1/124`, `P(i given c)=1/n_c`, so `P(i)=1/(124*n_c)` |

cRT draws use replacement across draws; the same class/image can repeat, including within a batch. There is no rejection of repeats, quota forcing or guarantee of exact per-class counts in a window. Do not replace this two-stage distribution with a uniformly shuffled feature cache, remove rare classes, or renormalize over validation-supported labels.

Natural pass ordering retains the completed SHA-256-derived CPU-generator/`torch.randperm` convention. New cRT generator API/stream keys, class/within-class draw consumption and exact resume state are **PENDING** and must be explicit before the pilot. The class-support index is immutable and each class's image index uses a deterministic manifest order. Do not use Python's process-dependent hash or global RNG coupling to feature loading/head initialization.

**E3 head batch, accumulation, epoch/window definition, draws, updates and horizon are PENDING.** The source values `2/2/1`, eight epochs, 17499 updates/epoch and 139992 total are completed full-model history, not a head recipe. Matching nominal epochs while giving cRT more presentations or updates is invalid.

At freeze, specify one complete budget basis. If a window is approved as one complete natural pass, its cRT counterpart necessarily has **34997 draws**, without implying 34997 distinct cRT images. For microbatch `m`, accumulation `a`, draws/window `D` and windows `E`:

```text
microbatches/window = ceil(D/m)
updates/window     = ceil(ceil(D/m)/a)
total updates      = E * updates/window
```

This is conditional accounting, not a selection of `m`, `a`, `D` or `E`. A fixed-update alternative must define total presentations, permutation cycling/terminal partial-pass behavior, tail groups and validation boundaries explicitly and match them across each pair. Freeze that choice before interpreting any head result.

Retain tails; with `D_group` actual feature rows in an accumulation group, divide summed CE by `D_group`. Repeated cRT rows count as presentations for loss/update accounting but do not increase unique image support. Train CE is presentation-weighted; validation NLL uses all 7497 unique validation samples. Report both presentation totals and sampled unique-image/class coverage without redefining buckets from draw counts.

## 9. Small head pilot and final protocol freeze

The master design authorizes the **concept** of a small head pilot, not its execution or numeric envelope. There is no automatic four-LR campaign for E3 and no separate search per sampling arm/final seed.

Before pilot authorization, review a bounded envelope specifying actual source lineage/cache inputs, pilot seed role/stream binding, finite candidate protocols, maximum trials/draws/updates/validation passes and selection/tie/convergence-evidence criteria. All of those E3-specific values are **PENDING**. Existing source tuning ID **1001** is not a fourth final seed and is not silently promoted to an E3 pilot choice. Do not add arbitrary new official final IDs or use split seed 406/pilot seed 3406 as defaults.

The pilot must address whether head optimization is adequate under the proposed matched protocol, using recorded loss/LR/validation histories and the predefined rule. A bounded run is not proof of convergence merely because it completes. Predeclare the pilot's coverage of the four arms and how its evidence selects one common official protocol; independent best-per-arm or best-per-family tuning would invalidate the intended design.

The pilot must select **ONE common official head-training protocol** shared by E3-E1-N, E3-E1-cRT, E3-E2-N and E3-E2-cRT and all three final source seeds. Protocol sharing is already fixed; only its unresolved values are PENDING. Apply the dimensional exceptions in Section 7 without independently tuned family profiles. No per-source-seed optimization or open-ended architecture/LR/initialization sweep is allowed.

Publish all pilot candidates/outcomes, failures/retries and the selection rationale. Exhausting the approved envelope causes a review stop, not an unrecorded extension or fifth trial. Pilot heads/results cannot count as final runs or supply trained initial states. Even if the same source cache is approved for reuse, final heads/optimizer/scheduler start freshly from the locked initialization contract.

After the pilot, require explicit `E3-HEAD-FREEZE` review of the complete common head protocol and exact bytes/checksum. A candidate must resolve all official-head fields, bind the approved cache/source manifests and enforce both cross-family protocol sharing and within-pair controls. Final declarations and execution require their separate authorization after this freeze.

## 10. Deterministic runtime, RNG and recovery

The extraction reference is the completed software lock: Python 3.11.17; Torch 2.8.0+cu128; TorchVision 0.23.0+cu128; Pillow 11.3.0; NumPy 2.1.3; pandas 2.2.3; scikit-learn 1.6.1; CUDA 12.8/cuDNN 91002. Preserve FP32, matmul precision `highest`, AMP/scaler/TF32/compile OFF, cuDNN benchmark=false/deterministic=true and deterministic algorithms=true/warn_only=false. Record actual effective settings, packages and device/driver. A reference version table is not new target acceptance.

Head-only CPU versus target-GPU execution, loader/worker/thread settings, extraction batch/chunk policy and allocation/walltime are **PENDING**. The UIT H200 MIG/Slurm environment and nine-hour attempts are completed-stage references, not automatic E3 resource authorization. Any approved GPU work must occur inside its explicitly authorized allocation. A reduced head workload does not justify silently changing runtime precision or claiming inherited target acceptance.

Keep final root IDs 2001/3001/4001. Inherit explicit Python/NumPy/CPU/all-CUDA seeding and the pipe-separated SHA-256 derived-seed convention: first eight bytes little-endian modulo `2**63-1`. New head-initializer/cRT/loader stream keys and pilot role remain PENDING. Separate streams prevent sampler or worker scheduling from changing the paired initial state. Record/save every actual generator state and cursor needed by the chosen sampling policy.

After authorization, a head checkpoint must include head/optimizer/full scheduler state, inactive-scaler policy, root/derived RNG and loader/sampler states, source/cache/init/protocol/declaration hashes, committed presentation/update counters, complete validation history and selected head weights/state. It must contain no trainable source-representation state or borrowed source optimizer progression.

Use a single writer, boundary-zero state, atomic committed checkpoints and authoritative `last` versus reporting-only `best` roles. Exact E3 commit/replay boundary units are **PENDING** with the budget protocol. They must permit deterministic discard/replay of uncommitted work, including cRT class/image draws and partial accumulation groups, without duplicate credit or additional budget. Restore the same immutable run; never resume training from reporting-only `best`, another seed/arm or an original E1/E2 checkpoint.

No nonfinite/OOM outcome can silently skip data/updates or change LR/batch/dtype. Preserve attempts/failures/retries and unknown partial work. A completed budget may recover publication but cannot train again. Changed consumed scientific files, caches, source binaries or runtime bindings require review/renewed relevant acceptance rather than self-approved continuation.

Maintain explicit consumed-source maps and canonical fingerprints separately from clean Git commit, test/acceptance harness inventories and operational job IDs. Use the completed canonical serialization `json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)` with default separators for object fingerprints; exact file-byte SHA-256 remains a separate binding. Future actual source closure/hashes are PENDING until code exists. Documentation/test-only clean commits do not by themselves change an unchanged scientific/runtime identity. No commit/hash/target observation is fabricated by this specification.

## 11. Selection, metrics and fixed frequency buckets

For retrained heads, full deterministic validation occurs at the **PENDING reviewed cadence**. At the first completed validation point, initialize selection; later replace only when `current_supported_macro_f1 - selected_supported_macro_f1 > 1e-6` on the `[0,1]` scale. Otherwise retain the earlier selected point. NLL, Top-k and test scores do not break ties. Log the selected update/presentation index and any defined pass/epoch number.

Every successful head run completes its entire frozen budget with early stopping OFF, regardless of when its selected head occurs. All its reported metrics and prediction/error analysis use that **same selected head**, not best-per-metric epochs or automatically `last`. Original-head arms use the already selected source checkpoints in Section 3 and have no E3 checkpoint search.

| Metric/output | Required completed-stage semantics |
|---|---|
| Top-1 / Top-3 / Top-5 | Rank all **124 original raw logits**, descending; ascending label ID for exact ties using stable ranking, with no logit perturbation or unsupported-label masking |
| Supported Macro-F1 | Mean full-split per-class F1 over the **106 GT-supported validation labels** |
| Unsupported class F1 | `null`/not evaluated and excluded from Macro-F1; a GT-supported class with zero true positives instead has F1=0 |
| Weighted-F1 | Full-split per-class F1 weighted by validation GT support, never train counts or sampling frequency |
| NLL | Sample-mean unweighted validation CE, diagnostic only; no percent formatting or selection/tie role |
| Per-class output | All 124 label IDs/names, unique `n_train`, fixed bucket, validation GT support, predicted count, precision/recall/F1 |

Use the existing metric denominator conventions, including precision `TP/max(predicted_count,1)` and recall `TP/max(GT_support,1)`; only unsupported **F1** is made null by the completed implementation. Do not blanket-convert unsupported precision/recall into a different schema. Accumulate the full-split confusion/counts before computing F1; never average minibatch F1.

Bucket membership comes solely from unique fixed-split train support in `class_support.csv`:

| Train-support bucket | Total labels | Unique train images | Val-supported labels | Validation images |
|---|---:|---:|---:|---:|
| `1-19` | 42 | 243 | 24 | 49 |
| `20-99` | 37 | 2077 | 37 | 445 |
| `100-499` | 29 | 7049 | 29 | 1513 |
| `>=500` | 16 | 25628 | 16 | 5490 |
| `1-2` — overlapping subset of `1-19` | 20 | 21 | 2 | 2 |

For each bucket report total labels, GT-supported labels, evaluation samples, supported Macro-F1 and Top-1/3/5. Bucket Macro-F1 averages supported member classes' F1 already computed on the **entire validation split**, retaining false positives contributed by other buckets. Bucket Top-k conditions on GT bucket membership while ranking all 124 outputs. No-support bucket metrics are null. Repeated cRT draws never change these memberships, denominators or independent sample counts.

The `1-2` subset has only two validation-supported labels and two validation images; it is not a fifth disjoint bucket. Improvement on those observations cannot establish reliable extreme-tail generalization. Geographic/panorama independence remains unproven, and seed variation does not remove sampling uncertainty.

## 12. Reports, comparisons and evidence validation

Produce the main six-arm validation table, each arm's three source-seed rows, and arithmetic **mean ± sample standard deviation (`ddof=1`)** from unrounded per-seed values. Display classification metrics/std as percentages with **four decimal places**; retain full precision in JSON/CSV. Averages are over individual selected models, not an averaged-logit ensemble, merged confusion matrix or cherry-picked seed.

Use the comparison hierarchy from Section 1. For family `f` and matching source seed `s`:

```text
PRIMARY:      balancing_effect(f,s) = cRT(f,s) - N(f,s)
SECONDARY:    retraining_effect(f,s) = N(f,s) - original(f,s)
EXPLORATORY:  compare balancing_effect(E1,s) versus balancing_effect(E2,s)
```

Report all five metrics and fixed buckets for these contrasts, including negative trade-offs. Classification differences use **percentage points**. If aggregating paired differences, compute them per matched seed first, then their mean/sample std; do not subtract standard deviations. Apply the same matched-seed calculation if reporting the exploratory difference between the E1 and E2 balancing effects. A combined `cRT - original head` change may be shown as descriptive context, not as the primary balancing effect.

Natural-versus-original combines reset/additional head optimization and the deterministic training-input policy, so it is not a pure balancing effect. The exploratory comparison uses one common head protocol, but the fixed E1/E2 representations still differ in feature dimension/architecture and source learning. **It is not a new controlled E1-versus-E2 architecture experiment** and does not isolate local detail, head capacity or compute. Three paired seeds do not establish significance.

Required later artifacts, with concrete filenames/layout finalized in planning:

| Artifact | Required content |
|---|---|
| Source handoff manifest | Six actual selected binaries; immutable recorded/verified hashes; original heads; reports/selected analyses and complete source bindings |
| Accepted cache indices/payloads | Six source bundles/twelve split payloads; ordered keys/labels, complete coverage, exact feature/runtime identity, checksums and original-head replay evidence |
| Pilot envelope and outcome ledger | Approved limits/candidates/inputs/seeds; every attempt/outcome/history; rationale and separate head-protocol freeze decision |
| Frozen head protocol and declarations | One common official protocol for all four retraining arms; all formerly PENDING runtime/optimization/budget/init/sampling fields; explicit family/source seed/arm; accepted cache/init/source/approval bindings |
| Per-run state and histories | Initial head hash, `best`/`last`, complete presentations/updates/validation records, LR/loss curves, environments and failure/replay inventories |
| Selected validation analysis | `analysis.json`, `overall.csv`, `per_class.csv`, `buckets.csv`, `predictions.csv`, `confusion.csv`, `most_confused.csv`, `learning_curves.csv`, with source/cache/head lineage |
| E3 aggregate/comparison and inventory | All six report arms and exactly three source seeds per arm; ddof=1, coverage, paired contrasts, exact hashes and explicit verification scope/limitations |

Preserve existing export field meanings and augment them with E3-specific lineage. Predictions contain every fixed validation path/GT label in manifest order, the predicted label and five distinct valid Top-5 label IDs; the predicted label must equal the first ranking. Confusion matrices are 124×124 with true labels on rows. Most-confused pairs sort by descending count, then true/predicted label ID as in completed reports.

Apply the E2 final-review correction: aggregation/closeout must reject missing/truncated/reordered predictions and CSVs even if their hashes were recomputed. Reconstruct discrete Top-k/confusion/full-split/per-class/bucket metrics from complete predictions and fixed metadata, then compare headers/rows of all derived CSVs to validated content/history. Top-5 IDs alone do not reconstruct NLL; bind its sample-mean CE calculation to selected-head logits/evaluation evidence instead of claiming that reconstruction proves it.

Publish the analysis index only after every export succeeds. Verify declaration ↔ source/cache/init/protocol ↔ committed history ↔ selected head ↔ predictions/exports ↔ aggregate lineage, not caller-supplied scalar scores. Keep reporting-only tooling/harness exact-hash inventories separate from training source identity unless actually consumed there.

Efficiency is secondary if measured. Separate source feature extraction, cache I/O, head training, validation/checkpoint I/O, attempt/replay and whole-study costs. Count repeated head feature rows as presentations, not new unique images, and do not equate E2 views/s with original images/s. Invent no runtime/VRAM/speedup or minimum improvement threshold.

## 13. E3 gates and authorization design

**E3-G0 is PASS for documentation/design preparation only: plan/tasks and E3-T00b preparation. E3-G1 and all later implementation/execution gates remain NOT AUTHORIZED.** Existing E1/E1-HR/E2 approvals do not authorize E3 caching or head training. Keep two distinct freezes: a reviewed **cache/acceptance/pilot contract** before implementation and pilot access, and the **one common final head protocol** after pilot review. This avoids requiring a pilot outcome before the pilot can be authorized while preventing official defaults from slipping through unresolved fields.

**E3-TARGET-AUTH and E3-G2 concern bounded acceptance of source checkpoint loading, feature extraction/runtime and deterministic recovery.** Any head/recovery fixture trains only a standalone classifier while representations remain frozen. These are not full-model training gates and do not reopen E1/E2, change their checkpoints or authorize further backbone optimization.

| Gate / stage | Concrete review requirement | Scope after explicit approval | Current status |
|---|---|---|---|
| E3-G0 | Specification reviewed for documentation/design preparation only | Plan/tasks and E3-T00b source-manifest/cache-pilot-candidate preparation only; no numeric freeze, implementation or execution | **PASS** |
| E3-G1 | Review plan and exact cache/pilot contract bytes; resolve every field needed by those stages; select method and explicitly authorize named implementation/CPU-preparation scope | Implement/verify only the approved E3 scope; final-head numeric outcomes may remain PENDING and official head execution must fail closed | **NOT GRANTED** |
| E3-TARGET-AUTH | Explicitly authorize bounded source-checkpoint-loading, feature-extraction/runtime and deterministic-recovery acceptance on actual E3 inputs | Train-only extraction and standalone-head/recovery fixtures with frozen representations; no full-model training, bulk cache, pilot selection or official head training | **NOT GRANTED** |
| E3-G2 | Accept the bounded source-loading/extraction/runtime/recovery evidence, verified source handoff, numerical-replay policy and concrete cache declarations | E3 readiness accepted; no E1/E2 reopening or full-model training; does not itself authorize bulk extraction | **NOT GRANTED** |
| E3-CACHE-AUTH | Bind the six source identities and exact train/val extraction/replay scope to explicit execution permission | Deterministic bulk train/val caches and original-head validation replay only | **NOT GRANTED** |
| E3-CACHE-ACCEPT | Review complete cache coverage/checksums, original-head replay and source/input/runtime identity | Accepted immutable inputs for a separately approved head stage | **NOT GRANTED** |
| E3-PILOT-AUTH | Approve the complete bounded pilot envelope, actual source/cache choices, seeds, candidate/selection policy and ceilings | Small head pilot only; no final heads or automatic envelope extension | **NOT GRANTED** |
| E3-HEAD-FREEZE | Review every pilot outcome and one common final head protocol for all four arms; resolve official PENDING fields, exact bytes/sidecar, dimensional exceptions and paired constraints | Locked common final configuration; no final job permission by itself | **NOT GRANTED** |
| E3-FINAL-AUTH | Bind the accepted identities/caches, locked head protocol and exactly twelve new paired head-run declarations | Those final runs and their selected validation analyses only | **NOT GRANTED** |
| E3-CLOSEOUT | Verify twelve complete head runs plus six unchanged original references; all exports/paired aggregate/lineage and failure inventories; explicit user review | Close E3 train/validation only | **NOT STARTED — GATED** |
| FINAL-EVALUATION | All intended comparison experiments have frozen protocols/checkpoints and closed train/validation; separate explicit study-wide user approval | Outside this specification | **LOCKED / NOT AUTHORIZED** |

Flow: **draft → G0 → reviewed plan/cache-pilot contract → G1 → implementation/CPU preparation → TARGET-AUTH → bounded target acceptance → G2 → CACHE-AUTH → CACHE-ACCEPT → PILOT-AUTH → pilot → STOP → HEAD-FREEZE → FINAL-AUTH → twelve head runs/analyses → CLOSEOUT**. No edge means automatic permission or job submission. Target harnesses and candidate receipts cannot approve themselves.

Each receipt must identify the gate, actual approved contract/evidence/cache/declaration hashes, allowed operation/splits and named scope. Freeze exact approved bytes with an external checksum; preserve old preparation snapshots. Do not create a receipt/contract self-hash cycle, fabricate future receipt digests or change immutable source evidence to satisfy E3 validators. Relevant changes require explicit re-review; failure provides no unilateral scientific fallback.

## 14. PENDING register and freeze obligations

All entries below are genuinely unresolved E3 choices or not-yet-observed inputs. They are not invitations to tune inherited feature geometry, label mapping, bucket definitions or reporting semantics.

| ID | PENDING item | Required resolution / latest stop |
|---|---|---|
| P01 | Actual existing external source-report/checkpoint/analysis paths, available historical source evidence and complete binary handoff | At E3-T00b supply actual existing report paths and verify their recorded SHA-256 values; inventory existing source inputs separately from planned cache/output roots and candidate future E3 receipt bindings. Complete six selected source lineages before E3-G2/cache declarations; recorded hashes/epochs remain fixed |
| P02 | Cache format, storage dtype/layout/chunk size; extraction batch/worker policy and recovery boundary | Review lossless preservation and complete ordered coverage before E3-G1 implementation contract and relevant target acceptance |
| P03 | Float-logit/NLL equivalence tolerances and replay comparison details | Freeze before acceptance/cache use; never waive discrete prediction, geometry or coverage errors |
| P04 | Bounded pilot source lineage/cache subset, pilot seed/stream role and actual target scope | Explicit candidate before E3-G1; approve its execution at E3-PILOT-AUTH; source-final pairing remains 2001/3001/4001 |
| P05 | Pilot candidate protocols, maximum trials/draws/updates/validations and convergence/selection/tie criteria | Complete bounded envelope before pilot authorization; select one common official protocol, with no independent family/arm LR search |
| P06 | One shared head initialization rule and exact initialization/RNG stream binding | Pilot/freeze explicitly; allow mathematically required fan-in-dependent bounds; same actual weight/bias state within each pair; no source-head warm-start |
| P07 | Common head optimizer family and numeric parameters; LR, weight/bias decay, schedule/warmup/floor and clipping | Define trial values in the pilot envelope, lock one common set for all four arms at HEAD-FREEZE; no silent source AdamW/LR adoption |
| P08 | Common head microbatch, accumulation, budget basis, total windows/draws/presentations/updates and validation cadence | Exact matched accounting/tails across all four arms at HEAD-FREEZE; no automatic eight epochs or 139992 updates |
| P09 | cRT RNG API/stream keys, deterministic class/image index/draw mechanics and serialization | Specify before pilot implementation/acceptance; preserve fixed two-stage probabilities and deterministic replay |
| P10 | E3 head CPU/GPU execution, exact loader/thread/runtime/allocation/walltime policy and commit/replay boundary | Freeze applicable pilot values before implementation/acceptance and final values before final authorization; retain one common head numerical policy and deterministic FP32 requirements |
| P11 | Concrete E3 contract/cache/run/report schemas, paths, consumed-source closure and approval/attempt bindings | Prepare in the separately authorized plan/contracts; require one common official head protocol; actual code/device/cache/head hashes are later observations, never invented now |

**Protocol sharing is FIXED, not PENDING:** E3-E1-N, E3-E1-cRT, E3-E2-N and E3-E2-cRT use one common official head protocol. Only the mathematically required dimensional differences in Section 7 are permitted.

Items needed to execute the pilot must be fully resolved in its approved contract even when the **final selected** LR/optimizer/init/budget remains PENDING until the pilot finishes. Conversely, final declarations must reject any unresolved official field after HEAD-FREEZE. A `PENDING`, `null`, missing value, “use default” or source-recipe fallback is not an executable configuration.

The following numeric values are already inherited/derived and are **not pending searches**: source seeds 2001/3001/4001; 124 labels; 34997/7497 train/validation rows; 768/1536 feature dimensions; fixed view sizes/count/order; class probability 1/124; selected source epochs/hashes; new-head parameter-count oracles; fixed bucket coverage; checkpoint tolerance 1e-6/earlier; ddof=1/four-decimal percentage display. Existing tuning ID 1001 and source training LRs are historical facts, not selected E3 pilot/head values.

## 15. Acceptance requirements for later implementation

The following behavioral evidence is required after implementation is authorized.

| Boundary | Required acceptance evidence |
|---|---|
| Source loading/freeze | Actual selected-binary/report/contract lineage; strict correct family/seed/head shape; every representation parameter/buffer unchanged; zero representation gradients/trainables; original head preserved |
| Feature definition | E1 pooling/final LayerNorm boundary; E2 shared encoder, per-view normalization, batch-major local mean/global-first concatenation; both raw layouts; no train jitter/stochastic depth |
| Cache integrity | Full train/val membership and ordered labels, dimensions/finiteness/indices/checksums; reject wrong source/seed/split, stale/reordered/partial/mixed chunks and test features before access |
| Replay | Frozen numerical policy and actual source-original logits/predictions/metric equivalence; mismatches block acceptance rather than replace references |
| Paired heads | Exact equal reset tensor hashes within each pair; standalone linear weight/bias only; correct 95,356/190,588 counts; representation/cache bytes unchanged after training |
| Common official head protocol | All four retraining arms share optimizer family/configuration, LR/schedule, weight-decay policy, batch/accumulation, presentation/update budget, validation cadence, checkpoint comparator and numerical policy; only mathematically required dimensional differences are accepted |
| Sampling | Natural complete permutations; full class-uniform then within-class-uniform replacement; all 124 classes retained; independent reproducible streams and preserved class/image draw cursors; no claim of exact empirical balance |
| Budget/loss/selection | Matched presentations/updates/tails/LR trajectories and validations; true group sample means; full fixed budget despite flat/decreasing F1; chronological tau/earlier comparator and one selected head for every metric |
| Recovery | Initial/committed-boundary restore and deterministic replay, including cRT draws, RNG/loader and partial-group discard; wrong arm/cache/seed/protocol or best-as-resume rejected; no extension after completion |
| Evidence/reporting | Complete prediction-derived metrics and content-validated CSVs beyond hashes; source-schema differences handled explicitly; no partial-seed aggregate, ensemble or invalid paired deltas |
| Authorization/isolation | E3-specific stage gates checked before allocation/data access; no receipt bypass; existing E1/E1-HR/E2 and frozen artifacts unchanged; final test rejected before opening |

Correctness acceptance does not require beating an original head or another arm. Target acceptance does not prove head convergence or statistical significance. Synthetic CPU fixtures are nonofficial; GPU skips, prior experiment receipts and source closeout confirmations cannot substitute for E3 target evidence.
