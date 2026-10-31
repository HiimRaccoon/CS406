# E1 — ConvNeXt-Tiny full-image baseline specification

**DRAFT FOR USER REVIEW — chưa frozen, chưa authorize implementation hoặc official runs.**

Ngày soạn: **2026-10-31**, đối chiếu repo tới commit **`d055f5c5abe2915ad5329e7bc5c7d5edcc1b9ce9`**. Tài liệu định nghĩa E1 sau khi UIT pilot đã hoàn tất. Phạm vi task hiện tại chỉ viết spec; không tạo training code, runnable config, training recipe hoặc job. Các mục **NEEDS REVIEW** là điều kiện phải giải quyết trước official runs, không phải chỗ để implementation tự chọn defaults.

## 1. Mục đích, nguồn chuẩn và giới hạn

E1 là baseline **country/territory classification 124 labels**, dùng một full-image view **896×384**, làm reference cho E1-HR/E2 và representation nguồn cho E3 về sau. Mục tiêu là benchmark classification performance trên **exact fixed split hiện có**, phân tích overall accuracy và long-tail recognition; không đặt một single winner metric giữa các phương pháp.

Nguồn được dùng theo vai trò:

| Nguồn | Thẩm quyền trong E1 |
|---|---|
| [Master spec](../master_spec.md), [Grillme](../grillme.md) | Các quyết định research/architecture đã thống nhất; master vẫn draft |
| [Frozen preprocessing config](../../configs/preprocessing/geoguessr_v1.json), [preprocessing spec](../preprocessing_v1.md) | Contract deterministic preprocessing v1; không sửa để phù hợp implementation |
| [Split provenance](../../artifacts/baseline124/provenance.json) và bốn fixed artifacts | Exact membership, label IDs, counts và fingerprints |
| [Pilot plan](../pilot_plan.md), [post-execution review](../pilot_config_review.md), [pilot report](../../reports/pilot/pilot_results.md) | Trạng thái pilot đã hoàn tất, các quyết định sau pilot và phần official còn mở |
| [Server runbook](../server_pilot_runbook.md), archived configs/history, `src/pilot/` | Bằng chứng runtime và reference implementation; không tự trở thành official E1 framework |

Các câu trong Grillme/preprocessing/runbook còn ghi “pilot deferred”, “chưa chạy UIT” hoặc “E1-HR chưa chốt” được hiểu theo thời điểm tài liệu lịch sử. Trạng thái post-execution mới hơn là conformance PASS, strict P0 PASS, bounded P1 COMPLETE; không yêu cầu P1-v2. Frozen preprocessing v1 không thay đổi vì cập nhật trạng thái này.

**Ngoài scope E1:** local views/fusion, E1-HR training, cRT/head retraining/cache, class balancing, LLRD, pretrained comparison, strong augmentation, geometry search, test-driven tuning và robustness/unseen-location study. Viết spec này không mở thêm experiment hoặc ablation.

## 2. Trạng thái các quyết định

| Thành phần | Trạng thái |
|---|---|
| Exact split, 124-label mapping, preprocessing v1 | FROZEN tại các nguồn chuẩn |
| E1 architecture, pretrained starting point, natural CE, max drop-path 0.1, decay 0.05, head LR multiplier 1 | Đã thống nhất; ghi lại trong bản spec chờ review |
| Brightness/contrast amplitudes 0.10/0.10 | REVIEWED / ACCEPTED từ train-only preview; P1 đã dùng |
| Official max_epochs | **8**, đã được user review/freeze sau pilot; là trần budget chung E1/E1-HR/E2 |
| LR tuning và seeds protocol | **4 LR × 1 tuning seed**, rồi selected LR × **3 final training seeds** |
| Exact four-LR values và numeric official seed IDs | **NEEDS REVIEW**, chưa được chọn |
| Official batch, runtime, init, optimizer details, scheduler và stopping details | **NEEDS REVIEW** theo bảng mục 12; pilot cung cấp reference, không tự promote |
| Master/spec approval, official implementation, smoke, tuning, final runs, test evaluation | Chưa hoàn thành |

## 3. Dataset, labels và quyền truy cập split

Reuse nguyên byte `artifacts/baseline124/{manifest.csv,summary.json,labels.json,class_support.csv}`. Nguồn cũ: `HiimRaccoon/cs406-geoguessr`, commit **`33cb29d72974f1677efab0f56ed92360e5a9f712`**. Không regenerate membership từ seed **406**, re-stratify, re-dedup để đổi split hoặc sort lại label mapping.

| Split | Unique images | GT-supported labels | Quyền trong E1 development/tuning |
|---|---:|---:|---|
| train | 34997 | 124 | Training; natural sampling; train-only diagnostic samples |
| val | 7497 | 106 | Full deterministic validation; checkpoint/LR selection |
| test | 7502 | 111 | Chỉ metadata; ảnh khóa đến final evaluation được authorize riêng |
| Tổng | 49996 | 124 outputs | Exact fixed membership |

Dataset raw có 49997 ảnh, đã loại **1 byte duplicate**; grouping cũ chỉ dựa SHA-256. Không claim geographic independence hoặc panorama-disjoint split. Không lọc extreme-tail khỏi training/evaluation. E1 luôn giữ đủ **124 logits**, kể cả các lớp không có GT trong validation/test.

`labels.json` là list: **index chính là label_id**, từ 0 đến 123. Ví dụ kiểm tra mapping: `0=Aland`, `120=United States`, `123=Vietnam`; không chuẩn hóa lại tên country/territory. Runtime data root chứa trực tiếp các label folders; relative paths trong manifest giữ nguyên. `data_root_at_preparation` trong summary là lịch sử, không phải path bắt buộc trên UIT.

Trước model allocation phải kiểm:

- SHA-256/bytes của bốn artifacts với provenance và frozen config; summary fingerprints, split counts, class coverage, từng `label_id ↔ label`, class support và unique paths/image hashes phải nhất quán.
- Dataset root resolve đúng manifest; chặn path traversal, file thiếu/corrupt, sai decoded layout hoặc label. Báo lỗi và ghi failure; không bỏ/replacement mẫu im lặng.
- Có valid target conformance/hash-audit record cho đúng dataset root và packages; audit đủ **34997 train + 7497 val**, không đọc test bytes. Dataset immutable sau audit; audit lại khi root/content thay đổi, không hash toàn bộ mỗi epoch.
- Train/validation loader từ chối `split=test` trước khi mở file. Đọc test membership/counts từ artifacts không cấp phép mở ảnh. Final test evaluator thuộc bước sau có authorization riêng.

Full hashes không duplicate ở đây: dùng [provenance](../../artifacts/baseline124/provenance.json) và `source_split.sha256` trong frozen config làm nguồn kiểm chứng.

## 4. Input và preprocessing E1

Preprocessing ID **`geoguessr-ui-global-local-v1`**, config SHA-256:

`4f49dd41738168cfc65d4322fbe1b659ef5595229bb487ad86dd671a26a8903f`

E1 chỉ lấy **global view** từ [reusable preprocessing](../../src/preprocessing/geoguessr_v1.py). Không tạo local views, không dùng PNG preview làm input, không dùng `weights.transforms()`, center-crop 224, resize kéo giãn hoặc backend thay thế.

Trình tự cho một ảnh:

1. Pillow decode RGB, không EXIF transpose; nhận đúng layout decoded `1536×662` hoặc `1030×703`.
2. `pil_to_tensor` → CPU RGB CHW **float32 [0,1]**; không intermediate uint8 requantization hoặc preprocessing AMP.
3. Bỏ footer **14 px**. Tạo valid-scene mask từ UI rectangles trong frozen config; giữ nguyên navigation arrows theo ngoại lệ đã review.
4. Train: áp dụng một draw brightness/contrast trên valid scene, brightness trước contrast. Exclude UI khỏi statistics; fill UI **sau** jitter bằng exact mean. Val/final test: không jitter.
5. Preserve-aspect-ratio letterbox về **WH=[896,384]**, tensor bilinear CPU float32, `antialias=True`; half-up rounding `floor(value+0.5)`, centered placement, remainder right/bottom.
6. Normalize sau resize/padding bằng ImageNet mean **[0.485,0.456,0.406]**, std **[0.229,0.224,0.225]**, `inplace=False`.

UI fill và letterbox fill là exact mean trên float32, không làm tròn thành màu uint8; không hard-remask biên sau resize. Pixel nội thất mask/padding gần zero sau normalize; antialias có thể trộn scene ở biên.

| Layout raw | Frame sau footer | Resized content | Padding L/T/R/B | Output CHW |
|---|---|---|---|---|
| 1536×662 | 1536×648 | 896×378 | 0/3/0/3 | [3,384,896] |
| 1030×703 | 1030×689 | 574×384 | 161/0/161/0 | [3,384,896] |

Mask coordinates/công thức photometric lấy đúng frozen config, không tạo một bản tọa độ độc lập trong E1. Navigation arrows và một số watermark được giữ có chủ ý; không claim loại sạch mọi UI.

### Augmentation

Train brightness và contrast có amplitude **0.10**, tương ứng mỗi factor trong **[0.90,1.10]**; hai factors được sample cho mỗi original image. Đề xuất giữ hai independent uniform draws của pilot, thuộc RNG adoption R7 trước official runs. Dùng công thức valid-scene brightness/contrast đã freeze, không thay bằng `ColorJitter` mặc định có order/statistics khác. Val/test deterministic, không TTA. Không flip, random crop, RandAugment, Random Erasing, Mixup/CutMix hoặc label smoothing trong main E1.

Đề xuất tái dùng RNG key `(training_seed, epoch, relative_path, "jitter")` và epoch-order seeding của pilot để jitter không phụ thuộc persistent-worker scheduling; **official RNG policy cần review tại mục 12**. Không dùng Python hash không ổn định để derive sample seeds.

## 5. Model contract

```mermaid
flowchart LR
    A[Raw RGB screenshot] --> B[Frozen UI preprocessing + E1 global 896 x 384]
    B --> C[ConvNeXt-Tiny features]
    C --> D[Global average pooling + pretrained final LayerNorm]
    D --> E[Flatten: 768-d representation]
    E --> F[New Linear 768 to 124, with bias]
    F --> G[124 raw logits]
```

| Boundary | Contract |
|---|---|
| Model input | Batched floating tensor [N,3,384,896]; một view/original image |
| Feature extractor | ConvNeXt-Tiny pretrained features, không thay block/depth/channel design |
| Representation | Giữ pretrained global average pooling và final LayerNorm; flatten thành [N,768] |
| Classifier | Linear(768,124), bias=True; mới khởi tạo, không giữ ImageNet 1000-class linear head |
| Forward output | [N,124] raw logits; không softmax trước CE |
| Trainable | Toàn bộ representation + new classifier từ epoch 1 |
| Stochastic depth | Explicit max **0.1**, cùng TorchVision block-index linear schedule 0→0.1 đã chọn; không tuning schedule/rate riêng |
| Evaluation mode | Tắt stochastic depth và train-time randomness trước validation/final inference |

Starting point: **`torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`**, filename **`convnext_tiny-983f1562.pth`**, full reviewed SHA-256:

`983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`

Kiểm exact filename, existence và full digest trước load; không chỉ prefix `983f1562`, không default alias, không tự download checkpoint khác khi thiếu file. Strict-load pretrained state, giữ pretrained representation/final LayerNorm, chỉ thay classification linear head. Mỗi tuning/final run khởi tạo độc lập từ cùng file, optimizer/scheduler mới; không warm-start từ P0/P1, trial LR khác hoặc checkpoint GeoGuessr cũ.

New-head initialization phải explicit, deterministic và log exact rule/framework/seed. Pilot đã dùng `linear_default`: kaiming_uniform_(a=sqrt(5)), bias U(-1/sqrt(768),+1/sqrt(768)), isolated CPU RNG rồi restore outer RNG. **Đề xuất giữ rule này; adoption và seed binding chưa được freeze cho official E1.** Không hard-code pilot init seed 3406.

## 6. Loss và optimization invariants

- **Standard unweighted CE** trên natural training distribution, `label_smoothing=0`; không class weighting, `sqrt_inverse`, oversampling hoặc balanced sampler. Long-tail treatment chủ động cô lập sang E3.
- Mỗi completed training epoch duyệt đủ **34997 unique train images một lần**, shuffle theo training seed/epoch, không replacement; giữ tail batch (`drop_last=False`). Validation đủ 7497 ảnh, không shuffle/augmentation.
- **AdamW**; fixed weight decay **0.05** cho trainable Conv/Linear weights, không decay bias hoặc LayerNorm. Explicit optimizer groups phải bao phủ mỗi trainable parameter đúng một lần. Pilot cũng không decay ConvNeXt `layer_scale`; official enumeration/adoption phải được review, không bỏ hoặc nhóm trùng parameter do tên module khác.
- `LR_head = LR_backbone`, multiplier **1**, không LLRD, không freeze-head-first phase. Chỉ tuning một base LR mỗi architecture; mọi parameter group dùng cùng scheduler trajectory.
- **LR warmup rồi cosine decay**, schedule theo optimizer updates. AdamW betas/eps, clipping/numerics và scheduler numeric values không lấy library defaults hoặc copy pilot im lặng; xem mục 12.

Train CE và validation NLL là sample mean, không average không trọng số các batch means. Mọi nonfinite loss/logits/gradient hoặc OOM phải fail/report; không silently skip update, bỏ batch hoặc đổi microbatch/resolution/LR để vẫn trả một success run.

### Batch và optimizer-update accounting

Với single GPU, microbatch `m`, accumulation `a`, nominal effective batch **B=m×a original images**. Loss của một accumulation group có `D` actual images là tổng CE trên group chia **D**, kể cả tail microbatch/group. Chỉ optimizer-step/scheduler-step sau khi group hoàn tất; validation không advance scheduler.

Với natural epoch, giữ tail:

```text
microbatches_per_epoch = ceil(34997 / m)
updates_per_epoch = ceil(microbatches_per_epoch / a)
planned_max_updates = updates_per_epoch * 8
```

**Pilot** đã dùng `m=2,a=1,B=2`. **Nếu** official review giữ nguyên thì 17499 updates/epoch và tối đa **139992** completed updates cho 8 epochs; đây là phép tính có điều kiện, **chưa freeze official batch**. Không dùng horizon P1 **52497** updates cho official 8 epochs. Future E1-HR/E2 phải giữ common effective batch và budget đã review; không tự ưu tiên E1 bằng batch/horizon/tuning budget khác.

## 7. Official run protocol

### 7.1 Preflight và smoke

Trước official execution: spec được review, numeric contract được freeze/version/hash, dataset/preprocessing/pretrained/environment được verify, implementation/smoke được review riêng. Smoke dùng synthetic inputs hoặc train-only samples để kiểm wiring/numerics/resume; không test access, không tính smoke thành final seed hoặc trial thắng.

Pilot hoàn tất không yêu cầu chạy P1-v2. Nếu implementation/environment khác làm cần acceptance check thì chỉ kiểm đúng phần thay đổi và ghi bằng chứng; không mở thêm architecture/optimization search.

### 7.2 LR tuning

E1 chạy đúng **4 distinct positive base LR values × 1 official tuning seed**, cùng maximum **8 epochs**, warmup/cosine, stopping/checkpoint rule và effective batch đã freeze. Exact grid và seed còn NEEDS REVIEW. Grid phải chung với E1-HR/E2 trước official search; không thêm trial E1 vì score chưa như mong muốn.

Mỗi trial chọn checkpoint theo mục 8; chọn LR bằng **supported validation Macro-F1 tại checkpoint đó**. Lưu/công bố đủ cả bốn LR, outcomes, best epoch/metrics và history của trial không được chọn. Tie rule giữa LR trials phải xác định trước. Failure/retry là record riêng với nguyên nhân; không bỏ failure khỏi search ledger hoặc âm thầm thay LR thứ năm.

### 7.3 Final training

Sau LR selection, freeze selected configuration rồi train **3 official final seeds** độc lập, từ cùng ImageNet pretrained file. Không chạy `4 LR × 3 seeds`, không chọn best seed theo test, không coi pilot checkpoint là một final seed. Official tuning/final numeric seed IDs và quan hệ giữa các seed sets phải được review trước; không mặc định seed tuning khác/trùng final set hoặc seed 3406 được dùng lại.

Không warm-start final run từ winning LR trial. Final seeds không mở lại LR/grid/augmentation; khác biệt seed chỉ ảnh hưởng optimization/batch/augmentation/init randomness theo policy chung. Split seed 406 và membership giữ nguyên.

Test chỉ được mở trong final evaluation phase sau khi protocol/checkpoints đã khóa và có authorization. Trong task spec, implementation, smoke, tuning và final training/validation, test images vẫn không được mở. Không refit thêm trên train+val vì điều đó đổi fixed benchmark protocol.

## 8. Checkpoint selection, early stopping và scheduler

**Đã chốt metric:** validation **Macro-F1 trên các lớp có ground-truth support**. Full validation ở mỗi completed epoch; mọi metrics được report cho một run phải xuất phát từ **chính `best.pt` được chọn**, không lấy best Top-1/Top-k của epoch khác. `last.pt` phục vụ resume, không tự là reported model.

Trước official search phải chốt và serialize rõ:

| Rule | Yêu cầu |
|---|---|
| Checkpoint tolerance `tau` | Giá trị rất nhỏ trên thang metric [0,1]; exact numeric còn NEEDS REVIEW |
| Tie-break | Một rule cố định: earlier hoặc NLL trong tolerance; official lựa chọn còn NEEDS REVIEW |
| Patience / min_delta | Numeric, monitoring-start policy và cách reset counter phải freeze trước trials |
| LR-trial tie-break | Không chọn tùy ý hoặc dựa test khi checkpoint scores bằng/gần nhau |
| Scheduler horizon | Tính từ maximum 8 epochs và reviewed batch; không đổi horizon theo observed best epoch hoặc early stop |

Reference comparator của pilot: replace checkpoint khi `current_f1 - selected_f1 > tau`; trong tolerance, `earlier` giữ checkpoint đã chọn, còn `nll` chỉ replace khi NLL tốt hơn. Pilot dùng **tau=1e-6, earlier**; **đề xuất tái dùng**, chưa coi là official freeze. Nếu dùng rule này, “best” có thể thấp hơn peak đơn lẻ trong tolerance; report selected score/epoch và full history để minh bạch.

Early-stopping tracking phải tách bạch với checkpoint tolerance/tie-break: xác định metric reference, improvement threshold, counter reset và điều kiện dừng sau patience completed validations. Không để cải thiện NLL trong F1 tie tự reset patience nếu policy không quy định. Các chi tiết này cần review, không tự suy ra patience từ pilot chỉ có 3 epochs.

Maximum **8** là upper bound, không ép mọi run phải train đúng 8 nếu reviewed early-stopping rule dừng trước. Lưu `completed_epochs`, `optimizer_updates`, `best_epoch`, `stop_reason` và history đầy đủ. Không kéo dài horizon, reset warmup, chọn epoch bằng test hoặc tune stopping riêng sau khi thấy curve từng architecture.

## 9. Metrics và error analysis

Gọi `C[i,j]` là confusion count với GT=i, prediction=j trên **toàn split được evaluate**. Per-class F1 dùng đủ TP/FP/FN của class đó, kể cả FP đến từ GT class/bucket khác. Với GT support>0 mà không prediction đúng, F1=0. Với support=0, F1 lưu **null/not evaluated**, không ép zero để đưa vào supported macro.

| Output | Definition |
|---|---|
| Top-1 / Top-3 / Top-5 | Sample accuracy, rank đủ **124 outputs**; không mask classes thiếu evaluation support |
| Macro-F1 supported | Mean per-class F1 trên supported GT classes của split; validation denominator **106**, final test **111** theo metadata |
| Weighted-F1 | Per-class F1 weighted bởi **GT evaluation support**, không dùng train frequencies |
| NLL | Unweighted mean CE trên evaluation samples; diagnostic/possible predefined tie-break, không thay selection metric |
| Per-class CSV | label_id/name, train count/bucket, GT support, predicted count, precision, recall, F1; ghi rõ unsupported classes |

Main classification metrics report cùng nhau; không tuyên bố một phương pháp thắng duy nhất chỉ dựa một scalar. Không dùng average minibatch Macro-F1 hoặc lọc logits xuống 106/111 classes.

### Frequency-bucket contract

Membership theo **unique n_train sau dedup/fixed split**, không theo raw counts, val/test support hoặc số lần resampling. Metadata đã đọc từ [class_support.csv](../../artifacts/baseline124/class_support.csv):

| n_train bucket | Total classes | Unique train images | Val supported classes | Val samples |
|---|---:|---:|---:|---:|
| 1–19 | 42 | 243 | 24 | 49 |
| 20–99 | 37 | 2077 | 37 | 445 |
| 100–499 | 29 | 7049 | 29 | 1513 |
| ≥500 | 16 | 25628 | 16 | 5490 |
| 1–2, subset của 1–19 | 20 | 21 | 2 | 2 |

Mỗi evaluated bucket báo total classes, GT-supported classes, evaluation samples, Macro-F1 và Top-1/3/5. Bucket Macro-F1 lấy từ per-class F1 **đã tính trên toàn evaluation split** rồi average supported classes trong bucket; không lọc GT rows trước khi tính F1. Bucket Top-k tính trên samples có GT thuộc bucket, nhưng vẫn rank đủ 124 outputs. Bucket không có supported GT: metrics null, không fabricate zero.

Final E1 report mean ± standard deviation qua đủ **3 seeds**, kèm từng seed; phải khai báo convention tính std trong reporting contract trước aggregate. Không chọn seed tốt nhất, không average logits thành ensemble để thay cho mean individual-model scores. Full validation history/learning curves và confusion/most-confused pairs phải có để nhìn tính ổn định, không chỉ peak score. Extreme-tail 1–2 samples được diễn giải thận trọng.

## 10. Runtime, RNG, resume và output contract

### Runtime và reproducibility

Target là single UIT GPU/Slurm, dự kiến partition `normal`, GRES `gpu:2g.35gb:1`; GPU work trong allocation. Pilot ghi H200 MIG 2g.35gb, Python 3.11.17, torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, cuDNN 91002. Official environment/runtime adoption còn review; ghi resolved package/driver/device/allocation và actual Torch settings, không chỉ config mong muốn.

Đề xuất giữ runtime/loader deterministic FP32 pilot làm starting contract; bảng mục 12 liệt kê các giá trị để review **một lần**, không mở runtime ablation. Không tự đổi AMP, TF32, workers hoặc compile riêng E1 giữa trials. Dataset/preprocessing conformance phải phù hợp target packages/root; archived pass không thay check cần thiết khi các điều kiện đó đã đổi.

Official seed contract phải cover Python/NumPy/CPU/CUDA, classifier initialization, sample order, augmentation, worker/DataLoader generators và process environment. Đề xuất tái dùng deterministic derived-seed mechanics trong [pilot dataset](../../src/pilot/dataset.py), nhưng bind với **official seed IDs**, không hard-code 3406/PYTHONHASHSEED pilot. Persistent workers phải nhận đúng epoch; restart job không được lặp jitter/order của epoch 0.

### Checkpoint/resume

Official checkpoint có identity/type riêng, không nhận `kind=bounded_pilot_checkpoint` làm official resume. `last.pt` ghi atomic ở completed epoch boundary; lưu model, optimizer, scheduler, AMP/scaler nếu dùng, completed epoch/update counters, full resolved config/hash, data/labels/preprocessing/pretrained fingerprints, official source/environment identity, RNG/loader state, best/history và **early-stopping state**.

Resume đúng run từ `last.pt`, giữ cùng seed/config/horizon/source/runtime identity; reject mismatch. Không resume từ `best.pt` để quay lại thêm budget, không warm-start từ pilot/trial khác. Interrupted partial epoch được replay từ last committed boundary; không tính replay thành extra completed epoch hoặc extend 8-epoch budget. Có initial boundary checkpoint để recover khi epoch đầu chưa hoàn thành; không ghi đè run/trial cũ bằng run mới. Single-writer lock ngăn hai jobs cùng folder.

Bundle UIT hiện có không chứa một interrupted/resumed diagnostic riêng. Pilot checkpoint code và synthetic CPU tests là reference, không tự xác nhận CUDA resume equivalence của official E1; acceptance phải kiểm dưới reviewed runtime khi implementation được authorize. Điều này không yêu cầu thêm P1-v2 hoặc mở convergence study mới.

Pilot `identity()`/`validate_gates()` và checkpoint helpers là reference. **Không require official source fingerprint bằng pilot source fingerprint**: code official sẽ khác; lưu reference pilot identity và official implementation identity riêng. Gate phải kiểm các contract phù hợp, không bypass chỉ vì pilot source equality không còn đúng. Spec này không tự approve một refactor/permanent `src/data/`, `src/models/`, `src/training/` framework.

### Required artifacts sau implementation được authorize

Tên/format dưới đây là contract đầu ra để review, chưa có official run files được tạo:

| Artifact | Nội dung bắt buộc |
|---|---|
| Resolved config + recipe/spec reference/hash | stage tuning/final, run/trial ID, official seed, base LR, all numeric/runtime policies, immutable source/data/preprocessing/pretrained identity |
| Environment/model/optimizer metadata | Versions/GPU/allocation/env, architecture/head/drop-path/init rule, enumerated decay groups, parameter counts |
| `history.json`, `metrics.csv`, learning curves | Mọi completed epoch: train loss/count, updates, LR endpoints, validation metrics, timing, memory; không xóa epoch/trial không đẹp |
| `best.pt`, `last.pt` | Chọn/report và resume theo hai vai trò ở trên; full state cần thiết |
| `run_summary.json` | Success/failure/interrupted/early-stop status, planned/completed budget, selected epoch/score, stop/failure reason, test-access status |
| Tuning ledger | Bốn reviewed LR trials cùng outcomes, selected LR và rationale theo validation rule |
| Selected-checkpoint analysis | Validation overall/per-class/bucket metrics, predictions theo manifest path/label ID với Top-k, confusion matrix; test outputs chỉ ở final-evaluation phase |
| Three-seed report | Từng seed và mean ± std, coverage/limitations; không gọi pilot scores là official results |

Timing phải ghi scope: train-only, train+validation, checkpoint/report I/O hoặc full job walltime. Report throughput theo original images và peak allocated/reserved riêng; không dùng P0 VRAM/throughput làm acceptance score threshold của official E1. Efficiency chỉ secondary information.

E3 về sau có thể lấy representation trước classifier từ từng **selected official final-seed E1 checkpoint**; E1 giữ identity/checkpoints để trace được lineage. Không cache features hoặc implement E3 trong task/spec E1 này.

## 11. Pilot evidence: được phép kết luận gì

| Evidence E1 | Kết quả đã đọc | Giới hạn |
|---|---|---|
| [Strict P0 summary](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/run_summary.json), [config](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/config_resolved.json) | pass, assessed, criteria, full epoch; batch 2/1; train epoch 2080.53 s; steady-state 16.9549 img/s, full epoch 16.8212 img/s; reserved peak 1.99023 GiB window / 1.99414 GiB epoch | Feasibility của declared workload, không phải convergence hoặc maximum batch |
| [Bounded P1 config](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/config_resolved.json), [summary](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/run_summary.json) | complete 3/3, 52497 updates, diagnostic seed 3406/LR 1e-4; official_experiment=false | Bounded diagnostic, không phải official run hoặc 8-epoch evidence |

Các số P0 chỉ chứng minh workload/batch đã đo khả thi dưới policy đó; không chứng minh batch optimal. P1 kiểm optimization behavior hữu hạn, không chứng minh convergence hoặc official 8 epochs là tối ưu. Cả P0/P1 ghi `test_images_opened=0`; target conformance và tất cả pilot profiles đã pass theo [report](../../reports/pilot/pilot_results.md). Manifest evidence đã được sửa self-entry; checksum của **47 payload files PASS**.

[P1 history](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/history.json) và [CSV](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/metrics.csv) khớp:

| Epoch | Train CE | Val NLL | Val Top-1 | Val Macro-F1 supported |
|---|---:|---:|---:|---:|
| 1 | 1.956166 | 1.430387 | 0.615179 | 0.191439 |
| 2 | 0.990609 | 0.977231 | 0.727758 | 0.347943 |
| 3 | 0.378864 | 0.856378 | 0.761238 | 0.395442 |

Best diagnostic epoch=3; Top-3=0.898093, Top-5=0.936508, Weighted-F1=0.747391 tại cùng epoch. Reserved VRAM ổn định 2141192192 bytes qua ba epochs. E1 P1 data-wait fraction khoảng 16.4–19.2%; không suy ra loader đã optimal hoặc tự mở worker search. Không đặt acceptance requirement “official E1 phải vượt pilot 0.395442”; failure/success của implementation dựa correctness/protocol, không ép score mong muốn.

## 12. Review register trước official execution

Các đề xuất dưới đây nhằm hoàn thiện **một common recipe**, không tạo experiment branch. Chưa có `configs/training/training_recipe_v1` và không xuất partial recipe khi các field bắt buộc còn mở.

| ID | Cần review/freeze | Pilot evidence / đề xuất để review | Trạng thái |
|---|---|---|---|
| R1 | Exact 4 distinct base LR values; trial tie/failure accounting | Chỉ có diagnostic 1e-4; **không tự tạo grid** hoặc refine theo pilot validation score | NEEDS REVIEW |
| R2 | Exact 1 tuning + 3 final seed IDs; quan hệ seed sets/init seed binding | Pilot 3406, split 406; **không tự chọn official IDs** | NEEDS REVIEW |
| R3 | Common effective batch và E1 microbatch/accumulation | Đề xuất giữ pilot **B=2, m=2, a=1**, đã strict-pass cả ba profiles; chưa promote | NEEDS REVIEW |
| R4 | Warmup duration/start factor; cosine formula/floor/horizon 8 epochs | Pilot warmup 1000, factor 0.1, floor 1e-6, horizon 3 epochs. Đề xuất giữ update-based warmup/cosine semantics; official numeric values phải review cùng grid/batch | NEEDS REVIEW |
| R5 | Patience, min_delta, monitoring start/reset; checkpoint tau/tie; LR-trial tie | Pilot không early-stop; tau=1e-6/earlier chỉ là reference/đề xuất cho checkpoint, không quyết định patience | NEEDS REVIEW |
| R6 | AdamW betas/eps, complete decay enumeration, clipping | Đề xuất betas [0.9,0.999], eps 1e-8, clipping off; decay 0.05/no-bias/no-LN đã chốt; pilot layer_scale no decay cần ghi explicit trong adoption | NEEDS REVIEW phần adoption |
| R7 | New-classifier initialization và RNG/worker/resume contract | Đề xuất linear_default isolated CPU RNG, sample/epoch keyed jitter/order của pilot; bind official seeds, verify resume/stopping behavior | NEEDS REVIEW |
| R8 | Official runtime/environment và Slurm/output adoption | Đề xuất FP32, AMP/scaler/TF32/compile off; highest matmul, cuDNN benchmark off, deterministic true/warn_only false; workers=2, pin/persistent=true, prefetch=2, threads=2; target package set đã đo. Review env vars/allocation/walltime, không copy pilot seed vào env | NEEDS REVIEW |
| R9 | Reporting convention và final-evaluation handoff | Freeze std convention, LR/metric tie conventions và test-unlock ownership trước aggregate/evaluation; không mở test trong training | NEEDS REVIEW |

**Đã resolved, không grill lại:** architecture, full-image 896×384, fixed split/labels, preprocessing v1, exact pretrained file/full hash, natural unweighted CE, drop-path max 0.1, weight decay 0.05, head multiplier 1, augmentation 0.10/0.10, max_epochs **8**, 4-LR/1-seed rồi 3-final-seed protocol. Không cần P1-v2 để viết/review E1 spec.

Review spec có thể thống nhất architecture/functional requirements trước, nhưng official LR search/final runs phải đợi R1–R8 được resolve thành versioned complete numeric recipe và các conventions liên quan ở R9 được ghi explicit. Không để `null`, “use default” hoặc pilot fallback tự resolve một official required field.

## 13. Implementation acceptance khi được yêu cầu sau review

| Acceptance check | Bằng chứng cần có |
|---|---|
| Dataset isolation | Exact fingerprints/counts/label mapping; train/val complete; test access từ chối trước file open |
| Preprocessing | Hai raw layouts → đúng [3,384,896], fill/padding/rounding/backend đúng v1; jitter không thay mask/padding; deterministic eval |
| Architecture/load | Full pretrained SHA mandatory, mismatch fail; giữ pretrained representation/LN; logits [N,124]; full fine-tuning, drop-path schedule/eval behavior đúng |
| Sampling/loss/update | Natural epoch đủ samples; unweighted CE; no duplicate/omitted optimizer parameters; accumulation tương đương sample mean, nhất là short tail |
| Schedule/stopping | Warmup/cosine update counts khớp reviewed 8-epoch horizon; checkpoint F1/tolerance/tie và patience đúng các edge cases; history không bị cắt |
| Resume | Interrupted/resumed diagnostic tương thích full run về order/jitter/RNG, LR, optimizer counters, best/early-stop state; từ chối wrong identity/pilot checkpoint |
| Metrics | Supported macro/weighted F1 từ full-split counts, unsupported null, Top-k đủ 124; bucket false positives không bị mất; report từ selected checkpoint |
| Trial/final separation | Bốn LR trial records, một tuning seed; ba final runs khởi tạo mới, config frozen; không dùng test hoặc cherry-pick seed |
| Artifacts/failure | Atomic checkpoints, nonfinite/OOM/failure statuses rõ, no overwrite; metadata/provenance trace được từ report tới selected checkpoint |

Test và smoke ở phase implementation phải kiểm behavior thực, không lấy class names hoặc pilot source layout làm bằng chứng E1 đã được implement. Kiểm correctness dùng synthetic/train-only inputs; không chạy training thật trong task viết spec này.

## 14. Checklist review và handoff

- [x] Đã đọc 26 nguồn được yêu cầu; định nghĩa E1 architecture/data/preprocessing/loss/reporting trong bản draft.
- [x] Tách fixed research decisions, actual pilot records và official fields còn NEEDS REVIEW.
- [x] Giữ exact split/preprocessing, không thêm experiment/ablation hoặc promote pilot LR/seed.
- [ ] User review/approve `docs/E1/specs.md` và quyết định các mục R1–R9 theo scope tương ứng.
- [ ] Complete versioned numeric training recipe được review/freeze, hash và link nguồn chuẩn.
- [ ] Implementation plan/review và authorization implementation riêng.
- [ ] E1 implementation, acceptance/smoke, official LR search và final 3 seeds.
- [ ] Final test evaluation được mở ở phase riêng sau khi checkpoints/protocol khóa.

Việc tạo/commit bản spec này không freeze master spec, không cập nhật checklist official E1 thành complete và không authorize implementation. Bước tiếp theo là **review bản spec và những quyết định còn mở**, không tự chuyển sang viết code hoặc submit job.

## 15. Source index — 26 file đã đọc

| # | File | Vai trò |
|---|---|---|
| 1 | [docs/master_spec.md](../master_spec.md) | Main design, current status, official freeze boundaries |
| 2 | [docs/grillme.md](../grillme.md) | Review decisions/history và scope controls |
| 3 | [docs/preprocessing_v1.md](../preprocessing_v1.md) | Frozen input/UI/resize contract |
| 4 | [docs/pilot_plan.md](../pilot_plan.md) | Completed gates và pending official recipe |
| 5 | [docs/pilot_config_review.md](../pilot_config_review.md) | Historical proposals vs post-execution resolution |
| 6 | [docs/server_pilot_runbook.md](../server_pilot_runbook.md) | Runtime/checkpoint/Slurm reference, chưa official launcher |
| 7 | [reports/pilot/pilot_results.md](../../reports/pilot/pilot_results.md) | Verified diagnostic results/limits và post-P1 review |
| 8 | [P1 E1 config_resolved.json](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/config_resolved.json) | Actual bounded pilot recipe |
| 9 | [P1 E1 run_summary.json](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/run_summary.json) | Completion/identity/best diagnostic checkpoint |
| 10 | [P1 E1 history.json](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/history.json) | Three epoch curves, counts, 124-class metrics/epoch |
| 11 | [P1 E1 metrics.csv](../../reports/pilot/evidence/uit_2026-10-31/p1/e1/metrics.csv) | Cross-check scalar metrics/LR/update/timing |
| 12 | [P0 E1 run_summary.json](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/run_summary.json) | Strict full-epoch feasibility evidence |
| 13 | [P0 E1 config_resolved.json](../../reports/pilot/evidence/uit_2026-10-31/p0/e1/config_resolved.json) | Declared batch/runtime/criteria/windows |
| 14 | [configs/preprocessing/geoguessr_v1.json](../../configs/preprocessing/geoguessr_v1.json) | Exact frozen geometry/UI/photometric contract |
| 15 | [artifacts/baseline124/summary.json](../../artifacts/baseline124/summary.json) | Counts, dedup, SHA-only grouping limitation |
| 16 | [artifacts/baseline124/labels.json](../../artifacts/baseline124/labels.json) | Exact 124-label index mapping |
| 17 | [artifacts/baseline124/class_support.csv](../../artifacts/baseline124/class_support.csv) | Train-frequency buckets/evaluation support |
| 18 | [artifacts/baseline124/provenance.json](../../artifacts/baseline124/provenance.json) | Source commit, artifact hashes/bytes |
| 19 | [src/pilot/p1_train.py](../../src/pilot/p1_train.py) | Bounded runner: no official search/early stopping |
| 20 | [src/pilot/engine.py](../../src/pilot/engine.py) | Sample-weighted CE, accumulation, update scheduler/tie behavior |
| 21 | [src/pilot/models.py](../../src/pilot/models.py) | Backbone/head/load/init/optimizer reference |
| 22 | [src/pilot/dataset.py](../../src/pilot/dataset.py) | Manifest integrity, isolation, natural sampler, seed/worker behavior |
| 23 | [src/pilot/checkpoint.py](../../src/pilot/checkpoint.py) | Atomic full-state epoch-boundary resume reference |
| 24 | [src/pilot/runtime.py](../../src/pilot/runtime.py) | Runtime/gates/identity/source fingerprints |
| 25 | [src/pilot/config.py](../../src/pilot/config.py) | Explicit unresolved-value validation; schema chỉ dành pilot |
| 26 | [src/preprocessing/geoguessr_v1.py](../../src/preprocessing/geoguessr_v1.py) | Reusable frozen preprocessing implementation |
