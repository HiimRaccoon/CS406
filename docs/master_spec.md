# CS406 GeoGuessr — master specification

**DRAFT FOR USER REVIEW — chưa frozen, chưa authorize implementation**

Tài liệu trung tâm để review thiết kế E1 → E1-HR → E2 → E3. **Status update 2026-10-31:** UIT conformance, strict P0 và bounded P1 đã hoàn tất; xem [pilot results](../reports/pilot/pilot_results.md). Official specification/implementation là phase tiếp theo, **chưa bắt đầu**; task hiện tại chỉ cập nhật status/report, không tạo training code hoặc official configs.

Nguồn quyết định research/training là [Grillme đã review](grillme.md); nếu khác master spec cũ, Grillme mới hơn được ưu tiên. Nguồn chuẩn về split/preprocessing/geometry là [fixed artifacts](../artifacts/baseline124/README.md), [provenance](../artifacts/baseline124/provenance.json) và [preprocessing v1 config](../configs/preprocessing/geoguessr_v1.json). Freeze các nguồn này không đồng nghĩa master spec đã frozen.

## A. Project objective

**Task: GeoGuessr country/territory classification**, không phải GPS regression.

~~~text
one screenshot -> model -> one of 124 labels
~~~

Primary objective: **benchmark classification score trên exact fixed split hiện tại**.

Main reporting metrics: **Top-1, Top-3, Top-5, Macro-F1 trên supported ground-truth classes, Weighted-F1**. Report cùng nhau để phân tích accuracy/long-tail trade-off, không thiết lập single winner metric giữa các phương pháp.

Trong mỗi run, chọn checkpoint bằng **validation Macro-F1 trên supported ground-truth classes**. Mọi metrics của run lấy từ chính checkpoint đó; không lấy best Top-1 của epoch khác. Test chỉ final evaluation, không chọn architecture, resolution, crop geometry, augmentation, LR hoặc checkpoint.

Geographic independence chưa được chứng minh; exact deduplication không loại hết same-place/same-panorama overlap. Location-disjoint generalization không phải primary research question. Test cũ từng evaluate/inspect; hiện chưa có bằng chứng được dùng để tune, nên không mô tả test là chưa từng được xem.

## B. Architecture và workflow

### E1 — Full-image baseline

~~~text
Raw GeoGuessr image
        |
        v
Frozen UI preprocessing v1
        |
        v
Global view 896x384
        |
        v
ConvNeXt-Tiny representation
        |
        v
768-d feature
        |
        v
Linear classifier
        |
        v
124 classes
        |
        v
Standard CE (training)
~~~

Natural training distribution; không class weighting. Representation giữ pretrained feature extractor, global average pooling và final LayerNorm; thay linear ImageNet classifier bằng head cho 124 labels.

### E1-HR — Higher-resolution full-image control

~~~text
Raw image
    |
    v
Same frozen UI preprocessing v1
    |
    v
Higher-resolution full image
    |
    v
Same ConvNeXt-Tiny representation
    |
    v
768-d feature -> Linear classifier -> 124 classes
~~~

**1344×576 đã accepted sau strict UIT P0** theo first-candidate feasibility rule; xem [pilot results](../reports/pilot/pilot_results.md). Không compute-match với E2, không chọn bằng validation/test classification score.

Mục tiêu: kiểm tra việc **chỉ tăng full-image resolution** có cải thiện classification score không. Loss và sampling giống E1. FLOPs/latency/VRAM là thông tin efficiency bổ sung, không phải ràng buộc chọn resolution.

### E2 — Shared-encoder global/local representation

~~~text
Raw image -> Frozen UI preprocessing v1
                    |
                    +-> Global 896x384 -> f_theta -> G  (768-d)
                    |
                    +-> Local 1 320x320 -> f_theta -> L1 (768-d)
                    +-> Local 2 320x320 -> f_theta -> L2 (768-d)
                    +-> Local 3 320x320 -> f_theta -> L3 (768-d)
                    +-> Local 4 320x320 -> f_theta -> L4 (768-d)
                                                       |
                                                       v
                                              mean(L1,L2,L3,L4)
                                                       |
                           G --------------------------+
                                                       |
                                                       v
                                          concatenate [G; mean(L)]
                                                       |
                                                       v
                                              1536-d fused feature
                                                       |
                                                       v
                                              Linear classifier
                                                       |
                                                       v
                                                  124 classes
                                                       |
                                                       v
                                               one fused CE
~~~

Mọi f_theta ở trên là **cùng một ConvNeXt-Tiny encoder, cùng weights**, không phải các encoder riêng. Average bốn local features trước khi concatenate với global. Không positional identity của từng crop, auxiliary local CE hoặc extra fusion MLP/attention.

Local crops lấy trực tiếp từ **UI-processed raw image**, không từ resized global hoặc padded canvas. Crop deterministic, có overlap, cùng policy ở train/validation/test.

| Layout raw | Raw local crop | Local output |
|---|---|---|
| 1536×662 | 384×384 | 320×320 |
| 1030×703 | 402×402 | 320×320 |

Global output là **896×384**. Centers theo frame sau UI/footer crop:

- (0.20W, 0.60H)
- (0.40W, 0.60H)
- (0.60W, 0.60H)
- (0.80W, 0.50H)

W/H tương ứng frame 1536×648 hoặc 1030×689 sau footer crop. Integer crop rectangles và rounding dùng đúng [frozen config](../configs/preprocessing/geoguessr_v1.json), không tính lại geometry lúc training. Geometry không mở lại thành search; các candidate không chọn chỉ là visual design alternatives lịch sử.

### E3 — Long-tail classifier study

~~~text
Selected E1/E2 checkpoint, for each source seed
                    |
                    v
          Freeze representation + model.eval()
                    |
                    v
           Deterministic feature cache
                    |
                    v
        Reset linear classifier to shared initial state
                    |
          +---------+---------------------+
          |                               |
          v                               v
   Natural sampling              Class-uniform sampling
          |                               |
          v                               v
Natural head retraining                  cRT
          |                               |
          +------------- compare ---------+
~~~

Cache E1: **768-d feature trước classifier**. Cache E2: **1536-d fused feature trước classifier**. Mỗi train/validation image đúng một feature cố định, không jitter; evaluation mode tắt stochastic depth. Backbone/representation không được cập nhật.

Main E3 comparison gồm **E1 original, E1 natural head retraining, E1 cRT, E2 original, E2 natural head retraining, E2 cRT**. Original checkpoint giữ làm reference; natural retraining kiểm soát gain do reset/train lại head và thay đổi augmentation. Chưa có E1-HR+cRT.

cRT chọn lớp rồi ảnh:

~~~text
P(class) = 1/124
P(image | class) = 1/n_train[class]
~~~

Trong mỗi cặp natural/cRT của cùng representation/source seed, dùng cùng **source checkpoint, feature cache, initial head state/rule, optimizer, LR, training budget và checkpoint rule**. Chỉ sampling strategy khác nhau. Không chồng class-weighted CE hoặc các balancing methods khác vào cRT.

Head LR/budget/initialization được chốt qua **small head pilot**, không mở một full LR-search branch riêng cho E3. Số draws/steps và sampling mechanics phải explicit khi freeze head protocol. Extreme-tail 1–2 ảnh train có thể bị lặp nhiều; không diễn giải vài prediction đúng thành bằng chứng mạnh về generalization.

## C. Main experiment matrix

| ID | Representation | Resolution/views | Sampling | Trainable |
|---|---|---|---|---|
| E1 | Full image | 896×384 | Natural | Full model |
| E1-HR | Full image | 1344×576, accepted sau P0 | Natural | Full model |
| E2 | Global + 4 local | 896×384 + 4×320×320 | Natural | Shared full model |
| E3-E1-N | E1 frozen feature | Cached, 768-d | Natural | Head only |
| E3-E1-cRT | E1 frozen feature | Cached, 768-d | Class-uniform | Head only |
| E3-E2-N | E2 frozen fused feature | Cached, 1536-d | Natural | Head only |
| E3-E2-cRT | E2 frozen fused feature | Cached, 1536-d | Class-uniform | Head only |

Các controls và recipe choices dưới đây không tự động trở thành ablations. Thứ tự E1 → E1-HR → E2 là tiến trình nghiên cứu, không phải checkpoint lineage.

## D. FIXED — các design choices đã thống nhất

“FIXED” ghi nhận lựa chọn trong bản draft để review; không thay trạng thái master spec thành frozen. Phần fine-tuning dưới đây áp dụng E1/E1-HR/E2; numeric head recipe của E3 vẫn pending.

| Thành phần | Lựa chọn |
|---|---|
| Backbone | ConvNeXt-Tiny |
| Pretrained enum | torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 |
| Weight file | convnext_tiny-983f1562.pth |
| Starting point | E1/E1-HR/E2 khởi tạo độc lập từ cùng file; không kế thừa E1 checkpoint/optimizer/scheduler |
| Fine-tuning | Full model từ epoch 1, có LR warmup; không classifier-only phase |
| Optimizer | AdamW |
| Scheduler | LR warmup, sau đó cosine LR decay; official max_epochs=8 đã được user review sau pilot. Official warmup duration, cosine parameters và minimum/final LR nếu protocol cần vẫn NEEDS REVIEW |
| Layer learning rates | Không LLRD; LR_head = LR_backbone |
| Weight decay | 0.05 cho trainable Conv/Linear weights; không decay bias và LayerNorm parameters; explicit parameter groups |
| Stochastic depth | **max_drop_path_rate = 0.1** đã chốt; E1/E1-HR/E2 dùng cùng explicit depth schedule, không tuning riêng. Linear 0 → 0.1 theo block depth là schedule của implementation TorchVision ConvNeXt-Tiny được chọn ([nguồn](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py)), không phải một schedule/hyperparameter đã được separately optimized bằng thí nghiệm |
| Loss | Standard CE; E2 chỉ một CE sau fusion, không auxiliary local loss |
| Class weights/sampling | **Natural training distribution, không class weighting hoặc sqrt_inverse weighting** |
| Long-tail treatment | Chủ động cô lập sang E3; không mang class-weight recipe từ repo cũ sang |
| Train augmentation | Chỉ mild brightness/contrast jitter; một draw mỗi original image, consistent trên mọi E2 view, giữ nguyên mask/padding |
| Validation/test | Deterministic preprocessing, không augmentation |
| Normalization | ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225] |
| Pretrained transforms | Không dùng nguyên weights.transforms(); dùng frozen GeoGuessr pipeline |
| Không dùng trong main recipe | Flip, random crop augmentation, Mixup/CutMix, RandAugment, Random Erasing, label smoothing |

E1-HR/E2 không được hưởng thêm GeoGuessr training time từ E1. Shared controls còn gồm fixed split/labels, pretrained file, preprocessing v1, common tuning budget, effective batch theo ảnh gốc và checkpoint rule.

## E. LR tuning và training seeds

### Hyperparameter tuning

Sau khi common training budget đã được freeze:

~~~text
E1      -> 4 LR candidates x 1 tuning seed
E1-HR   -> 4 LR candidates x 1 tuning seed
E2      -> 4 LR candidates x 1 tuning seed
~~~

Cùng bốn LR candidates, tuning seed, số trial, optimizer/scheduler protocol, max-epoch budget, warmup/patience và checkpoint-selection rule. Mỗi architecture được chọn LR riêng theo **validation Macro-F1 supported**. Lưu/công bố mọi LR đã thử và LR được chọn.

**Không chạy 4 LR × 3 seeds.** Bốn numeric LR values hiện chưa được tự điền.

### Final runs

Sau khi chọn LR, freeze selected configuration:

~~~text
E1      -> best LR x 3 training seeds
E1-HR   -> best LR x 3 training seeds
E2      -> best LR x 3 training seeds
~~~

Report **mean ± std**, không cherry-pick seed có test score cao nhất. Training seeds không thay fixed data split. Mỗi source seed có các E3 natural/cRT branches tương ứng; không trộn source features giữa seeds.

Effective batch size tính theo **original images**; năm E2 views vẫn là một mẫu. Microbatch/accumulation có thể khác để giữ cùng effective batch. Không tự cắt epochs/trials riêng cho architecture chậm; nếu compute buộc thay đổi, điều chỉnh common protocol trước official search. Slurm resume không cấp thêm training budget.

## F. Pilot completed — official numeric contract còn NEEDS REVIEW

**UIT pilot completed:** target conformance PASS, strict full-epoch P0 PASS và bounded P1 COMPLETE (3/3 epochs) cho cả E1/E1-HR/E2. [Pilot results](../reports/pilot/pilot_results.md) lưu actual numeric pilot records, identity, curves và evidence links. User đã review đủ để đi tiếp, không cần P1-v2; official **max_epochs=8** frozen sau P1. Official recipe **chưa fully frozen**, không coi diagnostic scores là official results hoặc tự promote pilot LR/seed.

| Nhóm | Trạng thái sau pilot / phần còn review |
|---|---|
| Resolution | Resolved: E1-HR 1344×576 accepted theo strict P0, không compute-match E2 |
| Batch | Pilot common batch đã freeze và khả thi; official batch/accumulation adoption/contract còn cần review |
| LR | **NEEDS REVIEW: exact four LR values**; không tự dùng diagnostic LR 1e-4 |
| Training seeds | Exact numeric ID của 1 tuning seed và 3 final training seeds; cần freeze trước official LR search/final runs. Split seed vẫn là 406, không thay đổi membership |
| Horizon/selection | Max epochs=8 đã review/freeze; official warmup, patience, tolerance/tie-break còn NEEDS REVIEW |
| Optimizer/scheduler | Pilot exact values đã ghi; official AdamW adoption, cosine schedule/floor/horizon cho 8 epochs còn NEEDS REVIEW |
| Numerics | Pilot runtime policy đã đo; official AMP/clipping/runtime adoption còn cần review |
| Augmentation | Resolved: brightness/contrast 0.10/0.10 accepted từ train-only preview; P1 đã dùng |
| Head | Pilot deterministic init đã ghi; official adoption/seed binding và E3 head protocol còn pending |
| Environment | Actual UIT versions/runtime đã ghi; official environment/RNG/resume contract cần xác nhận |
| Pretrained integrity | Resolved: full SHA-256 đã pin và ghi trong cả sáu runs; xem report |

Pilot đã đo cả ba profiles trên UIT H200 MIG 2g.35gb, common pilot effective batch theo ảnh gốc; full environment/measurement details nằm trong report. Numeric pilot values không tự trở thành official configuration.

Không suy ra “đã hội tụ” từ throughput hoặc bounded 3-epoch curves. [Pilot plan](pilot_plan.md) nay ghi completed gates và các official-only fields còn mở. Chưa tạo `training_recipe_v1`; master sẽ reference recipe sau review thay vì duplicate toàn bộ numeric contract.

## G. Frozen preprocessing và fixed split

Bốn exact artifacts tại [artifacts/baseline124](../artifacts/baseline124/README.md): manifest.csv, summary.json, labels.json, class_support.csv, kèm source commit/provenance và hashes. **Không regenerate split từ seed 406** và không reorder label mapping.

| Split | Unique images | Supported labels |
|---|---:|---:|
| Train | 34.997 | 124 |
| Validation — tên val trong manifest | 7.497 | 106 |
| Test | 7.502 | 111 |
| Tổng | 49.996 | 124 outputs |

Quy trình chung: decoded raw RGB → UI preprocessing → view construction → resize/letterbox → ImageNet normalization.

- Footer crop **14px**; score/controls/minimap+Guess masks theo hai raw layouts.
- **Navigation arrows intentionally retained** để tránh destructive masking; không mô tả là loại sạch mọi UI.
- UI/letterbox fill bằng **exact ImageNet mean trong RGB float32 [0,1]**, không làm tròn qua uint8.
- TorchVision **tensor bilinear resize, antialias=True**, explicit rounding/centered letterbox; System.Drawing chỉ là visualization renderer.
- E1/E1-HR/E2 dùng cùng UI/fill/resize/normalization policy; UI xử lý trước khi tạo global/local views.
- E2 geometry dùng integer rectangles đã freeze; không random hay content-adaptive crop.

Nguồn chuẩn là [preprocessing v1](preprocessing_v1.md), [config](../configs/preprocessing/geoguessr_v1.json) và [checksum](../configs/preprocessing/geoguessr_v1.sha256); không duplicate các UI mask coordinates vào master spec.

[Train-only UI QA](../reports/preprocessing_qa_v1/qa_review_vi.md) đã hoàn thành với 297 ảnh thuộc 124 labels, không dùng validation/test hoặc classification score và không tune geometry. **TorchVision preprocessing conformance trên target UIT đã PASS**; [target report](../reports/pilot/evidence/uit_2026-10-31/conformance/conformance_report.json) và [pilot results](../reports/pilot/pilot_results.md) ghi audit/train-only checks riêng, không đánh đồng với UI QA.

## H. Metrics và analysis

### Checkpoint và model comparison

Main metrics: **Top-1, Top-3, Top-5, Macro-F1 supported, Weighted-F1**, report cùng nhau. Không single winner metric giữa phương pháp.

Chỉ checkpoint selection trong run dùng validation Macro-F1 supported. Lưu full validation history/learning curves và best/last checkpoints; patience/tolerance/tie-break phải được freeze trước official search. Test khóa đến final evaluation.

Per-class output gồm **precision, recall, F1, ground-truth support và predicted count**. Ghi rõ số output classes, số evaluated classes và danh sách unsupported classes. Unsupported classes **không bị ép F1=0 rồi đưa vào macro average** chỉ vì không xuất hiện trong GT split. Top-k luôn xếp hạng trên đủ 124 outputs.

### Frequency buckets

Dựa trên **unique n_train** sau dedup/fixed split: **1–19, 20–99, 100–499, ≥500**. Phân tích riêng nhóm **1–2 train samples** bên trong 1–19. Membership cố định qua mọi experiment; resampling không làm đổi bucket.

Mỗi bucket report:

| Coverage | Classification metrics |
|---|---|
| Total labels; supported evaluation labels; evaluation samples | Macro-F1; Top-1; Top-3; Top-5 |

Bucket Macro-F1 lấy trung bình per-class F1 đã tính trên **toàn evaluation set**, giữ false positives từ các bucket khác; không lọc ground-truth rows trước khi tính F1.

Error analysis gồm confusion matrix, most-confused country pairs, per-class CSV và predictions CSV. Final comparison có mean ± std theo seeds; gain nhỏ ngang độ dao động không được diễn giải mạnh. FLOPs, parameter count, latency, throughput và peak VRAM chỉ là secondary efficiency information nếu đo.

### Run/caching records

Lưu complete config/seeds, source commit, pretrained identifier/full hash, label mapping, dataset/preprocessing fingerprints, environment/GPU/package versions, history và checkpoint state. Resume tại epoch boundary cần optimizer/scheduler/AMP state, epoch/update counters và RNG/sampling/augmentation state; không reset warmup hoặc overwrite run cũ.

E3 cache gắn với source checkpoint/seed, feature definition, split/label/preprocessing hashes và dtype/environment. Đây là yêu cầu reproducibility cho implementation sau, chưa có run/cache được tạo.

## I. Main execution workflow

**Master vẫn draft chờ user review; official implementation chưa bắt đầu.** UIT pilot trong workflow dưới đây đã hoàn tất và không cần P1-v2; numeric recipe/official spec là bước tiếp theo. Giữ sơ đồ làm trình tự tham chiếu, không authorize chạy thêm stage hoặc imply các official stages đã hoàn thành:

~~~text
Frozen dataset/split
        |
        v
Frozen preprocessing v1
        |
        v
Frozen E2 geometry
        |
        v
Review + freeze master spec
        |
        v
UIT pilot
        |
        v
Freeze numeric training recipe
        |
        v
E1 specification -> E1 LR tuning -> E1 final 3 seeds -> Review E1
        |
        v
E1-HR specification -> E1-HR tuning + final seeds -> Review E1-HR
        |
        v
E2 specification -> E2 tuning + final seeds -> Review E2
        |
        v
E3 specification
        |
        v
Feature caching
        |
        v
Natural/cRT head training
        |
        v
Final evaluation
        |
        v
Comparison + report
~~~

**Experiment-specific specification phải được review trước implementation tương ứng. Không implement nhiều experiment cùng lúc.** Các stage smoke/implementation nằm trong stage đã được review, không được bỏ qua vì diagram rút gọn. Test không được mở cho pilot/tuning.

# Project checklist

Các ô [x] xác nhận deliverable/định nghĩa hoặc completed pilot gate đã có evidence, không xác nhận official model đã implement hoặc master đã frozen. Research-design items có bằng chứng trong [Grillme](grillme.md); completed pilot items có [report/evidence](../reports/pilot/pilot_results.md). Các official stages và numeric decisions chưa review giữ [ ].

### Dataset & preprocessing

- [x] Fixed 124-class split identified
- [x] Exact split artifacts pinned in new repo
- [x] Split provenance/checksums recorded
- [x] Raw image layouts analyzed
- [x] UI preprocessing v1 frozen
- [x] Navigation-arrow policy frozen
- [x] E2 crop geometry frozen
- [x] Train-only UI QA completed
- [x] TorchVision numerical preprocessing conformance verified on target environment

### Research design

- [x] Main research objective defined
- [x] E1 architecture defined
- [x] E1-HR role defined
- [x] E2 architecture defined
- [x] E3 comparison design defined
- [x] Metrics protocol defined
- [x] Checkpoint-selection metric defined
- [x] LR tuning/seeds protocol conceptually defined
- [ ] Master specification reviewed and frozen by user

### Pilot

- [x] Actual UIT pilot environment/package/runtime record captured (official adoption còn review)
- [x] Exact pretrained weight SHA-256 recorded
- [x] E1 strict full-epoch timing/VRAM P0 PASS
- [x] E1-HR strict full-epoch timing/VRAM P0 PASS
- [x] E2 strict full-epoch timing/VRAM P0 PASS
- [x] E1 bounded P1 COMPLETE, 3/3 epochs (diagnostic only)
- [x] E1-HR bounded P1 COMPLETE, 3/3 epochs (diagnostic only)
- [x] E2 bounded P1 COMPLETE, 3/3 epochs (diagnostic only)
- [x] Final E1-HR resolution accepted by strict P0
- [x] Common effective pilot batch frozen (official adoption còn review)
- [x] Brightness/contrast amplitudes accepted via train-only review
- [x] Official max epochs=8 reviewed/frozen after pilot
- [ ] Official warmup/patience/tolerance/scheduler numeric contract frozen
- [ ] LR candidate grid frozen
- [ ] Exact official tuning/final training seed IDs frozen
- [ ] Remaining numeric training recipe frozen

### E1

- [ ] E1 experiment specification
- [ ] E1 implementation
- [ ] E1 smoke test
- [ ] E1 LR tuning
- [ ] E1 final seed 1
- [ ] E1 final seed 2
- [ ] E1 final seed 3
- [ ] E1 analysis completed

### E1-HR

- [ ] E1-HR experiment specification
- [ ] E1-HR implementation
- [ ] E1-HR smoke test
- [ ] E1-HR LR tuning
- [ ] E1-HR final seed 1
- [ ] E1-HR final seed 2
- [ ] E1-HR final seed 3
- [ ] E1-HR analysis completed

### E2

- [ ] E2 experiment specification
- [ ] E2 implementation
- [ ] E2 smoke test
- [ ] E2 LR tuning
- [ ] E2 final seed 1
- [ ] E2 final seed 2
- [ ] E2 final seed 3
- [ ] E2 analysis completed

### E3

- [ ] E3 experiment specification
- [ ] Deterministic E1 feature cache
- [ ] Deterministic E2 feature cache
- [ ] E3 head protocol pilot/freeze
- [ ] E1 natural head retraining
- [ ] E1 cRT
- [ ] E2 natural head retraining
- [ ] E2 cRT
- [ ] E3 analysis completed

### Final evaluation

- [ ] Final test evaluation unlocked
- [ ] Overall metrics generated
- [ ] Per-class metrics generated
- [ ] Frequency-bucket analysis generated
- [ ] Confusion analysis generated
- [ ] Mean ± std tables generated
- [ ] Efficiency information generated if measured
- [ ] Final experiment comparison/report completed

## Out of main scope unless explicitly reopened

- LLRD
- Separate encoders
- Random local crops
- Alternative fusion
- Auxiliary local CE
- Strong augmentation
- Alternative balancing strength
- E1-HR+cRT
- Geometry search
- Unseen-location/low-similarity robustness
- Pretrained-model comparison

Không tự tạo task, spec hoặc experiment cho các mục này.
