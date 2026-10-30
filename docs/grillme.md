# Grillme — biên bản review thiết kế CS406 GeoGuessr

Cập nhật đến ngày 2026-10-30. Tài liệu tổng hợp những câu hỏi, phản biện và câu trả lời giữa người dùng và assistant; không phải bản chép nguyên văn hội thoại.

**Đây là biên bản grill, không phải frozen master specification.** Các lựa chọn về model/training bên dưới ghi nhận điều đã thống nhất trong thảo luận, còn chờ người dùng review tổng thể. Chúng không cấp phép tạo training code, chạy pilot hoặc implement E1/E1-HR/E2/E3. [Master spec](master_spec.md) vẫn là draft; [pilot plan](pilot_plan.md) vẫn deferred.

Phần được freeze riêng hiện tại là preprocessing/geometry v1 và fixed-split reproducibility, có [config chuẩn](../configs/preprocessing/geoguessr_v1.json), [checksum](../configs/preprocessing/geoguessr_v1.sha256) và [split provenance](../artifacts/baseline124/provenance.json). Biên bản này không thay thế các nguồn đó.

## 1. Mục tiêu và cách đánh giá phương pháp

### Câu 1 — Benchmark hiện tại hay tổng quát sang địa điểm mới?

- **Vì sao hỏi:** exact deduplication không ngăn ảnh cùng panorama/địa điểm xuất hiện ở nhiều split; score trên split hiện tại chưa chứng minh location-disjoint generalization.
- **Người dùng chọn:** mục tiêu chính là tối đa classification performance trên fixed benchmark; giữ nguyên split cho E1/E1-HR/E2/E3.
- **Giới hạn:** same-place/same-panorama overlap được ghi nhận là limitation. Secondary robustness chỉ xem xét sau nếu phù hợp scope; không tự thay primary split.

### Câu 2 — Macro-F1 tăng nhưng Top-1 giảm thì chọn mô hình nào?

- **Vì sao hỏi:** “performance tốt hơn” có thể mang nghĩa khác nhau giữa overall accuracy và long-tail recognition.
- **Người dùng điều chỉnh câu hỏi:** không chọn A/B để ép một single best model hoặc đặt ngưỡng Top-1 giảm tối đa ngay lúc này.
- **Main reporting metrics đã thống nhất:** Top-1, Top-3, Top-5, Macro-F1 trên các lớp có ground-truth support và Weighted-F1; report cùng nhau để phân tích Pareto/trade-off, không thiết lập một single winner metric giữa các phương pháp.
- **Checkpoint selection:** vẫn theo validation Macro-F1 như Câu 3. Phải phân biệt chọn checkpoint trong một run với so sánh phương pháp giữa các run.

### Câu 3 — Metric chọn checkpoint là gì?

- **Vì sao hỏi:** Top-1 dễ bị head classes chi phối; Macro-F1 có thể nhiễu khi validation chứa các lớp rất ít ảnh.
- **Đã thống nhất:** chọn `best.pt` theo **validation Macro-F1 trên các lớp có ground-truth support**. Mọi metric khác của run lấy từ chính checkpoint đó; test không tham gia selection.
- **Safeguards được yêu cầu:** patience, lưu full history/learning curves, quy tắc tolerance/tie-break xác định trước. Giá trị tolerance và lựa chọn cụ thể giữa ưu tiên epoch sớm hơn hoặc NLL tốt hơn chưa được chốt bằng số.

## 2. E1-HR, global/local và UI preprocessing

### Câu 4 — Có cần E1-HR làm control cho E2?

- **Vì sao hỏi:** E2 thấy nhiều pixel/view hơn, nên E2 vượt E1 chưa đủ để quy toàn bộ gain cho cách tổ chức global/local.
- **Đã thống nhất:** thêm E1-HR, một full-image higher-resolution control; đây không phải hướng nghiên cứu riêng.
- **Điều chỉnh về sau:** yêu cầu compute-match ban đầu đã được người dùng bỏ ở Câu 23. Không dùng kết luận cũ về compute matching làm ràng buộc hiện tại.

### Câu 5 — Shared encoder hay separate encoders?

- **Vì sao hỏi:** encoder riêng tăng capacity và làm khó tách contribution của các view khỏi số tham số backbone.
- **Người dùng chọn:** một **shared ConvNeXt-Tiny encoder** cho global và tất cả local views.
- **Ngoài main scope:** separate encoders chỉ là optional/future ablation, không tự động triển khai.

### Câu 6 — Deterministic hay random local crops?

- **Vì sao hỏi:** random crop khi train và fixed crop lúc inference tạo thêm augmentation variable và train/inference mismatch. Output size chưa xác định field of view của vùng crop raw.
- **Người dùng chọn:** deterministic fixed crops có overlap, cùng rule ở train/val/test, lấy từ raw/high-resolution image.
- **Cách chốt geometry:** preview train images của cả hai layout trước; không khóa vùng crop chỉ từ phép tính hoặc input size của local.

### Câu 7 — Loại UI bằng crop hay mask?

- **Vì sao hỏi:** crop bỏ cả scene thật; mask giữ khung nhưng có thể tạo vùng trống và lãng phí local view.
- **Người dùng chọn:** **hybrid UI removal**, ưu tiên scene content. Crop mỏng ở mép nếu ít mất cảnh; mask UI bên trong hoặc nơi crop gây mất cảnh nhiều.
- **Control:** xử lý UI trước khi tạo views; cùng policy cho E1/E1-HR/E2. Geometry thực tế phải qua train-only preview.

### Câu 8 — Fusion global/local như thế nào?

- **Vì sao hỏi:** average cả năm view trộn vai trò global/local; concatenate từng crop giữ vị trí nhưng tăng head size và phụ thuộc geometry.
- **Người dùng chọn:** average bốn local features, concatenate với global feature, rồi linear classifier.
- **Chấp nhận:** E2 v1 không giữ positional identity của từng local crop. Alternative fusion là optional/future work.

## 3. Long-tail controls và fairness

### Câu 9 — cRT chỉ áp dụng E2 hay cả E1?

- **Vì sao hỏi:** chỉ có E2+cRT chưa trả lời E1+cRT có đạt trade-off tương đương với representation đơn giản hơn không.
- **Đã thống nhất:** ma trận E1, E1+cRT, E2, E2+cRT; hai nhánh cRT kế thừa representation checkpoint tương ứng, được chọn bằng validation Macro-F1.
- **Giới hạn scope:** chưa thêm E1-HR+cRT; chỉ xem xét lại nếu có lý do rõ ràng sau này.

### Câu 10 — Một hay nhiều training seeds?

- **Vì sao hỏi:** một run không tách được gain của phương pháp khỏi dao động khởi tạo, batch order và augmentation.
- **Người dùng chọn:** exploration/smoke dùng một training seed; final configuration của E1/E1-HR/E2 dùng ba training seeds. E3 kế thừa từng source seed tương ứng.
- **Báo cáo:** mean ± standard deviation, không chọn best test seed. Split seed 406 giữ nguyên; training seeds không chia lại dữ liệu. Gain nhỏ tương đương độ dao động không được diễn giải mạnh.

### Câu 11 — LR chung hay tuning riêng từng architecture?

- **Vì sao hỏi:** nhiều view qua shared encoder có thể tạo optimization dynamics khác E1, dù backbone giống nhau.
- **Người dùng chọn:** tuning LR riêng, cùng LR candidate set, số trial, tuning seed, optimizer/scheduler protocol, epoch budget và checkpoint rule.
- **Control:** effective batch size tính theo số ảnh gốc; năm view của E2 không phải năm mẫu độc lập. Công bố tất cả LR đã thử và LR được chọn.

### Câu 12 — LLRD thuộc main recipe không?

- **Vì sao hỏi:** LLRD thêm decay factor và cách chia tầng; cần phân biệt pretrained fine-tuning recipe với contribution của representation.
- **Người dùng chọn:** không dùng LLRD trong main recipe ban đầu; toàn backbone dùng một LR trong mỗi run.
- **Giới hạn kết luận:** không dùng không đồng nghĩa LLRD không hữu ích. Không tự mở LLRD ablation.

### Câu 13 — Full fine-tuning hay classifier-only trước?

- **Vì sao hỏi:** freeze/unfreeze thêm phase transition và hyperparameters; LR warmup khác với đóng băng backbone.
- **Người dùng chọn:** full fine-tuning từ epoch 1 với LR warmup; không classifier-only warmup trong main recipe.
- **Ngoài scope:** head-first warmup chỉ là phương án tương lai nếu có bằng chứng cần thiết.

### Câu 14 — Augmentation chính đến mức nào?

- **Vì sao hỏi:** augmentation có thể đổi driving-side cues, màu sắc địa lý hoặc regularization, gây lẫn với thay đổi representation.
- **Đã thống nhất:** train chỉ brightness/contrast jitter nhẹ; một bộ parameters mỗi ảnh gốc, áp dụng nhất quán cho global và tất cả local views; không làm đổi mask/padding.
- **Val/test:** deterministic, không augmentation. Chưa dùng flip, random crop, RandAugment, erasing, Mixup/CutMix hoặc label smoothing trong main recipe.
- **Lưu ý:** ảnh Mixup/CutMix không thực tế về địa lý chưa đủ để chứng minh kỹ thuật đó không hữu ích; việc này chưa được kiểm chứng và không nằm trong main scope. Biên độ jitter cụ thể còn pending.

### Câu 15 — Có cần natural-sampling head-retraining control?

- **Vì sao hỏi:** reset head, train thêm và thay augmentation có thể tạo gain dù chưa balancing.
- **Người dùng đồng ý:** thêm natural head retraining cho cả E1 và E2, làm control giải thích gain của cRT.
- **Protocol:** representation/cache, initial head state theo seed, optimizer, LR, step/epoch budget và selection rule giống nhánh cRT; chỉ sampling khác. Đây là control thuộc E3, không phải hướng nghiên cứu riêng.

### Câu 16 — Cân bằng hoàn toàn cả lớp chỉ có 1–2 ảnh train?

- **Vì sao hỏi:** full balancing lặp extreme-tail rất nhiều; lặp feature không tạo thêm thông tin, có nguy cơ memorization.
- **Người dùng chọn:** class-uniform cho cRT chính: chọn lớp với xác suất `1/124`, rồi chọn đều ảnh trong lớp. Không thêm sampling-strength hyperparameter.
- **Diễn giải:** extreme-tail cần thận trọng. `sqrt(n_train)` chỉ là optional/future balancing strategy, không tự thêm vào main matrix.

### Câu 17 — Frequency buckets đếm bằng raw count hay train count?

- **Vì sao hỏi:** số ảnh mô hình được học khác tổng số ảnh raw; resampling cũng không làm tăng số ảnh độc lập.
- **Đã thống nhất:** dùng unique `n_train` sau dedup và fixed split; buckets 1–19, 20–99, 100–499, ≥500. Phân tích riêng nhóm 1–2 trong bucket 1–19.
- **Control:** membership freeze qua mọi model. Mỗi bucket báo tổng labels, supported labels, evaluation samples, Macro-F1 và Top-1/3/5.
- **Metric detail:** bucket Macro-F1 lấy từ per-class F1 tính trên toàn evaluation set; không lọc ground-truth rows trước rồi làm mất false positives từ bucket khác.

### Câu 18 — Cache deterministic feature hay giữ augmentation khi retrain head?

- **Vì sao hỏi:** feature cache giúp giảm compute nhưng thay đổi augmentation variability so với full fine-tuning.
- **Người dùng chọn:** freeze representation, evaluation mode, tắt stochastic depth; mỗi train/val image đúng một deterministic feature, không brightness/contrast jitter ở E3.
- **Control:** E1 cache trước classifier, E2 cache fused feature trước classifier; natural/cRT dùng chính xác cùng cache. Không mở cache nhiều jittered versions trong main scope.

### Câu 19 — E1-HR/E2 khởi tạo pretrained hay warm-start từ E1?

- **Vì sao hỏi:** warm-start từ E1 được hưởng thêm GeoGuessr training time, làm lẫn architecture gain với cumulative optimization.
- **Người dùng chọn:** E1/E1-HR/E2 khởi tạo độc lập từ cùng chính xác pretrained checkpoint; không kế thừa checkpoint, optimizer hoặc scheduler của E1.
- **Phân biệt:** E1 → E2 → E3 là tiến trình thí nghiệm, không phải checkpoint lineage. Chỉ head-retraining/cRT kế thừa representation checkpoint nguồn.

## 4. Test history và giới hạn scope

### Câu 20 — Test cũ đã từng ảnh hưởng thiết kế chưa?

- **Vì sao hỏi:** khóa test từ hiện tại không xóa được lịch sử đã xem test trước đó.
- **Người dùng ghi nhận bảo thủ:** gần B nhất — test đã từng evaluate/inspect trong pipeline cũ, nhưng hiện không có bằng chứng score/lỗi test được dùng để tune architecture, preprocessing, augmentation hoặc hyperparameters.
- **Protocol mới:** train/validation cho mọi quyết định, test chỉ final evaluation. Nếu phát hiện bằng chứng cũ dựa vào test, reclassify thành C; không khẳng định test chưa từng được xem.

### Câu 21 — Có làm low-visual-similarity robustness subset ngay không?

- **Vì sao hỏi:** similarity ảnh chỉ là proxy; khác góc nhìn cùng nơi có thể khác ảnh, hai nơi khác nhau có thể giống ảnh.
- **Người dùng defer:** không mở secondary robustness analysis ở giai đoạn này; chuyển quyết định sang evaluation-spec phase.
- **Naming:** nếu tương lai không có panorama ID/tọa độ đáng tin cậy, chỉ gọi “low visual similarity to train”, không gọi “unseen-location evaluation”.

### Câu 22 — Head LR multiplier có lớn hơn 1 không?

- **Vì sao hỏi:** head mới và backbone pretrained có nhu cầu cập nhật khác nhau, nhưng multiplier thêm một lựa chọn chưa kiểm chứng.
- **Người dùng chọn:** `m=1`, tức `LR_head=LR_backbone`; chỉ tuning một base LR mỗi architecture.
- **Kiến trúc:** average/concatenate không có tham số học riêng; không mặc định tồn tại một fusion module cần LR group khác.

### Câu 23 — E1-HR có phải match FLOPs hoặc latency E2?

- **Vì sao hỏi ban đầu:** một resolution không đồng thời match FLOPs, latency và VRAM; cần xác định ý nghĩa đối chứng.
- **Người dùng sửa hướng:** **không dùng FLOPs hoặc latency làm ràng buộc matching**. E1-HR kiểm tra higher full-image resolution; E2 kiểm tra cấu hình global/local trên benchmark.
- **Hiện tại:** E1-HR chọn theo ảnh gốc, aspect ratio, khả năng UIT GPU; không chọn bằng test performance. Candidate 1344×576 chưa phải resolution frozen.
- **Báo cáo:** metrics classification là chính; FLOPs, parameters, latency, throughput, VRAM là secondary efficiency information. Không tuyên bố đã cô lập compute bằng một comparison không compute-matched.

### Câu 24 — Chỉ fused CE hay auxiliary CE từng local view?

- **Vì sao hỏi:** một crop trời/cây/mặt đường không nhất thiết đủ thông tin để tự phân loại quốc gia; auxiliary loss thêm mục tiêu và loss weight.
- **Người dùng chọn:** một standard CE trên prediction sau fusion. Local views được phép bổ sung thông tin không đầy đủ cho global; gradient vẫn truyền qua mọi view.
- **Class-weight policy cho E1/E1-HR/E2:** standard CE trên natural training distribution, không class weighting hoặc `sqrt_inverse` weighting. Long-tail treatment được chủ động cô lập sang E3; không mang class-weight recipe từ repo cũ sang ConvNeXt experiments.
- **Ngoài scope:** per-view auxiliary CE không thuộc main recipe.

### Câu 25 — Có stochastic depth trong main recipe?

- **Vì sao hỏi:** augmentation policy chưa xác định regularization bên trong backbone; không nên phụ thuộc library default.
- **Fixed design decision đã chốt:** `max_drop_path_rate = 0.1`, dùng cùng explicit depth schedule cho E1/E1-HR/E2, không tuning riêng từng architecture.
- **E3:** evaluation mode phải tắt stochastic depth khi extract deterministic features.

### Câu 26 — Fixed weight decay hay tuning thêm một chiều?

- **Vì sao hỏi:** mở LR × weight-decay grid làm tăng search budget và cần control công bằng.
- **Người dùng chọn:** fixed `weight_decay=0.05` cho E1/E1-HR/E2; chỉ tuning LR trong main recipe.
- **Parameter groups đã nêu:** decay trainable Conv/Linear weights; không decay bias hoặc LayerNorm parameters. Không dựa vào defaults. Mở weight-decay tuning sau này cần yêu cầu riêng, không tự động thành ablation.

### Câu 27 — Geometry chọn bằng preview hay validation score?

- **Vì sao hỏi:** E2 thử nhiều geometry rồi chọn theo validation sẽ nhận thêm architecture-search budget dù số LR trials bằng baseline.
- **Người dùng chọn:** chọn từ preview **train only**, dựa coverage, zoom, overlap, sky/non-informative regions, UI và semantic context; freeze trước LR search.
- **Giới hạn:** không dùng validation score để chọn geometry E2 v1. Alternative geometry không tự tạo main experiment; nếu tương lai nghiên cứu riêng thì ghi rõ additional tuning.

### Câu 28 — Có hard compute budget/deadline không?

- **Vì sao hỏi:** thiếu thời gian cuối quá trình có thể dẫn đến cắt epochs/trials riêng cho architecture chạy chậm, phá fairness.
- **Người dùng trả lời:** chưa có hard total GPU-hours budget hoặc deadline. Slurm có per-job walltime; có thể tổ chức run/resume.
- **Hướng đã thảo luận:** benchmark cả E1/E1-HR/E2 rồi mới freeze common max epochs, warmup, patience và rule trước official LR search. Không tự giảm riêng architecture chậm. **Pilot hiện vẫn deferred theo scope correction mới nhất.**

### Câu 29 — Head retraining có mở LR search riêng đầy đủ không?

- **Vì sao hỏi:** linear head trên frozen features có optimization khác full fine-tuning; kế thừa LR máy móc có thể làm head chưa hội tụ.
- **Điều chỉnh đã chốt:** không mặc định mở một full LR search riêng cho E3. Chỉ một small head pilot để xác định protocol đủ hội tụ, rồi freeze LR, optimizer, budget, initialization và selection rule.
- **Fairness:** natural/cRT cùng representation dùng exactly the same protocol; khác biệt duy nhất là sampling.
- **Ngân sách main được làm rõ:** E1/E1-HR/E2 mỗi architecture **4 LR × 1 training seed**, chọn LR bằng validation Macro-F1, rồi final configuration × **3 training seeds**. Không 4 LR × 3 seeds. Đây là kế hoạch thảo luận, chưa phải lệnh chạy.

## 5. Exact pretrained checkpoint, split và geometry cuối

### Câu 30 — Pretrained checkpoint cụ thể nào?

- **Vì sao hỏi:** “ImageNet pretrained” chưa đủ để tái lập starting point và preprocessing.
- **Người dùng chọn:** `torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, file `convnext_tiny-983f1562.pth`, chung cho E1/E1-HR/E2 khởi tạo độc lập.
- **Metadata yêu cầu:** PyTorch/TorchVision versions và full SHA-256 của weight file. Chưa khẳng định đã tải/check hash file pretrained tại local.
- **Preprocessing:** chỉ giữ ImageNet mean/std, không dùng nguyên `weights.transforms()` vì có center-crop 224×224. Explicit `stochastic_depth_prob=0.1`. Không mở pretrained-checkpoint comparison.

### Câu 31 — Fixed split manifest đang ở đâu?

- **Vì sao hỏi:** chỉ seed 406 và tổng số mẫu không tái tạo được exact membership; chọn ảnh preview phải chắc chắn thuộc train.
- **Người dùng cung cấp:** repo `HiimRaccoon/cs406-geoguessr`, `artifacts/baseline124/manifest.csv` và `summary.json`; không regenerate split.
- **Đã kiểm tra:** 49.996 unique rows; train 34.997, val 7.497, test 7.502; coverage 124/106/111 labels.
- **Pin hiện tại:** commit nguồn `33cb29d72974f1677efab0f56ed92360e5a9f712`; bốn file manifest/summary/labels/class_support đã được copy nguyên byte vào [artifacts/baseline124](../artifacts/baseline124/README.md), kèm provenance và bảo vệ newline conversion.

### Câu 32 — Có giữ navigation arrows như ngoại lệ không?

- **Vì sao hỏi:** arrows đổi vị trí/hướng/khoảng cách; một fixed central mask đủ rộng sẽ che cả road markings/lane structure. Detector thêm subproblem và rủi ro false positives.
- **Người dùng đồng ý:** intentionally retain arrows; crop/mask static hoặc predictable UI theo layout, dùng chung mọi architecture.
- **Methodology:** preprocessing không loại sạch mọi UI. Không mở arrow-detection experiment trong main scope.

### Câu 33 — Ưu tiên zoom local hay coverage rộng?

- **Vì sao hỏi:** global đã là 896×384; local output 320×320 không tự bảo đảm giữ thêm detail nếu crop raw quá rộng.
- **Người dùng chọn:** ưu tiên local detail với mức zoom rõ rệt, chấp nhận local không phủ kín scene vì global đã giữ context.
- **Cách làm:** preview candidate cạnh 384–448 ở layout lớn; layout nhỏ phải giữ effective zoom tương ứng, không copy cứng số pixel hoặc scale chỉ theo width raw. Chưa chốt bằng tính toán đơn thuần và không dùng validation score.

### Câu 34 — Chốt geometry nào sau visual preview?

- **Đã preview:** 65 train images, gồm 24 layout lớn và toàn bộ 41 train images layout nhỏ; so sánh kích thước bằng hình ảnh, không train model để chọn.
- **Người dùng chốt:** global **896×384**, local **320×320**; crop raw **384×384** cho layout 1536×662, **402×402** cho layout 1030×703.
- **Centers:** `(0.20W,0.60H)`, `(0.40W,0.60H)`, `(0.60W,0.60H)`, `(0.80W,0.50H)`, tính trên UI-processed raw frame. Crop trực tiếp frame đó, không từ resized global.
- **Đánh đổi đã báo cáo:** scale local/global khoảng 1,43×; overlap diện tích crop kế cận khoảng 17–20% ở layout lớn, 41–49% ở layout nhỏ. Một số crop vẫn chứa nhiều vegetation hoặc wall; không bảo đảm mọi view đều có manh mối mạnh.
- **Scope:** 416/448 chỉ giữ như visual design alternatives; không thử bằng validation score và không tự biến thành main experiments.

## 6. Những điều chỉnh quan trọng xuyên suốt cuộc grill

### Resolution proposal đã thay đổi theo dữ liệu thực tế

Raw data có 49.997 ảnh: 49.942 ảnh 1536×662, 55 ảnh 1030×703. Tỷ lệ phổ biến khoảng 2,32:1 khiến canvas 2:1 có padding đáng kể.

Các proposal cũ `512×256`, `896×448` và local `256×256` đã bị thay thế, không phải fixed design hiện tại. E1/E2 global là 896×384; E2 local 320×320. E1-HR chưa khóa size. Không suy ra mọi ảnh có cùng aspect ratio chỉ vì tất cả là landscape.

### Scope correction: không biến mọi phản biện thành experiment

Người dùng yêu cầu giới hạn main scope ở E1/E1-HR/E2/E3. Natural head retraining là control cần cho validity của E3. Các recipe choices về augmentation, fusion, optimizer, stochastic depth và weight decay không bắt buộc tạo ablation tương ứng.

Chỉ ghi optional/future work cho LLRD, alternative fusion, random crops, separate encoders, auxiliary losses, strong augmentation, intermediate balancing và robustness. Không triển khai nếu chưa có yêu cầu riêng hoặc nhu cầu validity rõ ràng được thảo luận.

### Compute matching đã bị hủy, equal tuning protocol vẫn giữ

E1-HR không cần khớp FLOPs/latency E2. Điều này không hủy các controls về fixed split, pretrained starting point, LR candidate budget, effective batch size, selection rule hoặc seeds. Benchmark score là primary comparison; efficiency được báo cáo bổ sung nếu thực hiện.

## 7. Preprocessing/split review đã hoàn thành sau Câu 34

Người dùng yêu cầu hoàn tất đúng preprocessing/geometry/split reproducibility, không mở thêm experiment branch. Các phần sau đã được đưa vào versioned preprocessing v1:

- Exact four split artifacts và source commit/provenance; không regenerate từ seed.
- Footer crop **14px**, score/controls/minimap+Guess masks theo từng layout, navigation-arrow exception.
- Fill UI/letterbox bằng exact ImageNet mean trên **RGB float32 [0,1]**; không dùng màu uint8 làm tròn của PNG preview để định nghĩa training input.
- Resize bằng **TorchVision tensor bilinear, `antialias=True`**, explicit rounding và centered letterbox; System.Drawing chỉ phục vụ visualization.
- Geometry 384/402 và các integer crop rectangles đã freeze; legacy recommendation file trỏ về versioned config.
- UI QA mở rộng trên **297 ảnh train thuộc 124 labels**: 256 layout lớn và toàn bộ 41 layout nhỏ. Hash, decoded dimensions và 13 UI contact sheets đã được kiểm tra; không cần đổi template và không tune geometry.

Nguồn chi tiết:

- [Preprocessing v1](preprocessing_v1.md).
- [Config v1](../configs/preprocessing/geoguessr_v1.json) và [checksum](../configs/preprocessing/geoguessr_v1.sha256).
- [UI QA review](../reports/preprocessing_qa_v1/qa_review_vi.md) và [freeze verification](../reports/preprocessing_qa_v1/freeze_verification.json).
- [Crop geometry visual review](../reports/crop_geometry_preview/geometry_review_vi.md).
- [Fixed split provenance](../artifacts/baseline124/provenance.json).

**Giới hạn xác nhận:** chưa chạy numerical conformance bằng TorchVision tại local vì Python launcher không khả dụng. System.Drawing UI QA không thay thế kiểm tra đúng backend trên UIT. Đây là một implementation acceptance check còn cần thực hiện khi được phép, không phải lý do mở geometry search hoặc tuyên bố đã chạy training.

## 8. Những gì chưa được chốt/freeze ở master level

Các giá trị sau còn cần dữ liệu pilot hoặc review sau, không được tự điền rồi coi là kết luận đã có:

- Final E1-HR resolution; không compute-match E2 và không chọn bằng test.
- Microbatch, gradient accumulation, common effective batch size.
- Bốn LR values, số epochs, warmup, patience, tolerance/tie-break và scheduler numeric parameters.
- Numeric training details như AMP/clipping, jitter amplitudes, initialization và package/environment lock.
- Small E3 head protocol: LR, optimizer, budget và sampling mechanics chi tiết, bảo đảm hai nhánh chỉ khác sampling.
- Secondary evaluation/robustness protocol, nếu người dùng muốn mở lại ở evaluation-spec phase.

Người dùng đã làm rõ: **master spec hiện chỉ là draft**; tạm thời không mở rộng, không tạo experiment-specific specs, không implement E1/E1-HR/E2/E3 và không tạo training code. Sau khi toàn bộ grill và preprocessing review kết thúc, người dùng sẽ review lại các quyết định rồi mới yêu cầu cập nhật/freeze master spec.

Biên bản này chỉ lưu lại cuộc grill và các thay đổi quyết định; không tự động chuyển công việc sang pilot hoặc implementation.
