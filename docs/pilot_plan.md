# UIT pilot và các training parameters còn cần freeze (DRAFT / DEFERRED)

**Trạng thái kế hoạch: DRAFT; pilot execution / P0 / P1: DEFERRED — chưa chạy.**

- **Pilot harness implementation: implemented**, self-contained trong `src/pilot/`; preprocessing reusable trong `src/preprocessing/`.
- **Pilot execution / P0 / P1: DEFERRED — chưa chạy**; chưa có GPU benchmark hoặc Slurm job.
- **Official E1/E1-HR/E2 implementation: chưa bắt đầu**; pilot code không phải official framework.
- **Master spec vẫn draft, chưa authorize official experiments**. Các gate bên dưới không tự cấp phép chạy pilot/experiment.

Task hiện tại chỉ patch harness/tests/tài liệu trước deployment; không chạy GPU/Slurm hoặc sửa frozen preprocessing. Các giá trị chưa được review vẫn pending; không tự chọn numeric values, fallback resolutions, full pretrained hash hoặc seed IDs.

## Gate 1: dataset và environment

1. Chọn runtime root `/datastore/binhvp/<my-folder>/`, giữ đường dẫn trong run/Slurm config; không sửa các relative paths của manifest.
2. Xác minh SHA-256 của bốn artifact theo `provenance.json`, original summary fingerprints, label IDs, split counts và class support. Trước pilot đầu tiên trên một bản dataset mới, kiểm tra file tồn tại và SHA-256 ảnh theo manifest; lưu audit gắn với dataset root/fingerprint, không hash lại toàn bộ mỗi epoch.
3. Ghi package lock của PyTorch/TorchVision/Pillow và phiên bản CUDA/cuDNN/driver, GPU/MIG name, CPU/RAM allocation. Mọi main experiment dùng cùng environment lock. Không tự đổi backend khi một dependency không hỗ trợ contract.
4. Tải đúng pretrained file `convnext_tiny-983f1562.pth`, tính/review full SHA-256 và **bắt buộc pin `expected_weight_sha256` trước P0/P1**; dùng cùng file cho cả ba architecture. Config check từ chối null/invalid full hash; model chỉ load khi actual full digest khớp chính xác expected digest. Conformance không cần weight. Không dùng default alias hoặc chỉ SHA prefix làm integrity gate.

## Gate 2: conformance với preprocessing v1

Đọc config và checksum từ `configs/preprocessing/`. Dùng chính thức tensor backend đã khai báo, không dùng ảnh PNG preview làm model input. Chỉ dùng train samples cố định của hai layout cho các kiểm tra này; không dùng score để sửa geometry.

- Raw dimensions, footer crop, UI rectangles, local coordinates và số view phải khớp integer tables.
- Global có tensor shape `[3,384,896]`; mỗi local `[3,320,320]`. Full-image content/padding khớp 896×378 với pad 0/3/0/3 hoặc 574×384 với pad 161/0/161/0.
- Input float32 RGB [0,1], exact mean fill trước normalization, bilinear tensor resize với `antialias=True`, không intermediate uint8, không `weights.transforms()` hoặc PIL/System.Drawing resize.
- Kiểm tra pixel interior của mask và padding sau normalize trong numerical tolerance đã ghi nhận; kiểm tra riêng edge behavior có antialias, không đòi mọi pixel sát biên đều zero.
- Một reference tile lấy từ UI-processed raw phải khớp crop/resize cùng rectangle trực tiếp; không khớp một pipeline crop từ resized global.
- Repeated deterministic evaluation preprocessing của cùng ảnh trong cùng environment phải khớp. Train jitter dùng cùng một draw cho mọi view và giữ fill/padding; val/test/E3 cache tắt jitter.
- Nếu backend/library update làm khác numerical outputs, ghi nhận và xử lý versioning, không đổi v1 âm thầm. Numerical tolerance trong cùng runtime phải chặt; không tuyên bố bitwise equality giữa GPU/CPU hoặc giữa các decoder khác nhau khi chưa kiểm tra.

Gate này kiểm chứng implementation của specification đã freeze; không phải geometry search. Local CPU TorchVision conformance và train/validation hash audit đã pass trong bước implement harness; target UIT environment/root vẫn phải chạy lại gate trước pilot. Đây không phải P0/P1 execution hoặc GPU benchmark.

## Gate 3: hai stage pilot và các pre-run rules

E1 global 896×384; E2 giữ đúng global + bốn local của preprocessing v1. Dùng một predetermined training seed chung cho pilot E1/E1-HR/E2; exact numeric ID phải được review/freeze trước khi chạy, không tự lấy split seed 406 làm training seed. **Không dùng test trong P0 hoặc P1.**

### P0 — feasibility/timing pilot

P0 chỉ xác minh:

- Preprocessing và model forward/backward hoạt động.
- Microbatch khả thi, OOM hoặc numerical failure.
- Peak allocated/reserved VRAM.
- Steady-state throughput theo original images.
- Full-epoch wall time và DataLoader/CPU bottleneck.

P0 không dùng để kết luận convergence hoặc model quality. Phân biệt GPU benchmark warmup với LR warmup. Có ít nhất một epoch đầy đủ để phản ánh I/O và tail batches, không chỉ vài forward passes; measurement protocol phải được chốt trước khi đo.

Tìm microbatch khả thi rồi chọn common effective batch size tính theo ảnh gốc. Microbatch/accumulation được phép khác nhưng số mẫu trên optimizer update và loss scaling phải nhất quán, kể cả accumulation cuối epoch. Năm view của một ảnh E2 vẫn là một training sample.

### P1 — bounded training/convergence pilot

P1 xem learning/validation curves và optimization stability để hỗ trợ freeze **common max epochs, LR warmup, early-stopping patience/tolerance và scheduler horizon**. Không suy ra các giá trị này chỉ từ throughput của P0.

**Bounded P1 budget phải được định nghĩa và review trước khi chạy:** giới hạn epochs/optimizer steps, điều kiện dừng và số diagnostic attempts được phép phải explicit, hữu hạn. Các giới hạn numeric hiện pending, không được tự điền trong task này. Không train open-ended hoặc tự tăng budget cho đến khi thấy kết quả mong muốn.

Hết budget mà chưa đủ cơ sở kết luận thì report giới hạn/bất ổn để người dùng review, không tự gia hạn hoặc lặp trial theo validation score. Không tự giảm training horizon riêng cho architecture chậm hơn. Nếu phải giảm compute, điều chỉnh common protocol trước official search.

### Deterministic E1-HR resolution selection

**1344×576 chỉ là first candidate**, không phải final resolution. Trước P0 phải review/freeze:

- Ordered candidate/fallback list, bắt đầu bằng first candidate trên; các fallback numeric values hiện chưa được chọn/review.
- Feasibility criteria về raw-image/aspect-ratio suitability, memory và runtime/walltime, kèm batch/accumulation policy được phép và các threshold cần thiết.
- Rule chọn candidate, stop/fail rule và cách record kết quả.

Xét theo đúng thứ tự đã freeze: **candidate đầu pass rule thì chọn/freeze ngay; chỉ thử fallback nếu candidate trước fail rule**. Không thử tiếp để tìm resolution có score cao hơn; không dùng validation classification score hoặc test score để chọn. Không compute-match E2.

Nếu hết approved list mà không candidate nào pass thì report lại; không tự thêm fallback. Đây là feasibility selection theo rule, không phải resolution search. Task hiện tại không cung cấp thêm fallback sizes hoặc feasibility threshold numeric.

### LR scale sanity-check và bốn official candidates

Pilot không được trở thành LR search ngoài official tuning budget. Chỉ cho phép sanity-check/bracket learning-rate scale bằng dấu hiệu optimization stability: divergence/NaN, exploding updates hoặc effectively no learning. Các sanity-check attempts và budget được định nghĩa trước, record đầy đủ cả failures.

P1 có thể ghi validation curves theo protocol, nhưng **không iterative refine LR grid dựa trên validation Macro-F1**, không chọn LR pilot vì có classification score tốt nhất và không repeated trial search trá hình. Nếu stability checks chưa đủ cơ sở thì report lại, không tự mở rộng search.

Sau pilot, freeze đúng **4 base LR values** thành một tập chung trước official LR search. Numeric values hiện pending. Official selection mới dùng validation Macro-F1 theo budget 4 LR × 1 tuning seed/architecture, rồi best LR × 3 final seeds.

## Runtime/timing reproducibility controls

Trước comparable timing, record/freeze ít nhất các settings sau trong pilot protocol; không dựa vào defaults:

| Nhóm | Settings phải ghi explicit |
|---|---|
| DataLoader | `num_workers`, `pin_memory`, `persistent_workers`, prefetch policy nếu dùng; ghi rõ disabled/not applicable nếu không dùng |
| Mixed precision | AMP dtype, GradScaler policy |
| TF32 | TF32 policy cho matmul/cuDNN theo API của version đang pin |
| cuDNN | Benchmark và deterministic policy |
| Compile | `torch.compile` on/off; cấu hình liên quan nếu bật |
| PyTorch numerics | Float32 matmul precision, deterministic-algorithm policy và relevant matmul/determinism settings của runtime |

Các runtime settings là **protocol, không phải ablation**. Dùng cùng policy cho các architecture ở nơi áp dụng được; mọi khác biệt cần thiết phải được khai báo trước. Nếu diagnostic thay đổi một setting, phải record và đo lại affected comparable timings dưới policy cuối, không trộn measurements từ các settings khác nhau.

Measurement protocol phải khai báo trước:

- GPU benchmark warmup và measured window/repetitions; numeric lengths hiện pending.
- CUDA synchronization quanh thời điểm bắt đầu/kết thúc timing để hoàn tất GPU work trong window.
- Reset peak-memory statistics sau warmup, trước measured workload; record cả **peak allocated và peak reserved VRAM** trong workload đó.
- Steady-state samples/sec theo **original images**, không theo số global/local views; ghi rõ measured window có bao gồm data loading/transfer hay không, nhất quán giữa các architecture.
- Full-epoch walltime riêng, gồm I/O, tail batches và accumulation cuối epoch; ghi rõ validation/checkpoint overhead có nằm trong timing hay được tách riêng.
- Microbatch, accumulation, effective batch, DataLoader/CPU bottleneck, OOM và numerical failures kèm môi trường cụ thể.

Không chọn numeric runtime settings trong task patch tài liệu này.

## New-classifier initialization

Trước pilot phải freeze **một explicit deterministic initialization rule cho new 124-class classifier của E1, E1-HR và E2**, không chỉ cho E3. Record rule cho weights/bias, parameters của rule, exact framework/version và seed/RNG behavior. Mỗi run khởi tạo độc lập, dùng cùng rule; các head khác feature dimension không bắt buộc có cùng shape/weights.

Có thể dùng framework/default initialization nếu đó là lựa chọn cuối được review, nhưng phải ghi rõ exact rule/version, thời điểm và cách seed/RNG được dùng; không chỉ ghi “default”. Exact rule và numeric seed IDs hiện pending, không tự chọn distribution, scale hoặc seed ở đây.

Với E3, natural/cRT của cùng source seed phải bắt đầu từ **identical head weights**, không chỉ cùng tên initialization hoặc seed rule. Rule này cũng thuộc common E3 head protocol được freeze trước official head runs.

## Những trường cần điền trước official LR search

| Trường | Trạng thái/quy tắc |
|---|---|
| E1-HR full-image WH | Pending deterministic feasibility rule; ordered list/criteria freeze trước P0, không validation-score selection |
| P1 pilot budget | Review/freeze finite epochs/steps, stop rules và diagnostic-attempt limit trước P1; không open-ended |
| Common effective batch size | Pending; tính theo ảnh gốc |
| Microbatch/accumulation từng architecture | Pending memory/timing; cùng effective batch |
| Bốn base LR candidates | Freeze đúng 4 values sau bounded stability sanity-check, trước official search; không refine theo validation Macro-F1 |
| AdamW betas/eps | Pending numerical recipe; explicit chung |
| Max epochs | Pending đủ pilot curves; chung ba architecture |
| LR warmup length/start factor | Pending; cùng policy |
| Cosine final LR/floor và update horizon | Pending; chung policy, scheduler theo optimizer updates |
| Early-stopping patience/min-delta/tolerance | Pending; cùng rule, dựa supported Macro-F1 |
| Tie-break checkpoint | Freeze trước search: ưu tiên checkpoint sớm hơn trong tolerance hoặc một rule explicit đã thống nhất; không chọn tùy ý |
| AMP dtype/GradScaler | Pending UIT runtime; explicit và giống nhau |
| Runtime/timing policy | DataLoader, TF32, cuDNN, compile, matmul/determinism và measurement protocol freeze trước comparable timing |
| Gradient clipping | Pending stability; explicit norm/threshold hoặc tắt, không bật riêng một architecture sau khi thấy kết quả |
| Brightness/contrast amplitudes | Pending mild-range preview/pilot; freeze chung, không mở augmentation search branch |
| Training/tuning seeds và DataLoader RNG policy | Exact seed IDs, batch-order/augmentation generators và resume policy phải được freeze; không đổi fixed split |
| New-head initialization | Freeze deterministic rule/version/seed behavior cho E1/E1-HR/E2; E3 natural/cRT cùng initial weights theo source seed |
| Validation cadence | Full validation mỗi epoch; lưu toàn bộ history |

Không điền các con số này bằng suy đoán rồi mô tả là kết quả UIT pilot. Khi đã freeze, lưu một versioned training recipe và hash riêng, giữ preprocessing v1 không đổi.

## Official search và final runs

E1/E1-HR/E2: **4 LR × 1 seed mỗi architecture**, cùng budget; chọn LR theo supported validation Macro-F1. Sau đó **3 final seeds** cho selected configuration, không 4 LR × 3 seeds. Lưu và công bố cả trial không được chọn. Không dùng test trong pilot/search/early stopping.

## Small E3 pilot — bounded, common protocol

Chọn **một predetermined source training seed trước khi chạy**; exact ID hiện pending. Dùng frozen E1/E2 representations của source seed đó, eval mode và deterministic feature cache đúng v1. Chỉ train + validation, không test. Khai báo/review trước **bounded head-pilot budget**, gồm step/sample/epoch limit và diagnostic-attempt/stop rules; không train open-ended, mở full LR grid hoặc repeated trial search theo validation.

Mục tiêu là freeze **một common linear-head protocol cho tất cả E3 branches**, gồm optimizer, LR, schedule, initialization, step/sample budget và checkpoint-selection rule. Cả E1-feature và E2-feature dùng cùng protocol; khác feature dimension không tự cho phép chọn optimizer/LR/budget riêng. Nếu có lý do kỹ thuật bắt buộc phải khác protocol giữa hai representations, **report lại trước, không tự quyết**.

Trong mỗi cặp natural/cRT cùng representation/source seed, source checkpoint, cache, initial head weights và toàn bộ common protocol giống nhau; **chỉ sampling policy khác**. Class-uniform là chọn đều lớp rồi chọn đều ảnh trong lớp. Khai báo số draws/steps bằng nhau; natural/cRT replacement mechanics thuộc sampling policy, không được âm thầm làm khác training budget.

Sau small pilot, freeze recipe trước khi chạy các nhánh head của ba source seeds. Ghi cache identity bằng source checkpoint SHA-256, preprocessing hash, split/label hashes và feature definition; không trộn cache giữa source seeds. Việc mô tả bước sau không authorize chạy ở hiện tại.

## Pilot outputs — chỉ tạo sau khi được authorize và hoàn tất

Pilot sau này phải tạo hai loại artifact:

1. **`reports/pilot/pilot_results.md`**: environment/versions, seed và runtime policy, predeclared P0/P1/head-pilot budgets, measurements, pass/fail, failed attempts và rationale cho từng numeric value được freeze. Ghi log ordered E1-HR candidate decisions và LR sanity-checks để phân biệt với official score-based tuning.
2. **Một versioned frozen training recipe dưới `configs/training/`**, ví dụ đường dẫn dạng `<recipe_id>.json` kèm `<recipe_id>.sha256`: exact numeric values, seeds, runtime policy, optimizer/scheduler, batch/accumulation, augmentation amplitudes, initialization, E1-HR resolution, checkpoint rules và E3 common head protocol; kèm dataset/preprocessing references và checksum/hash.

Các tên placeholder không phải recipe đã freeze. **Không tạo results hoặc frozen training recipe trong task patch này.** Khi có recipe được review/freeze, `master_spec.md` chỉ reference nguồn đó, không duplicate toàn bộ numeric contract. Không sửa master spec trong task hiện tại.

## Resume và acceptance

Slurm job walltime có thể yêu cầu nhiều job nối tiếp. Resume tại epoch boundary, phục hồi optimizer/scheduler/GradScaler và RNG/update counters. Kiểm tra một interrupted/resumed diagnostic run có cùng epoch budget và scheduler trajectory như kế hoạch; không reset warmup hoặc thêm epochs khi resume. Resume không gia hạn bounded P1/head-pilot budget hoặc tạo diagnostic attempts mới ngoài protocol đã review.

Chỉ khi các gates trên pass và training parameters được freeze mới chuyển sang official LR search. Giữ implementation/training approval riêng theo chỉ thị người dùng; tài liệu này không chạy Slurm job.
