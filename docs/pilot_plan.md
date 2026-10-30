# UIT pilot và các training parameters còn cần freeze (DRAFT — deferred)

**Trạng thái: DRAFT, deferred; chưa được phép triển khai hoặc chạy.** Giữ tài liệu này như ghi chú thảo luận hiện có, không tiếp tục mở rộng cho tới khi người dùng review xong toàn bộ grill/preprocessing và yêu cầu bước tiếp theo. Công việc hiện tại chỉ gồm preprocessing/geometry/split reproducibility; không tự động chuyển sang pilot hoặc tạo training code.

Chưa có kết quả GPU benchmark trong workspace local và chưa bắt đầu implementation/training. Các gate dưới đây không phải authorization để chạy experiment.

## Gate 1: dataset và environment

1. Chọn runtime root `/datastore/binhvp/<my-folder>/`, giữ đường dẫn trong run/Slurm config; không sửa các relative paths của manifest.
2. Xác minh SHA-256 của bốn artifact theo `provenance.json`, original summary fingerprints, label IDs, split counts và class support. Trước pilot đầu tiên trên một bản dataset mới, kiểm tra file tồn tại và SHA-256 ảnh theo manifest; lưu audit gắn với dataset root/fingerprint, không hash lại toàn bộ mỗi epoch.
3. Ghi package lock của PyTorch/TorchVision/Pillow và phiên bản CUDA/cuDNN/driver, GPU/MIG name, CPU/RAM allocation. Mọi main experiment dùng cùng environment lock. Không tự đổi backend khi một dependency không hỗ trợ contract.
4. Tải đúng pretrained file `convnext_tiny-983f1562.pth`, lưu full SHA-256; dùng cùng file cho cả ba architecture. Không dùng default alias để thay checkpoint ngoài ý muốn.

## Gate 2: conformance với preprocessing v1

Đọc config và checksum từ `configs/preprocessing/`. Dùng chính thức tensor backend đã khai báo, không dùng ảnh PNG preview làm model input. Chỉ dùng train samples cố định của hai layout cho các kiểm tra này; không dùng score để sửa geometry.

- Raw dimensions, footer crop, UI rectangles, local coordinates và số view phải khớp integer tables.
- Global có tensor shape `[3,384,896]`; mỗi local `[3,320,320]`. Full-image content/padding khớp 896×378 với pad 0/3/0/3 hoặc 574×384 với pad 161/0/161/0.
- Input float32 RGB [0,1], exact mean fill trước normalization, bilinear tensor resize với `antialias=True`, không intermediate uint8, không `weights.transforms()` hoặc PIL/System.Drawing resize.
- Kiểm tra pixel interior của mask và padding sau normalize trong numerical tolerance đã ghi nhận; kiểm tra riêng edge behavior có antialias, không đòi mọi pixel sát biên đều zero.
- Một reference tile lấy từ UI-processed raw phải khớp crop/resize cùng rectangle trực tiếp; không khớp một pipeline crop từ resized global.
- Repeated deterministic evaluation preprocessing của cùng ảnh trong cùng environment phải khớp. Train jitter dùng cùng một draw cho mọi view và giữ fill/padding; val/test/E3 cache tắt jitter.
- Nếu backend/library update làm khác numerical outputs, ghi nhận và xử lý versioning, không đổi v1 âm thầm. Numerical tolerance trong cùng runtime phải chặt; không tuyên bố bitwise equality giữa GPU/CPU hoặc giữa các decoder khác nhau khi chưa kiểm tra.

Gate này kiểm chứng implementation của specification đã freeze; không phải geometry search. Python launcher local không khả dụng trong phiên chuẩn bị spec nên gate chưa được thực thi bằng TorchVision tại local.

## Gate 3: feasibility và timing

Một training seed chung cho pilot E1/E1-HR/E2. E1 global 896×384; E2 đúng global + bốn local của v1. E1-HR bắt đầu từ candidate 1344×576 nếu phù hợp GPU, chưa khóa resolution và không cố match compute E2. Nếu cần thay candidate vì OOM/throughput hoặc aspect ratio, ghi nhận quyết định trước official LR search; không chọn bằng test score.

Đo time/epoch, samples/second theo ảnh gốc, peak allocated/reserved VRAM, microbatch, accumulation, CPU/data-loader bottleneck và training stability. Chạy warmup cho benchmark GPU trước timing; phân biệt benchmark warmup với LR warmup. Có ít nhất một epoch đầy đủ để phản ánh I/O và tail batches, không chỉ dựa vào vài forward passes.

Tìm microbatch khả thi cho từng architecture rồi chọn common effective batch size tính theo ảnh gốc. Microbatch/accumulation được phép khác nhưng số mẫu trên optimizer update và loss scaling phải nhất quán, kể cả accumulation cuối epoch. Không xem năm view của một ảnh là năm training samples.

Để chốt max epochs/patience cần xem learning/validation curves của pilot đủ dài; một benchmark throughput ngắn không chứng minh đã hội tụ. Không tự giảm training horizon của architecture chậm hơn. Nếu phải giảm compute, điều chỉnh common protocol trước official search.

## Những trường cần điền trước official LR search

| Trường | Trạng thái/quy tắc |
|---|---|
| E1-HR full-image WH | Pending pilot; full-image resolution cao hơn, giữ aspect ratio hợp lý |
| Common effective batch size | Pending; tính theo ảnh gốc |
| Microbatch/accumulation từng architecture | Pending memory/timing; cùng effective batch |
| Bốn base LR candidates | Pending pilot; một tập chung, head multiplier 1 |
| AdamW betas/eps | Pending numerical recipe; explicit chung |
| Max epochs | Pending đủ pilot curves; chung ba architecture |
| LR warmup length/start factor | Pending; cùng policy |
| Cosine final LR/floor và update horizon | Pending; chung policy, scheduler theo optimizer updates |
| Early-stopping patience/min-delta/tolerance | Pending; cùng rule, dựa supported Macro-F1 |
| Tie-break checkpoint | Freeze trước search: ưu tiên checkpoint sớm hơn trong tolerance hoặc một rule explicit đã thống nhất; không chọn tùy ý |
| AMP dtype/GradScaler | Pending UIT runtime; explicit và giống nhau |
| Gradient clipping | Pending stability; explicit norm/threshold hoặc tắt, không bật riêng một architecture sau khi thấy kết quả |
| Brightness/contrast amplitudes | Pending mild-range preview/pilot; freeze chung, không mở augmentation search branch |
| Training/tuning seeds và DataLoader RNG policy | Freeze danh sách seeds, batch-order/augmentation generators, resume policy |
| New-head initialization | Freeze explicit cùng rule; natural/cRT cùng initial state theo source seed |
| Validation cadence | Full validation mỗi epoch; lưu toàn bộ history |

Không điền các con số này bằng suy đoán rồi mô tả là kết quả UIT pilot. Khi đã freeze, lưu một versioned training recipe và hash riêng, giữ preprocessing v1 không đổi.

## Official search và final runs

E1/E1-HR/E2: **4 LR × 1 seed mỗi architecture**, cùng budget; chọn LR theo supported validation Macro-F1. Sau đó **3 final seeds** cho selected configuration, không 4 LR × 3 seeds. Lưu và công bố cả trial không được chọn. Không dùng test trong pilot/search/early stopping.

## Small E3 pilot

Dùng frozen representations của checkpoint nguồn, eval mode, deterministic feature cache đúng v1. Chốt một linear-head recipe đủ hội tụ: optimizer/LR, reset initialization, schedule, virtual-epoch sample count, optimizer-step budget và selection rule. Không full LR grid branch.

Trong mỗi cặp natural/cRT cùng representation, mọi trường trên và initial head weights giống nhau; chỉ sampling đổi. Class-uniform là chọn đều lớp rồi chọn đều ảnh trong lớp. Khai báo số draws/steps bằng nhau; natural sampling replacement mechanics và cRT replacement mechanics thuộc sampling policy, không được âm thầm biến thành khác training budget.

Sau small pilot, freeze recipe rồi chạy các nhánh head của ba source seeds. Ghi cache identity bằng source checkpoint SHA-256, preprocessing hash, split/label hashes và feature definition; không trộn cache giữa source seeds.

## Resume và acceptance

Slurm job walltime có thể yêu cầu nhiều job nối tiếp. Resume tại epoch boundary, phục hồi optimizer/scheduler/GradScaler và RNG/update counters. Kiểm tra một interrupted/resumed diagnostic run có cùng epoch budget và scheduler trajectory như kế hoạch; không reset warmup hoặc thêm epochs khi resume.

Chỉ khi các gates trên pass và training parameters được freeze mới chuyển sang official LR search. Giữ implementation/training approval riêng theo chỉ thị người dùng; tài liệu này không chạy Slurm job.
