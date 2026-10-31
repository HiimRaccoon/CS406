# UIT pilot — execution completed; official recipe còn cần review

**Trạng thái execution (2026-10-31): UIT conformance PASS; strict P0 PASS; bounded P1 COMPLETE cho E1/E1-HR/E2.**

- **Pilot harness implementation: implemented**, self-contained trong `src/pilot/`; preprocessing reusable trong `src/preprocessing/`.
- **Pilot execution: completed trên UIT**; archived evidence từ commit `b2ac6eebac8a29aa7ad548b810a6d2540f2a258f` đã được đối chiếu với config, summary, history, CSV, timing và memory.
- **Official E1/E1-HR/E2 implementation: chưa bắt đầu**; pilot code không phải official framework.
- **Master spec vẫn draft, chưa authorize official experiments**. Các gate bên dưới không tự cấp phép chạy pilot/experiment.

User đã review pilot đủ để chuyển sang official specification/implementation; **không cần P1-v2**. Official **max_epochs=8** đã được review/freeze sau P1; **exact four-LR grid và official tuning/final seed IDs vẫn NEEDS REVIEW**. Diagnostic LR `1e-4` và pilot seed `3406` không tự thành official values. Chưa fully freeze official recipe.

Task cập nhật này chỉ ghi status/report, không chạy lại GPU/Slurm hoặc sửa source/frozen preprocessing. Xem [pilot results](../reports/pilot/pilot_results.md) và [evidence snapshot](../reports/pilot/evidence/uit_2026-10-31/SHA256SUMS.txt). Các nguyên tắc bên dưới vẫn là protocol; những yêu cầu pre-run đã được đáp ứng có ghi trạng thái, không viết lại review lịch sử như thể số đo đã có từ trước.

## Gate 1: dataset và environment

**PASS cho actual UIT pilot.** Root `/datastore/binhvp/BuiTrongTan/CS406/data/geoguessr`; fingerprints bốn fixed artifacts khớp repo, counts **34997/7497/7502** không đổi; target report ghi hash audit đủ train+val và zero test images. Python **3.11.17**, torch **2.8.0+cu128**, torchvision **0.23.0+cu128**, CUDA **12.8**, cuDNN **91002**, H200 MIG **2g.35gb**, driver **580.126.20**, package/runtime records đã lưu. Full pretrained SHA-256 đã pin trong cả sáu configs và model metadata: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`.

1. Chọn runtime root `/datastore/binhvp/<my-folder>/`, giữ đường dẫn trong run/Slurm config; không sửa các relative paths của manifest.
2. Xác minh SHA-256 của bốn artifact theo `provenance.json`, original summary fingerprints, label IDs, split counts và class support. Trước pilot đầu tiên trên một bản dataset mới, kiểm tra file tồn tại và SHA-256 ảnh theo manifest; lưu audit gắn với dataset root/fingerprint, không hash lại toàn bộ mỗi epoch.
3. Ghi package lock của PyTorch/TorchVision/Pillow và phiên bản CUDA/cuDNN/driver, GPU/MIG name, CPU/RAM allocation. Mọi main experiment dùng cùng environment lock. Không tự đổi backend khi một dependency không hỗ trợ contract.
4. Tải đúng pretrained file `convnext_tiny-983f1562.pth`, tính/review full SHA-256 và **bắt buộc pin `expected_weight_sha256` trước P0/P1**; dùng cùng file cho cả ba architecture. Config check từ chối null/invalid full hash; model chỉ load khi actual full digest khớp chính xác expected digest. Conformance không cần weight. Không dùng default alias hoặc chỉ SHA prefix làm integrity gate.

## Gate 2: conformance với preprocessing v1

**PASS trên target UIT environment/root.** [Report](../reports/pilot/evidence/uit_2026-10-31/conformance/conformance_report.json) xác nhận cả hai train layouts, tensor/fill/padding/raw-local/determinism checks; audit **34997 train + 7497 val** hashes; `test_images_opened=0`. Frozen preprocessing v1 không thay đổi.

Đọc config và checksum từ `configs/preprocessing/`. Dùng chính thức tensor backend đã khai báo, không dùng ảnh PNG preview làm model input. Chỉ dùng train samples cố định của hai layout cho các kiểm tra này; không dùng score để sửa geometry.

- Raw dimensions, footer crop, UI rectangles, local coordinates và số view phải khớp integer tables.
- Global có tensor shape `[3,384,896]`; mỗi local `[3,320,320]`. Full-image content/padding khớp 896×378 với pad 0/3/0/3 hoặc 574×384 với pad 161/0/161/0.
- Input float32 RGB [0,1], exact mean fill trước normalization, bilinear tensor resize với `antialias=True`, không intermediate uint8, không `weights.transforms()` hoặc PIL/System.Drawing resize.
- Kiểm tra pixel interior của mask và padding sau normalize trong numerical tolerance đã ghi nhận; kiểm tra riêng edge behavior có antialias, không đòi mọi pixel sát biên đều zero.
- Một reference tile lấy từ UI-processed raw phải khớp crop/resize cùng rectangle trực tiếp; không khớp một pipeline crop từ resized global.
- Repeated deterministic evaluation preprocessing của cùng ảnh trong cùng environment phải khớp. Train jitter dùng cùng một draw cho mọi view và giữ fill/padding; val/test/E3 cache tắt jitter.
- Nếu backend/library update làm khác numerical outputs, ghi nhận và xử lý versioning, không đổi v1 âm thầm. Numerical tolerance trong cùng runtime phải chặt; không tuyên bố bitwise equality giữa GPU/CPU hoặc giữa các decoder khác nhau khi chưa kiểm tra.

Gate này kiểm chứng implementation của specification đã freeze; không phải geometry search. Local CPU gate đã có từ bước implement harness; target UIT gate nay cũng đã pass trước pilot. Conformance tự nó không phải P0/P1 hoặc GPU timing benchmark.

## Gate 3: hai stage pilot và các pre-run rules

**Gate 3 P0 PASS; bounded P1 PASS/COMPLETE cả ba profiles.** E1 global 896×384; E2 giữ đúng global + bốn local của preprocessing v1. Actual common pilot seed **3406**, khác split seed **406**; common pilot effective batch **2** (microbatch **2**, accumulation **1**). **Không dùng test trong P0 hoặc P1**; cả sáu summaries ghi `test_images_opened=0`.

### P0 — feasibility/timing pilot

**Completed:** `p0-strict-e1-v1`, `p0-strict-e1hr-v1`, `p0-strict-e2-v1` đều `pass`, `assessed`, `criteria`, `full_epoch_measured=true`. Actual windows **10 warmup + 100 measured updates**, rồi full train epoch; common criteria **max reserved 8 GiB / min steady-state 6 img/s / max train epoch 6000 s**. Full-epoch seconds lần lượt **2080.53 / 4473.22 / 4321.36**, full-epoch img/s **16.82 / 7.82 / 8.10**. [Report](../reports/pilot/pilot_results.md) phân biệt measured-window và full-epoch peaks, steady-state/full-epoch throughput; không suy diễn rằng các ngưỡng đã được tối ưu.

P0 chỉ xác minh:

- Preprocessing và model forward/backward hoạt động.
- Microbatch khả thi, OOM hoặc numerical failure.
- Peak allocated/reserved VRAM.
- Steady-state throughput theo original images.
- Full-epoch wall time và DataLoader/CPU bottleneck.

P0 không dùng để kết luận convergence hoặc model quality. Phân biệt GPU benchmark warmup với LR warmup. Có ít nhất một epoch đầy đủ để phản ánh I/O và tail batches, không chỉ vài forward passes; measurement protocol phải được chốt trước khi đo.

Tìm microbatch khả thi rồi chọn common effective batch size tính theo ảnh gốc. Microbatch/accumulation được phép khác nhưng số mẫu trên optimizer update và loss scaling phải nhất quán, kể cả accumulation cuối epoch. Năm view của một ảnh E2 vẫn là một training sample.

### P1 — bounded training/convergence pilot

**Completed:** `p1-e1-v1`, `p1-e1hr-v1`, `p1-e2-v1` đều COMPLETE **3/3 epochs**, **17499 updates/epoch**, tổng **52497**. Diagnostic LR **1e-4**, AdamW, FP32 deterministic, brightness/contrast **0.10/0.10**; scheduler warmup **1000** updates, start factor **0.1**, cosine floor **1e-6**; checkpoint tolerance **1e-6**, tie-break **earlier**. Bounded horizon không được gia hạn.

Validation Macro-F1 supported qua ba epochs: E1 **0.1914 → 0.3479 → 0.3954**, E1-HR **0.1964 → 0.3592 → 0.4195**, E2 **0.2126 → 0.3821 → 0.4491**; cả ba best diagnostic epoch **3**. LR trajectory khớp giữa profiles, reserved peak ổn định, không reported NaN/divergence/OOM trong các runs complete. **Score chỉ diagnostic, không phải official result hoặc bằng chứng đã hội tụ.** User đã review đủ để đi tiếp, không yêu cầu thêm P1-v2.

P1 xem learning/validation curves và optimization stability để hỗ trợ freeze **common max epochs, LR warmup, early-stopping patience/tolerance và scheduler horizon**. Không suy ra các giá trị này chỉ từ throughput của P0.

**Bounded P1 budget phải được định nghĩa và review trước khi chạy:** giới hạn epochs/optimizer steps, điều kiện dừng và số diagnostic attempts được phép phải explicit, hữu hạn. Actual archived budget là **3 epochs/profile**, một completed v1 run mỗi profile trong bundle; không suy diễn những attempts ngoài evidence. Không train open-ended hoặc tự tăng budget cho đến khi thấy kết quả mong muốn.

Hết budget mà chưa đủ cơ sở kết luận thì report giới hạn/bất ổn để người dùng review, không tự gia hạn hoặc lặp trial theo validation score. Không tự giảm training horizon riêng cho architecture chậm hơn. Nếu phải giảm compute, điều chỉnh common protocol trước official search.

### Deterministic E1-HR resolution selection

**1344×576 đã accepted sau strict P0**: first candidate pass nên dừng, không dùng validation classification score. Actual ordered list chỉ `[[1344,576]]`, không fallback hoặc resolution search; batch policy 2/1 khớp run. Đây là kết quả sau khi đo, không sửa lịch sử first-candidate proposal trong pre-P0 review. Các yêu cầu trước một selection protocol vẫn là:

- Ordered candidate/fallback list, bắt đầu bằng first candidate trên; không tự thêm fallback chưa review.
- Feasibility criteria về raw-image/aspect-ratio suitability, memory và runtime/walltime, kèm batch/accumulation policy được phép và các threshold cần thiết.
- Rule chọn candidate, stop/fail rule và cách record kết quả.

Xét theo đúng thứ tự đã freeze: **candidate đầu pass rule thì chọn/freeze ngay; chỉ thử fallback nếu candidate trước fail rule**. Không thử tiếp để tìm resolution có score cao hơn; không dùng validation classification score hoặc test score để chọn. Không compute-match E2.

Nếu hết approved list mà không candidate nào pass thì report lại; không tự thêm fallback. Đây là feasibility selection theo rule, không phải resolution search. Actual first candidate đã pass; không cần fallback mới.

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

- GPU benchmark warmup và measured window/repetitions; actual strict P0 dùng **10/100 optimizer updates**, sau đó full epoch.
- CUDA synchronization quanh thời điểm bắt đầu/kết thúc timing để hoàn tất GPU work trong window.
- Reset peak-memory statistics sau warmup, trước measured workload; record cả **peak allocated và peak reserved VRAM** trong workload đó.
- Steady-state samples/sec theo **original images**, không theo số global/local views; ghi rõ measured window có bao gồm data loading/transfer hay không, nhất quán giữa các architecture.
- Full-epoch walltime riêng, gồm I/O, tail batches và accumulation cuối epoch; ghi rõ validation/checkpoint overhead có nằm trong timing hay được tách riêng.
- Microbatch, accumulation, effective batch, DataLoader/CPU bottleneck, OOM và numerical failures kèm môi trường cụ thể.

Runtime đã dùng: DataLoader **2** workers, pin/persistent **true**, prefetch **2**, CPU threads **2**; FP32, autocast/scaler/TF32/compile off, matmul **highest**, cuDNN benchmark **false**, cuDNN deterministic và deterministic_algorithms **true** (warn_only **false**). Environment: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `PYTHONHASHSEED=3406`. Đây là actual pilot policy; không tự promote thành official contract. Giữ timing scope và data-wait observations trong [report](../reports/pilot/pilot_results.md).

## New-classifier initialization

Trước pilot phải freeze **một explicit deterministic initialization rule cho new 124-class classifier của E1, E1-HR và E2**, không chỉ cho E3. Record rule cho weights/bias, parameters của rule, exact framework/version và seed/RNG behavior. Mỗi run khởi tạo độc lập, dùng cùng rule; các head khác feature dimension không bắt buộc có cùng shape/weights.

Actual pilot dùng `linear_default` theo PyTorch **2.8.0+cu128**: weight `kaiming_uniform_(a=sqrt(5))`, bias uniform `[-1/sqrt(fan_in),+1/sqrt(fan_in)]`, isolated CPU RNG seed **3406** ngay trước init rồi restore outer RNG. E1/E1-HR fan-in **768**, E2 **1536**. Exact rule/seed behavior đã được ghi trong model metadata, không tune initialization. Official seed IDs/adoption vẫn cần review; không copy pilot seed.

Với E3, natural/cRT của cùng source seed phải bắt đầu từ **identical head weights**, không chỉ cùng tên initialization hoặc seed rule. Rule này cũng thuộc common E3 head protocol được freeze trước official head runs.

## Những trường cần điền trước official LR search

| Trường | Trạng thái/quy tắc |
|---|---|
| E1-HR full-image WH | **Resolved: 1344×576 accepted**, first candidate strict pass, không score selection |
| P1 pilot budget | **Complete: 3 epochs / 52497 updates/profile**, không cần P1-v2; không phải official horizon |
| Common effective batch size | **Pilot frozen: 2 original images**; official adoption/contract **NEEDS REVIEW** |
| Microbatch/accumulation từng architecture | **Pilot: 2/1 cho cả ba**; official adoption **NEEDS REVIEW** |
| Bốn base LR candidates | **NEEDS REVIEW: exact 4 values chưa chọn**; diagnostic 1e-4 không tự thành grid |
| AdamW betas/eps | Pilot **[0.9,0.999] / 1e-8**, decay 0.05; official numeric adoption **NEEDS REVIEW** |
| Official max epochs | **FROZEN AFTER USER REVIEW: 8**, common E1/E1-HR/E2; chưa có official run |
| LR warmup length/start factor | Pilot **1000 / 0.1**; official warmup **NEEDS REVIEW** |
| Cosine final LR/floor và update horizon | Pilot **1e-6 / 52497 updates**; official 8-epoch schedule **NEEDS REVIEW**, không kế thừa horizon 3 epochs |
| Early-stopping patience/min-delta/tolerance | **NEEDS REVIEW** cho official; bounded P1 không chứng minh patience phù hợp; pilot checkpoint tolerance 1e-6 |
| Tie-break checkpoint | Pilot **earlier**, tolerance 1e-6; official explicit rule/adoption **NEEDS REVIEW** |
| AMP dtype/GradScaler | **Pilot: FP32, scaler off** đã đo; official adoption **NEEDS REVIEW** |
| Runtime/timing policy | Actual loader/numerics/environment đã lưu; official lock/adoption **NEEDS REVIEW**, không mở runtime ablation |
| Gradient clipping | **Pilot: off**, không reported instability; official adoption **NEEDS REVIEW** |
| Brightness/contrast amplitudes | **REVIEWED/ACCEPTED: 0.10 / 0.10** từ train-only preview; P1 đã dùng đúng |
| Training/tuning seeds và DataLoader RNG policy | **NEEDS REVIEW: exact 1 tuning + 3 final seed IDs**, RNG/worker/resume contract; không promote pilot **3406**, split vẫn **406** |
| New-head initialization | Pilot exact linear_default rule/version/seed behavior đã ghi; official rule/adoption/seed binding **NEEDS REVIEW**; E3 protocol deferred |
| Validation cadence | Pilot full validation mỗi epoch, history đầy đủ; giữ yêu cầu cùng protocol/checkpoint metrics ở official spec |

Không điền các số còn mở bằng suy đoán hoặc coi mọi pilot setting đã được user freeze cho official training. Sau official-spec review mới tạo versioned training recipe và hash riêng; giữ preprocessing v1 không đổi.

## Official search và final runs

E1/E1-HR/E2: **4 LR × 1 seed mỗi architecture**, cùng budget; chọn LR theo supported validation Macro-F1. Sau đó **3 final seeds** cho selected configuration, không 4 LR × 3 seeds. Lưu và công bố cả trial không được chọn. Không dùng test trong pilot/search/early stopping.

## Small E3 pilot — bounded, common protocol

Chọn **một predetermined source training seed trước khi chạy**; exact ID hiện pending. Dùng frozen E1/E2 representations của source seed đó, eval mode và deterministic feature cache đúng v1. Chỉ train + validation, không test. Khai báo/review trước **bounded head-pilot budget**, gồm step/sample/epoch limit và diagnostic-attempt/stop rules; không train open-ended, mở full LR grid hoặc repeated trial search theo validation.

Mục tiêu là freeze **một common linear-head protocol cho tất cả E3 branches**, gồm optimizer, LR, schedule, initialization, step/sample budget và checkpoint-selection rule. Cả E1-feature và E2-feature dùng cùng protocol; khác feature dimension không tự cho phép chọn optimizer/LR/budget riêng. Nếu có lý do kỹ thuật bắt buộc phải khác protocol giữa hai representations, **report lại trước, không tự quyết**.

Trong mỗi cặp natural/cRT cùng representation/source seed, source checkpoint, cache, initial head weights và toàn bộ common protocol giống nhau; **chỉ sampling policy khác**. Class-uniform là chọn đều lớp rồi chọn đều ảnh trong lớp. Khai báo số draws/steps bằng nhau; natural/cRT replacement mechanics thuộc sampling policy, không được âm thầm làm khác training budget.

Sau small pilot, freeze recipe trước khi chạy các nhánh head của ba source seeds. Ghi cache identity bằng source checkpoint SHA-256, preprocessing hash, split/label hashes và feature definition; không trộn cache giữa source seeds. Việc mô tả bước sau không authorize chạy ở hiện tại.

## Pilot outputs — report đã tạo, official recipe chưa tạo

Các deliverables được phân biệt rõ:

1. **[reports/pilot/pilot_results.md](../reports/pilot/pilot_results.md) — hoàn tất**: actual environment/identity, conformance, strict P0 measurements, P1 curves/runtime, augmentation review, PASS conclusions, no-test records, reviewed decisions và giới hạn evidence. Snapshot nguyên byte ở `reports/pilot/evidence/uit_2026-10-31/`; 47 payload checksums pass, original manifest self-entry không khớp và được ghi rõ. Không tạo exploratory/head-pilot results khi bundle không có evidence đó.
2. **Versioned frozen official training recipe dưới `configs/training/` — CHƯA TẠO**: exact LR grid/seeds và official-only numeric contract còn cần review. Không dùng collector để auto-derive hoặc xuất partial recipe. Sau khi review/freeze, recipe/hash mới là numeric source cho master spec; E3 common head protocol được review riêng ở phase sau.

Master spec chỉ cập nhật checklist/status được evidence hỗ trợ và reference report; không rewrite architecture, duplicate toàn bộ numeric contract hoặc authorize official implementation trong task này.

## Resume và acceptance

Slurm job walltime có thể yêu cầu nhiều job nối tiếp. Resume tại epoch boundary, phục hồi optimizer/scheduler/GradScaler và RNG/update counters. Kiểm tra một interrupted/resumed diagnostic run có cùng epoch budget và scheduler trajectory như kế hoạch; không reset warmup hoặc thêm epochs khi resume. Resume không gia hạn bounded P1/head-pilot budget hoặc tạo diagnostic attempts mới ngoài protocol đã review.

Chỉ khi các gates trên pass và training parameters được freeze mới chuyển sang official LR search. Giữ implementation/training approval riêng theo chỉ thị người dùng; tài liệu này không chạy Slurm job.
