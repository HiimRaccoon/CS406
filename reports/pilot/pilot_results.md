# UIT pilot results — completed, diagnostic only

Cập nhật 2026-10-31 từ evidence người dùng cung cấp trong `uit_2026-10-31/`, được lưu nguyên byte tại [evidence/uit_2026-10-31/](evidence/uit_2026-10-31/SHA256SUMS.txt). Đây là báo cáo **pilot đã thực thi trên UIT**, không phải kết quả official E1/E1-HR/E2 hoặc test evaluation. Task lập báo cáo chỉ đọc archived metadata và train-only augmentation preview; không chạy lại model/GPU/Slurm.

## Kết luận và phạm vi

| Gate | Kết quả | Evidence |
|---|---|---|
| Gate 1 — dataset/environment/pretrained integrity | PASS cho pilot đã ghi nhận | Target conformance, environment/config/model metadata của sáu runs |
| Gate 2 — UIT preprocessing conformance v1 | PASS | [conformance_report.json](evidence/uit_2026-10-31/conformance/conformance_report.json) |
| Gate 3 / strict P0 — E1, E1-HR, E2 | PASS cả ba, `criteria`, `assessed`, full epoch | P0 summaries/config/timing/memory/context |
| Bounded P1 — E1, E1-HR, E2 | COMPLETE cả ba, 3/3 epochs | P1 summaries/history/CSV/timing/memory |
| Official training recipe, LR search, final seeds/test | Chưa thực hiện; recipe chưa fully frozen | Còn official-only decisions cần review |

User đã review pilot đủ để chuyển sang official specification/implementation; **không cần P1-v2**. Official implementation/search vẫn **chưa bắt đầu**. Không sử dụng các diagnostic checkpoints làm official runs hoặc warm-start cho official architectures.

## Evidence provenance và integrity

Snapshot gồm **47 payload files + original SHA256SUMS.txt**, không có model weights/checkpoints hoặc raw test images. Folder `p0/<profile>` ánh xạ tới server run `p0-strict-<profile>-v1`; `p1/<profile>` tới `p1-<profile>-v1`. Absolute server paths trong resolved configs được giữ nguyên làm lịch sử execution, không phải đường dẫn portable để chạy trên máy khác.

- Source Git commit của cả sáu runs: `b2ac6eebac8a29aa7ad548b810a6d2540f2a258f`, `git_dirty=false`.
- Source fingerprint: `ed27cca10c183cdacedb04ab66558f91c68d41c00d9f67d72e86a85fc2990553`.
- Preprocessing SHA-256: `4f49dd41738168cfc65d4322fbe1b659ef5595229bb487ad86dd671a26a8903f`.
- Pretrained file: `convnext_tiny-983f1562.pth`, enum `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`.
- Full pretrained SHA-256 đã pin và được ghi nhất quán trong config/model/identity: `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`. Task này không tải hoặc hash lại weight file ngoài snapshot.

| Fixed artifact | SHA-256 đối chiếu khớp local frozen artifacts |
|---|---|
| manifest.csv | `9cbd8b70e8522b5aa4a20fba4fa7b2ecea855cf377f98e41ebb162a07f8a8040` |
| summary.json | `cf3dcf909aefd17c531cd115d6c30e54bbeb3dbd13962f974317ccde8a2f3e4b` |
| labels.json | `5f7e8f5810e9ac8c43c1c22b2e4febe2e7a6174629b9031d3e943fce511832aa` |
| class_support.csv | `ff46f4074464644f573b4ca64eb61320a61a8cc3daed3a3886a350cbc3c6fb75` |

**Checksum exception:** cả **47 payload hashes khớp** manifest được cung cấp. Manifest có thêm dòng tự trỏ tới `SHA256SUMS.txt`, expected `18c2d02f19577cbc93ee6f9b2593b45ff5c38eac424d75e95dc7b9dbdbc5a4d3`, trong khi full digest của manifest hiện tại là `bf7610217a5b2a5b744c7f68d2eb9608640a9da9de712226c493bb5db8a2fa91`. Không khẳng định toàn bộ 48 dòng đều pass và không sửa file gốc để che khác biệt này. Exception nằm ở self-entry, không ở payload dùng cho các gate.

Đã kiểm tra offline: context/fingerprint strict P0 hợp lệ; config/timing/memory/context khớp summaries; P1 có đúng ba strict P0 references và metadata đáp ứng các common batch/runtime/seed/init/identity checks của gate. History, metrics CSV, timing và memory khớp từng epoch. Đây là kiểm tra archived evidence, không phải rerun CUDA hoặc forensic chứng minh độc lập mọi hoạt động trên server. Snapshot không chứa scheduler/accounting log đầy đủ, exploratory measurement-only runs hoặc một UIT interrupted/resumed diagnostic riêng; không suy diễn các kết quả này.

## Environment và runtime đã đo

| Thành phần | Actual pilot record |
|---|---|
| Python / OS | 3.11.17; Linux 6.8.0-106-generic x86_64, glibc 2.39 |
| PyTorch / TorchVision | 2.8.0+cu128 / 0.23.0+cu128 |
| Pillow / numpy / pandas / scikit-learn | 11.3.0 / 2.1.3 / 2.2.3 / 1.6.1 |
| CUDA / cuDNN / driver | 12.8 / 91002 / 580.126.20 |
| GPU | NVIDIA H200 MIG 2g.35gb; logical total 34.896.609.280 bytes = 32.5 GiB |
| Allocation | 4 CPUs; SLURM_MEM_PER_NODE=16384 MiB; per-run MIG UUIDs khác nhau, cùng device capacity/type |
| Data root | `/datastore/binhvp/BuiTrongTan/CS406/data/geoguessr` |
| Batch | Microbatch 2, accumulation 1, effective 2 **original images** cho cả ba profiles |
| DataLoader | num_workers=2, pin_memory=true, persistent_workers=true, prefetch_factor=2, CPU threads=2 |
| Numerics | FP32; autocast/GradScaler/TF32/torch.compile off; float32 matmul precision highest |
| Determinism | cuDNN benchmark=false, cuDNN deterministic=true, deterministic_algorithms=true, warn_only=false |
| Runtime env | CUBLAS_WORKSPACE_CONFIG=:4096:8; OMP_NUM_THREADS=2; MKL_NUM_THREADS=2; PYTHONHASHSEED=3406 |
| New classifier init | linear_default: kaiming_uniform_(a=sqrt(5)); bias U(-1/sqrt(fan_in),+1/sqrt(fan_in)); isolated CPU RNG seed 3406 rồi restore outer RNG; fan-in 768 (E1/E1-HR), 1536 (E2) |

Source: sáu `environment.json`/`config_resolved.json` files. Recorded Slurm job IDs: P0 **7760/7761/7762**, P1 **7947/7948/7949** cho E1/E1-HR/E2. `SLURM_TIMELIMIT` không được capture trong các records; không tự điền walltime allocation từ con số minh họa của runbook. Conformance ghi target package environment với GPU=null, không phải GPU training run; GPU/runtime policy lấy từ P0/P1 records.

## Target preprocessing conformance — PASS

[Target report](evidence/uit_2026-10-31/conformance/conformance_report.json) ghi:

- Exact counts: **34.997 train / 7.497 val / 7.502 test**, 124 output labels; không regenerate split.
- Hash audit đủ **34.997 train + 7.497 validation images**, theo target dataset root/fixed fingerprints. Đây là audit được ghi trên UIT; task documentation không đọc lại raw dataset để lặp audit.
- Hai representative **train** layouts 1536×662 và 1030×703: footer/crops, global/local tensor shapes, fill/padding, deterministic behavior và local-from-raw source checks PASS; không dùng `weights.transforms()`.
- Frozen preprocessing/config không thay đổi. Conformance report ghi **test_images_opened=0**.

## Strict P0 — PASS cả ba profiles

Actual protocol `cs406-p0-strict-fp32-v1`: `feasibility.mode=criteria`, warmup **10** optimizer updates, steady-state **100** updates, sau đó full train epoch **34.997** ảnh. Common constraints trong configs: max peak reserved **8 GiB**, min steady-state **6 original images/s**, max full train epoch **6000 s**. Đây là tiêu chí acceptance của pilot đã được ghi nhận, không phải efficiency objective hoặc official training hyperparameters; không suy diễn rationale engineering khác ngoài dữ liệu được cung cấp.

| Profile / actual run | WH global | Full train epoch (s) | Full-epoch img/s | Steady-state img/s | Measured allocated / reserved GiB | Full-epoch reserved GiB | Status |
|---|---|---:|---:|---:|---:|---:|---|
| [E1 / p0-strict-e1-v1](evidence/uit_2026-10-31/p0/e1/run_summary.json) | 896×384 | 2080.53 | 16.8212 | 16.9549 | 1.81491 / 1.99023 | 1.99414 | pass, assessed |
| [E1-HR / p0-strict-e1hr-v1](evidence/uit_2026-10-31/p0/e1hr/run_summary.json) | 1344×576 | 4473.22 | 7.8237 | 7.8568 | 3.57919 / 3.88672 | 3.88672 | pass, assessed |
| [E2 / p0-strict-e2-v1](evidence/uit_2026-10-31/p0/e2/run_summary.json) | 896×384 + 4 locals 320×320 | 4321.36 | 8.0986 | 8.2201 | 3.49096 / 3.72461 | 3.73438 | pass, assessed |

Phân biệt **steady-state** với **full-epoch throughput**, và measured-window peak với full-epoch peak; không dùng lẫn tên. Timings là end-to-end train batches gồm loader/transfer/numerical checks, **không gồm validation/checkpoint I/O**. P0 tắt photometric jitter. GPU synchronization và reset peak-memory được thực hiện theo runner đã pin; JSON lưu workload/window/policy tương ứng.

Ba summaries có `full_epoch_measured=true`, `failures=[]`, `failure_kind=null`, `test_images_opened=0`; mode/context fingerprints hợp lệ. Cùng batch/runtime/seed/init/pretrained/data/preprocessing/source identity, được P1 gate chấp nhận. E1-HR ordered list chỉ có **1344×576**, first candidate pass nên resolution được accepted; không fallback hoặc chọn theo validation score. HR record vẫn mang criteria_id lịch sử `cs406-prep0-hr-v1-exploratory-a`, nhưng actual protocol/mode/thresholds/context là strict; giữ nguyên ID, không sửa history.

Full-epoch P0 data-wait fraction: E1 **1.26%**, E1-HR **0.44%**, E2 **1.43%**. Không có bằng chứng loader chặn phần lớn thời gian trong các P0 windows này; không claim worker count optimal. Không có exploratory measurement-only archive trong bundle này, nên không tạo bảng số liệu giả cho A/B hoặc dùng chúng làm acceptance evidence.

## Bounded P1 — COMPLETE, chỉ diagnostic

Common protocol `cs406-p1-bounded-v1`: pilot seed **3406**, AdamW diagnostic LR **1e-4**, betas **[0.9,0.999]**, eps **1e-8**, decay **0.05**, clipping off. Budget đã khai báo **3 epochs / 52.497 optimizer updates** mỗi profile; **17.499 updates/epoch**. Mỗi run khởi tạo độc lập từ cùng pretrained checkpoint, không tiếp tục P0 training.

P1 brightness/contrast **0.10/0.10**; FP32 deterministic policy/batch như bảng runtime. Scheduler: warmup **1000** updates, start factor **0.1**, sau đó cosine tới **1e-6** trên horizon **52.497** updates. Checkpoint dùng supported validation Macro-F1, tolerance **1e-6**, tie-break **earlier**. Những số này mô tả pilot đã chạy; không mặc định áp dụng horizon hoặc numeric schedule này cho official 8-epoch recipe.

| Profile / actual run | Epoch 1 Macro-F1 | Epoch 2 Macro-F1 | Epoch 3 Macro-F1 | Completed | Best diagnostic epoch |
|---|---:|---:|---:|---|---:|
| [E1 / p1-e1-v1](evidence/uit_2026-10-31/p1/e1/history.json) | 0.191439 | 0.347943 | 0.395442 | 3/3 | 3 |
| [E1-HR / p1-e1hr-v1](evidence/uit_2026-10-31/p1/e1hr/history.json) | 0.196432 | 0.359160 | 0.419519 | 3/3 | 3 |
| [E2 / p1-e2-v1](evidence/uit_2026-10-31/p1/e2/history.json) | 0.212609 | 0.382093 | 0.449134 | 3/3 | 3 |

Các metric sau lấy **cùng epoch 3 checkpoint được chọn**, không lấy best metric từ epoch khác. Validation có **7.497 samples, 106 GT-supported classes**, Top-k rank đủ 124 outputs.

| Profile | Top-1 | Top-3 | Top-5 | Macro-F1 supported | Weighted-F1 | Validation NLL |
|---|---:|---:|---:|---:|---:|---:|
| E1 | 0.761238 | 0.898093 | 0.936508 | 0.395442 | 0.747391 | 0.856378 |
| E1-HR | 0.773643 | 0.901561 | 0.941176 | 0.419519 | 0.760716 | 0.813772 |
| E2 | 0.775110 | 0.908630 | 0.943311 | 0.449134 | 0.762730 | 0.802901 |

Source đối chiếu: [E1 CSV](evidence/uit_2026-10-31/p1/e1/metrics.csv), [E1-HR CSV](evidence/uit_2026-10-31/p1/e1hr/metrics.csv), [E2 CSV](evidence/uit_2026-10-31/p1/e2/metrics.csv) và respective histories/summaries.

**Không xếp hạng phương pháp chính thức từ bảng pilot này.** Chỉ một diagnostic seed/LR, horizon ngắn; chưa official LR search, final 3-seed mean ± std hoặc test evaluation. Macro-F1 còn tăng đến epoch cuối, nên 3-epoch curves không chứng minh đã hội tụ hoặc 8 epochs là tối ưu.

### Scheduler, memory và runtime stability

LR endpoints từ history/CSV, giống nhau cả ba profiles và khớp formula scheduler đã pin:

| Epoch | Updates | First update LR | Last update LR |
|---|---:|---:|---:|
| 1 | 17499 | 1.000000e-5 | 7.6974249387e-5 |
| 2 | 17499 | 7.6971697715e-5 | 2.6627687006e-5 |
| 3 | 17499 | 2.6625041615e-5 | 1.000000e-6 |

| Profile | Epoch wall seconds (1 → 2 → 3) | Peak reserved GiB (mỗi epoch) | Peak allocated GiB (range) |
|---|---|---:|---:|
| E1 | 2694.36 → 2767.23 → 2742.34 | 1.99414 | 1.81491 |
| E1-HR | 4695.19 → 4693.04 → 4686.21 | 3.88672 | 3.57919 |
| E2 | 4504.00 → 4490.61 → 4492.67 | 3.73438 | 3.49096–3.49368 |

P1 `epoch_seconds` gồm train và validation, được chốt **trước checkpoint/history writes** theo runner; không mô tả nó là full Slurm-job time hoặc đã gồm checkpoint I/O. Reserved peak không tăng qua ba epochs; losses/metrics hữu hạn, train loss giảm; summaries COMPLETE không báo NaN/divergence/OOM. Không suy ra absence of all hardware incidents ngoài recorded runs.

P1 E1 data-wait fraction **16.4–19.2%**, cao hơn P0; E1-HR khoảng **0.05%**, E2 **0.06–0.18%**. Đây là khác biệt đo được cần giữ khi đọc throughput, không mở loader/augmentation tuning hoặc thêm pilot branch trong task này.

## Augmentation review

[Train-only preview 0.10](evidence/uit_2026-10-31/augmentation/augmentation_preview_010.jpg) có cả hai raw layouts, original/deterministic draw và endpoints B=C=0.90/1.10. Người dùng đã review và **accepted brightness=0.10, contrast=0.10**; P1 configs ghi đúng amplitudes này. Preview dùng để review mild jitter, không phải classification-score selection. UI mask/padding được giữ theo policy; không sửa frozen preprocessing hoặc crop geometry.

## No-test-access record

Conformance và **cả sáu** P0/P1 summaries đều ghi `test_images_opened=0`; conformance samples thuộc train, audit chỉ train+val; mỗi P1 epoch đánh giá validation, không test. P1 summaries đều `official_experiment=false`; P0 ghi `model_quality_or_convergence_claim=false`. Fixed test membership/count chỉ đọc qua metadata, không mở ảnh. Task documentation này không mở/evaluate test images, không chạy server job. Phạm vi bảo đảm là các execution records được cung cấp và các kiểm tra hiện tại, không xóa lịch sử test của pipeline cũ.

## Quyết định sau pilot và các việc còn NEEDS REVIEW

**Đã review:** pilot đủ để đi tiếp, không cần P1-v2; **official max_epochs=8** frozen cho common main budget. Đây là quyết định do user cung cấp sau P1, tách khỏi measured P1 budget=3. E1-HR **1344×576** accepted theo feasibility; common **pilot** batch=2 đã freeze; jitter **0.10/0.10** đã accepted bằng train-only review.

Trước official E1 runs còn phải review/freeze:

- **Exact four-LR numeric grid** và **1 tuning seed ID + 3 final training seed IDs**; không promote diagnostic LR **1e-4** hoặc pilot seed **3406**. Protocol giữ 4 LR × 1 tuning seed/architecture, rồi best LR × 3 final seeds; split seed **406**, membership không đổi.
- Official adoption/contract cho batch/accumulation, loader/FP32/runtime/optimizer/init/environment và RNG/resume policy. Pilot có exact values/evidence nhưng không tự tạo official config.
- Official warmup duration/start factor, cosine horizon/floor phù hợp **8 epochs**, early-stopping patience/min-delta, checkpoint tolerance/tie-break. Pilot numeric schedule/selection có thể dùng làm reference để review, không tự kế thừa.
- Official E1 specification và review/authorization implementation, smoke, LR tuning, final runs. Các checklist official E1/E1-HR/E2/E3/test vẫn mở; E3 common head protocol thuộc phase sau, không blocker cho việc viết E1 spec và không được coi là đã chạy.

**Không tạo `training_recipe_v1` hoặc official experiment configs** khi các trường này còn mở. Master spec chỉ tham chiếu report/status và recipe được review ở task sau; không duplicate toàn bộ numeric contract hoặc chuyển pilot code thành official framework.
