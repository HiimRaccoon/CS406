# Pre-P0 pilot configuration review

## POST-EXECUTION STATUS — 2026-10-31

**Historical review:** toàn bộ nội dung pre-P0 bên dưới được giữ theo thời điểm chuẩn bị, không sửa bảng pending thành những giá trị giả như đã biết trước khi đo. Trạng thái mới nhất nằm ở mục này, [pilot plan](pilot_plan.md) và [pilot results](../reports/pilot/pilot_results.md).

Target UIT conformance **PASS**, audit đủ 34.997 train + 7.497 validation hashes; fixed counts 34.997/7.497/7.502 không đổi. Ba fresh strict P0 (`p0-strict-e1-v1`, `p0-strict-e1hr-v1`, `p0-strict-e2-v1`) **PASS**, assessed, full epoch; ba P1 (`p1-e1-v1`, `p1-e1hr-v1`, `p1-e2-v1`) **COMPLETE**, mỗi run 3/3 epochs. Các summary ghi zero test-image access. [Evidence nguyên byte](../reports/pilot/evidence/uit_2026-10-31/SHA256SUMS.txt) lưu cả config/runtime/history, không sửa status để promote measurement-only runs.

| Trường từng pending | Đã được resolved bằng execution/review |
|---|---|
| Dataset/environment/paths và full weight hash | Target root/environment đã ghi trong evidence; cùng full pretrained SHA-256 đã pin và dùng ở cả sáu runs; xem report |
| Target conformance | PASS trên UIT; cả hai raw layouts, train/validation hash audit, test_images_opened=0 |
| Batch cho comparable pilot | Common effective batch **2**, microbatch **2**, accumulation **1**, cả ba profiles |
| Strict P0 windows/criteria | Warmup **10** updates, measured **100**, full epoch; max reserved **8 GiB**, min steady-state **6 img/s**, max train epoch **6000 s** trong actual configs; cả ba pass |
| E1-HR candidate | **1344×576 accepted**, first candidate pass; không cần fallback, không chọn bằng classification score |
| Pilot seed/LR/optimizer | **3406**, diagnostic **1e-4**, AdamW betas **[0.9,0.999]**, eps **1e-8**, weight decay **0.05** |
| Bounded P1 | **3 epochs / 52.497 updates** mỗi profile, warmup **1000**, start factor **0.1**, cosine floor **1e-6**; complete, không auto-extend |
| Augmentation | Train-only preview đã review, brightness **0.10**, contrast **0.10** accepted; P1 dùng đúng các biên độ này |
| Runtime/init/checkpoint pilot | FP32 deterministic; exact loader/numerics/init trong evidence; checkpoint tolerance **1e-6**, tie-break **earlier** |

**Pilot-frozen không đồng nghĩa official-training-frozen.** Sau P1, user đã review đủ để đi tiếp, **không yêu cầu P1-v2** và chốt official **max_epochs=8**. Đây là quyết định post-pilot, không phải run 8 epochs đã thực hiện hoặc bằng chứng hội tụ hoàn toàn. **Exact four-LR values và exact official tuning/final seed IDs vẫn NEEDS REVIEW**; không promote `1e-4`/`3406`. Official warmup/cosine horizon/floor, patience/tolerance/tie-break, batch/runtime/optimizer/init adoption và RNG/resume contract còn phải được xác nhận trong official spec; không tự kế thừa chỉ vì có trong pilot config.

Official implementation/search/final seeds/test evaluation **chưa bắt đầu**. Chưa tạo `training_recipe_v1`; report chỉ ghi evidence và các quyết định đã review. Checksum manifest chỉ liệt kê 47 payload files; cả 47 hashes đã được tính lại và kiểm tra PASS, payload giữ nguyên byte.

---

## Historical pre-P0 review (giữ nguyên nội dung tại thời điểm chuẩn bị)

Base preparation đã được review ở commit `82a9614`; task tiếp theo chỉ thêm minimal P0 measurement-only mode và strict evidence guards. Không chạy GPU/Slurm, không official experiment, không thay architecture/preprocessing/split. Master spec vẫn draft. File này không phải `training_recipe_v1`.

**Cập nhật:** minimal `measurement_only` đã implement để first diagnostics không cần fake thresholds. Vẫn cần data root, offline pretrained path/full reviewed SHA và target conformance thật; chưa có GPU run. `criteria` vẫn cần legitimate constraints trước strict acceptance; không promote measurement reports thành pass.

## 1. Trạng thái các quyết định

- **FROZEN BEFORE P0:** các lựa chọn chuẩn bị protocol `cs406-prep0-fp32-v1`, được chọn theo yêu cầu hiện tại, không phải kết quả GPU. Áp dụng common cho E1/E1-HR/E2; không tự trở thành official training recipe.
- **INITIAL P0 ATTEMPT ONLY:** một input cần để bắt đầu đo, không phải maximum feasible batch hoặc final training value. Từng attempt được khai báo trước, không auto-loop.
- **MUST BE DETERMINED BY GPU P0:** measurements và quyết định feasibility/final comparable batch cần evidence UIT.
- **DEFER UNTIL P1 / OFFICIAL SEARCH:** không điền full P1 hoặc official LR/seeds/horizon ở task này.

Original P0 templates thêm `feasibility.mode=null` để bắt buộc chọn explicit, không default; P1 templates giữ nguyên. Chín file [reviewed/](../configs/pilot/reviewed/README.md) vẫn là **`.example.json`**, chưa server-ready. Sáu exploratory examples dùng `measurement_only`, ba comparable-final dùng `criteria`; mọi thresholds vẫn null tới review phù hợp. Không có report hoặc measurement GPU mới.

## 2. Review từng field pending

Đã kiểm mọi leaf `null` của cả bảy JSON templates gốc: 58 paths ở preparation ban đầu, thêm `feasibility.mode` thành 59. `P0/P1` chỉ các profile tương ứng; `N/A` là field không được dùng bởi policy đã chọn, không phải giá trị server còn thiếu. Mỗi field chưa chắc chắn có hàng riêng.

| Field | Current status | Freeze before P0? | Proposed value / rule | Reason | Needs GPU evidence? |
|---|---|---|---|---|---|
| `protocol_id` | null P0/P1 | Có, P0 | `cs406-prep0-fp32-v1-exploratory-a`, `...-exploratory-b`, `...-comparable-final`; P1 ID sau P0 | Phân biệt phase/batch contract | Không |
| `seed` | null P0/P1 | Có, pilot only | **3406**, chung E1/E1-HR/E2 | Deterministic, khác split seed 406; không seed-search | Không |
| `paths.data_root` | null cả 7 files | Bắt buộc trước command | Absolute root chứa label folders; CLI/env được override | Không sửa manifest paths | Cần server path, không GPU measurement |
| `paths.weights` | null P0/P1 | Bắt buộc trước P0/P1 | File thật `convnext_tiny-983f1562.pth`, path từ CLI/env/config | Offline exact checkpoint | Cần file, không GPU |
| `expected_weight_sha256` | null P0/P1 | Bắt buộc trước P0/P1 | Hash file thật → user review → cùng full 64-char digest vào mọi P0/P1 config | Không invent hash; không prefix-only gate | Không GPU; cần file thật/review |
| `initialization.method` | null P0/P1 | Có, pilot | `linear_default` | Framework-consistent, không tune distribution | Không |
| `initialization.seed` | null P0/P1 | Có, pilot | **3406**, trước init từng new head | Cùng rule/seed, khác fan-in vẫn đúng architecture | Không |
| `initialization.std` | null P0/P1 | N/A | Giữ null: chỉ dùng khi `trunc_normal` | `linear_default` không đọc field này | Không |
| `initialization.bias` | null P0/P1 | N/A numeric field | Giữ null; bias dùng rule Linear mặc định được ghi bên dưới | Không mặc định zero bias ngoài rule đã chọn | Không |
| `optimizer.betas` | null P0/P1 | Có, common pilot | `[0.9, 0.999]` | AdamW standard defaults; không thêm search dimension | Không |
| `optimizer.eps` | null P0/P1 | Có, common pilot | `1e-8` | AdamW standard epsilon | Không |
| `optimizer.weight_decay` | 0.05 đã chốt | Giữ nguyên | Conv/Linear weights 0.05, bias/LN/layer-scale no decay theo implementation hiện có | Không đổi recipe | Không |
| `optimizer.lr` — P0 | null | Initial attempt | `1e-4`, chung, constant LR của P0 | Một scale nhỏ để forward/backward; không validation tuning, không claim optimal | Stability cần P0 |
| `optimizer.lr` — P1 | null | Chưa | Freeze sau P0 stability review; không tự kế thừa nếu có numerical issue | Bounded convergence pilot riêng | Cần P0 stability |
| `augmentation.enabled` — P0 | null | Có | `false` | Timing/VRAM không gồm photometric overhead; không đo model quality | Không |
| `augmentation.brightness` — P0 | null | Có | `0.0`, inactive | Explicit no-jitter; không đổi frozen transform | Không |
| `augmentation.contrast` — P0 | null | Có | `0.0`, inactive | Explicit no-jitter | Không |
| `augmentation.enabled` — P1 | null | Rule đã thống nhất; config sau P0 | `true`, shared brightness/contrast draw trên ảnh gốc | P1 phải kiểm actual reviewed training behavior | Không cần GPU để chọn rule |
| `augmentation.brightness` — P1 | null | Defer numeric | Mild amplitude còn chờ train-only visual review trước P1 | Không chọn bằng classification score | Không; cần review preview |
| `augmentation.contrast` — P1 | null | Defer numeric | Mild amplitude còn chờ train-only visual review trước P1 | Giữ cues màu/contrast; không augmentation search | Không; cần review preview |
| `clipping.enabled` | null P0/P1 | Có, common pilot | `false` | Giữ diagnostics rõ, không thêm norm hyperparameter; nonfinite loss/grad vẫn fail | Không cần để chọn; stability được đo |
| `clipping.max_norm` | null P0/P1 | N/A | Giữ null khi disabled | Không invent threshold không được dùng | Không |
| `runtime.amp.enabled` | null P0/P1 | Có, first pilot protocol | `false` | Không giả định BF16/FP16 capability; debug ở FP32 | Không |
| `runtime.amp.dtype` | null P0/P1 | Có | `float32` với autocast disabled | Explicit arithmetic policy | Không |
| `runtime.scaler.enabled` | null P0/P1 | Có | `false` | FP32 không gradient scaling | Không |
| `runtime.scaler.init_scale` | null P0/P1 | N/A | null | Scaler disabled | Không |
| `runtime.scaler.growth_factor` | null P0/P1 | N/A | null | Scaler disabled | Không |
| `runtime.scaler.backoff_factor` | null P0/P1 | N/A | null | Scaler disabled | Không |
| `runtime.scaler.growth_interval` | null P0/P1 | N/A | null | Scaler disabled | Không |
| `runtime.tf32_matmul` | null P0/P1 | Có | `false` | Không dùng reduced-precision matmul trong first protocol | Không |
| `runtime.tf32_cudnn` | null P0/P1 | Có | `false` | Cùng convolution arithmetic policy | Không |
| `runtime.matmul_precision` | null P0/P1 | Có | `highest` | Nhất quán với TF32 off trong pinned torch | Không |
| `runtime.cudnn_benchmark` | null P0/P1 | Có | `false` | Tránh benchmark-based kernel choice trong first debug protocol | Không |
| `runtime.cudnn_deterministic` | null P0/P1 | Có | `true` | Prefer deterministic convolution algorithms | Target hỗ trợ phải được xác minh |
| `runtime.deterministic_algorithms` | null P0/P1 | Có | `true` | Fail rõ nếu operation không hỗ trợ deterministic path | Target execution phải xác minh |
| `runtime.deterministic_warn_only` | null P0/P1 | Có | `false` | Không âm thầm tiếp tục sau nondeterministic warning | Không |
| `runtime.torch_compile` | null P0/P1 | Có | `false` | Loại compile startup/cache khỏi first pilot debugging | Không |
| `runtime.num_workers` | null P0/P1 | Có, initial common policy | `2` | Conservative worker count; không dùng hết 16 CPUs | Bottleneck cần đo; không claim optimal |
| `runtime.pin_memory` | null P0/P1 | Có | `true` | Common CUDA transfer policy | Tác động throughput cần đo |
| `runtime.persistent_workers` | null P0/P1 | Có | `true` | Giữ worker lifecycle nhất quán trong run | Startup/epoch timing cần đo |
| `runtime.prefetch_factor` | null P0/P1 | Có | `2` | Bounded queue: 2 workers × 2 prefetched batches | RAM/I/O cần kiểm, không tuning grid |
| `runtime.cpu_threads` | null P0/P1 | Có, initial common policy | `2`; allocation tối thiểu 4 CPUs cho main + workers | Hạn chế oversubscription, không tự request toàn bộ CPU/RAM | Cần confirm allocation, không benchmark để chọn |
| `gates.conformance_report` | null P0/P1 | Bắt buộc trước P0 | Actual passing UIT report với full train/val hash audit | Local report không thay target root/environment | Cần target conformance, không classification |
| `budget.warmup_updates` | null P0 | Initial bounded windows | Exploratory `5`; comparable proposal `10` | GPU benchmark warmup, không LR warmup | Độ ổn định window phải kiểm |
| `budget.measure_updates` | null P0 | Initial bounded windows | Exploratory `20`; comparable proposal `100` | Finite diagnostic window, không train tới score mong muốn | Measurement stability cần kiểm |
| `budget.full_epoch` | null P0 | Có theo phase | Exploratory `false`; comparable-final **true** | Detect OOM nhanh; final evidence cần actual full epoch | Full-epoch time cần đo |
| `batch.microbatch` | null P0/P1 | Initial only | A: `2`; chỉ nếu OOM thì B: `1` | Không tuyên bố max feasible batch | Có |
| `batch.accumulation` | null P0/P1 | Initial only | A: `1`; B: `2` | Giữ exploratory effective batch 2 khi giảm microbatch | Final value cần P0/review |
| Effective original-image batch | Derived, chưa final | Initial only | A/B đều `2`; comparable-final còn pending | Không tính 5 E2 views thành 5 samples | Có + common-budget review |
| `feasibility.mode` | null trong ba P0 templates; explicit trong examples | Có, theo phase | Exploratory `measurement_only`; comparable-final `criteria`; P1 không thêm mode | Tách first diagnostics khỏi acceptance; không implicit default | Không |
| `feasibility.max_peak_reserved_gib` | null P0 | Freeze rule, chưa numeric | Proposal: `0.90 × actual logical-device total bytes / 2^30` | 10% engineering safety margin, không scientific/optimal threshold | Cần actual MIG capacity metadata |
| `feasibility.min_images_per_second` | null P0 | Chưa | Derive từ approved epoch/job budget; không lấy 0/epsilon để pass | Chưa có deadline/throughput target hợp lệ | Cần budget + measurements để review |
| `feasibility.max_epoch_seconds` | null P0 | Chưa | Derive job walltime minus overhead/reserve và epochs/job | Không suy từ arbitrary 72h example thành threshold project | Cần actual constraints + epoch timing |
| `global_wh` — E1-HR | null P0/P1 | Candidate only | First candidate `[1344,576]`; không final | Đã có proposal, chưa có feasibility evidence | Có |
| `resolution_policy.ordered_candidates` | `[[1344,576]]` | Giữ first-only policy | Không thêm numeric fallback; microbatch B trước khi review resolution mới | Tránh biến thành resolution search | Fallback cần evidence + user review |
| `resolution_policy.criteria_id` | null P0/P1 | Có cho declared attempt | `cs406-prep0-hr-v1-<phase>`; thresholds vẫn pending | Không tái dùng criteria ID qua protocol khác | Không |
| `resolution_policy.prior_failure_reports` | `[]` | Có cho first-only | `[]`; không carry A OOM sang B làm resolution evidence | Candidate chưa đổi; mỗi attempt có protocol riêng | Không |
| `resolution_policy.batch_policy.effective_batch` | null HR | Initial only | Exploratory `2`; comparable null tới review | Không gọi là final training batch | Có cho final |
| `resolution_policy.batch_policy.candidate_batches[0].microbatch` | null HR | Initial only | A `2`, B `1`; phải khớp actual batch | Không dùng oversized batch ngoài reviewed policy | Có |
| `resolution_policy.batch_policy.candidate_batches[0].accumulation` | null HR | Initial only | A `1`, B `2`; future comparable theo B_final/microbatch | Ràng buộc compatible batch evidence | Có |
| `gates.p0_summaries` | null P1 | Sau P0 | Ba actual comparable passing full-epoch summaries | Exploratory/measurement-only không đủ P1 gate | Có |
| `budget.max_epochs` | null P1 | Sau P0, trước P1 | Bounded pilot horizon/attempt count từ approved budget | Không invent convergence hoặc official horizon | Có timing + budget review |
| `scheduler.warmup_updates` | null P1 | Sau P0 | Freeze cùng policy trước bounded P1 | Phụ thuộc updates/epoch và chosen horizon | Có P0 timing/batch |
| `scheduler.start_factor` | null P1 | Defer P1 | Một common explicit value, review cùng warmup | Không cần cho P0 constant-LR runner | Không cần GPU để chọn số, nhưng defer cùng recipe |
| `scheduler.min_lr` | null P1 | Defer P1 | Review common floor khi LR/horizon đã chọn | Không dùng P0 LR thành official cosine floor | Cần recipe review, không score-tuning |
| `checkpoint.tolerance` | null P1 | Defer P1 | Prefer very small tolerance; proposal `1e-6` trên [0,1], chưa freeze | Không cần diagnostic checkpoint trong P0 | Không cần GPU để chọn; review trước P1 |
| `checkpoint.tie_break` | null P1 | Defer P1 | Proposal `earlier`, chưa freeze | Xác định trước P1; không cherry-pick epoch | Không |
| Early-stopping patience | Master pending, chưa có JSON value | Defer | Official patience sau bounded P1 curves; P1 không auto-extend | Không giả convergence | Có P1 curves |
| Final official four-LR grid | Master pending | Không | Giữ đúng 4 values sẽ review sau pilot; chưa điền | P0 `1e-4` không phải hidden LR trial search | Cần pilot stability, không iterative validation refine |
| Official tuning seed ID | Master pending | Không | Chưa chọn | Pilot 3406 không tự thành official tuning seed | Không; defer protocol review |
| Three official final seed IDs | Master pending | Không | Chưa chọn | Giữ 3-seed protocol; không chọn bằng score | Không; defer protocol review |

Các field đã có giá trị như `purpose`, `stage`, `experiment`, artifact/preprocessing paths, `paths.runs_dir`, `runtime.device`, E1/E2 global WH và conformance sample/thread settings giữ nguyên, trừ các override không user-specific trong examples. Không field null nào bị bỏ qua trong review; null inactive khác null bắt buộc.

## 3. Protocol rationale và giới hạn

**Seed:** pilot training seed = classifier seed = **3406**, split seed vẫn **406**, membership không thay đổi. Head initialization dùng isolated CPU RNG, seed ngay trước initialize rồi restore outer CPU RNG. Weight rule `kaiming_uniform_(a=sqrt(5))`; bias uniform `[-1/sqrt(fan_in), +1/sqrt(fan_in)]`. E1/E1-HR fan-in 768, E2 1536; cùng rule không có nghĩa cùng-shaped weights. Exact rule/version được harness log. Không tune initialization.

**Optimizer:** chỉ chọn standard betas/eps, giữ decay groups đã review. Các defaults này có nguồn [PyTorch 2.8 AdamW](https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html). LR `1e-4` là một starting diagnostic scale chung, không phải LR tối ưu hoặc một official grid. Nonfinite/divergence thì stop/report và review scale, không tự refine theo validation Macro-F1.

**Numerics:** first protocol dùng FP32, AMP/GradScaler/TF32/compile off. Đây là lựa chọn bảo thủ không cần giả định GPU hỗ trợ BF16. Đổi sang AMP sau này phải review common policy và chạy lại comparable P0; FP32 measurements không dự đoán VRAM/throughput của AMP. Không kết luận FP32 tốt hơn AMP. [PyTorch AMP guide](https://docs.pytorch.org/docs/2.8/notes/amp_examples.html) giải thích riêng autocast và scaling.

**Determinism:** trước CUDA process, common environment policy là `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=3406`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`. Runtime đã record các env variables này; JSON không tự export chúng. Strict deterministic failure phải report, không tự downgrade riêng một architecture. Seed không bảo đảm bitwise equality khác GPU/library/platform. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html).

**Augmentation:** P0 disable jitter để đo đúng scope no-photometric-overhead, giữ UI/resize/view construction đầy đủ. Main/P1 train jitter vẫn phải bật theo recipe đã review. P0 numbers không phải timing đầy đủ của P1 augmentation/validation/checkpoint. Trước P1 phải review amplitudes và đo/log actual P1 epoch behavior; không cần thêm augmentation comparison.

**Loader:** 2 workers, 2 main CPU threads, pin/persistent on, prefetch 2 là initial common policy dưới capacity tối đa đã biết. Confirm allocation ≥4 CPUs; RAM và package/CUDA policy vẫn cần server review. Không claim loader optimized; nếu bottleneck, report để review một common revised policy, không search nhiều worker counts.

## 4. Manual attempts và chuyển sang comparable-final

Examples có nhãn **STARTING P0 ATTEMPT — not frozen training recipe** trong README; không run-ready khi còn mandatory nulls.

| Phase | E1 | E1-HR | E2 | Budget |
|---|---|---|---|---|
| Exploratory A | microbatch 2, accumulation 1 | candidate 1344×576; 2/1 | 2/1 | warmup 5 + measure 20; full_epoch=false |
| Exploratory B, chỉ khi A GPU OOM | 1/2 | Cùng candidate 1344×576; 1/2 | 1/2 | Cùng 5 + 20; full_epoch=false |
| Comparable-final, sau evidence/review | Feasible microbatch + reviewed accumulation | Same candidate được xác nhận + reviewed batch table | Feasible microbatch + reviewed accumulation | Proposal 10 + 100; **full_epoch=true** |

Mỗi attempt config/run riêng, giữ failure logs. A pass thì không chạy B để tìm score. Lỗi data/hash/package/determinism/numerical không phải lý do giảm batch hay resolution; sửa/report đúng nguyên nhân. B vẫn OOM thì stop và `NEEDS USER REVIEW`, không tự thêm candidate thấp hơn. Không khám phá maximum batch bằng upward search ở task này; nếu batch 2 pass, chỉ biết 2 feasible trong window đã đo.

Đổi microbatch A→B ở HR là **batch attempt mới**, với `protocol_id`/`criteria_id` và batch contract riêng; không dùng failure A để reject resolution khi chuyển B. Hai examples chỉ có một candidate và `prior_failure_reports=[]`. Nếu cần resolution fallback sau này, user phải review toàn ordered list + candidate batch table + criteria trước một selection protocol mới; historical contexts khác nhau không được reuse. Chưa propose numeric fallback vì chưa biết candidate đầu có fail ở microbatch 1 không.

Sau exploratory, chọn feasible microbatch `m_i`, rồi user review một `B_final` chung chia hết cho tất cả `m_i`; accumulation `a_i=B_final/m_i`. `LCM(m_i)×k` là cách tạo các lựa chọn hợp lệ, **không auto-chọn k hoặc final batch**. Không gọi B=2 exploratory là official effective batch. Kiểm input windows thỏa `warmup+measure <= ceil(ceil(34997/m_i)/a_i)`; nếu không, review một common finite window trước comparable runs.

Tạo **ba fresh comparable-final runs** với common B_final, cùng numerical/runtime/optimizer/seed/init policy, actual constraints, full epoch và environment. Không warm-start từ exploratory models, không chỉnh status/result files thủ công. Nếu đổi runtime hoặc threshold contract, rerun affected measurements; không dùng old failure để reject resolution dưới protocol mới.

## 5. Circular dependency đã được xử lý bằng minimal mode

Harness cũ bắt buộc thresholds ở mọi P0, gây circular dependency khi chưa có approved throughput/job budget. Patch hiện tại yêu cầu `feasibility.mode` explicit và chỉ bỏ numeric threshold requirements trong `measurement_only`; không bỏ full SHA, artifacts/conformance, seed/init/optimizer/runtime, finite windows, batch/HR policy, no-test/no-search gates.

Batch policy không buộc biết max feasible batch trước measurement nếu khai báo **initial attempt** như A/B ở mục 4. Giới hạn hiện tại là mỗi candidate chỉ có một batch row trong một resolution-selection protocol; vì vậy batch retry dùng protocol riêng, không carry OOM sang resolution rejection. Chưa cần thêm batch-search code hoặc final effective batch để chuẩn bị first attempt.

| Mode | Requirements và result |
|---|---|
| `measurement_only` | Thresholds có thể null; finite workload vẫn đo/sync/reset memory như trước. Success: `measured`, `not_assessed`, `fallback_eligible=false`. OOM: fail/oom; infrastructure/numerical: fail/infrastructure_or_numerical. Mọi measurement-only outcome đều ineligible cho HR fallback, P1 hoặc recipe |
| `criteria` | Required positive VRAM/throughput thresholds; positive epoch threshold khi full_epoch=true. Giữ actual pass/fail comparisons. Genuine strict criteria/OOM failures chỉ eligible HR khi verified matching context/batch policy; pass không phải fallback failure |

Mode được bind vào `feasibility_context` version 2 (explicit `feasibility_mode` và feasibility dict) và fingerprint. HR validator từ chối measured status, noncriteria/missing/forged mode, ineligible hoặc unmatched context; không dùng measurement-only OOM để reject resolution. Reports cũ thiếu mode evidence phải được đo lại theo strict protocol, không sửa JSON để promote.

P1 gate và recipe export chỉ nhận P0 **criteria + pass + full epoch**, với mode/fingerprint được verify và các identity/common-batch/runtime checks giữ nguyên. Collector vẫn giữ pass/fail/measured/complete/interrupted records để diagnostic reporting; không derive thresholds hoặc tự freeze recipe.

Workflow: **UIT conformance → measurement-only exploratory P0 → review actual VRAM/throughput/epoch constraints → freeze legitimate thresholds → fresh strict comparable P0 E1/E1-HR/E2 → strict passing full-epoch summaries mới được gate P1**. Một declared config vẫn chỉ chạy một workload; không thêm LR/batch/resolution/threshold search hoặc P1 implementation.

## 6. Cách derive thresholds mà không invent measurement

| Threshold | Rule đề xuất | Trạng thái hiện tại |
|---|---|---|
| Peak reserved VRAM | `0.90 × total_memory_bytes / 2^30` của logical MIG device thực nhận; cùng capacity cho comparable profiles | 10% margin là engineering proposal, không số đo; numeric GiB còn null |
| Max train-epoch seconds | Từ approved job walltime W, startup/warmup/checkpoint/validation allowances và số completed epochs/job E: phần train budget mỗi epoch | Cần user/job-policy review; chưa có hard project deadline |
| Min steady-state original-images/s | Lower bound từ `34997 / approved_train_epoch_seconds`, kiểm riêng full-epoch overhead | Không dùng measured score để chọn; numeric còn null |

Ví dụ 72h = 259200s chỉ là walltime minh họa cần confirm actual partition/job allocation; **không dùng ngay 259200 làm epoch threshold**. Một full train epoch gần 72h sẽ không để chỗ startup/validation/resume/checkpoint. Các allowances phải có engineering rationale và phase scope; P0 full-epoch timer không gồm P1 validation/checkpoint I/O.

Nếu chọn measurement-only để có số liệu đầu, sau đó review thresholds/common batch **trước comparable-final**, giữ cùng criteria giữa ba profiles và rerun full epochs. Không chọn threshold sau khi xem score hoặc sửa old report để biến nó thành pass. VRAM safety policy từ capacity có thể review trước P0; throughput/epoch policy cần actual job plan thay vì một số tùy ý.

## 7. P1 / official fields freeze sau P0

P1 template không được điền ở task này. Chốt bounded max_epochs/attempt count, warmup length/start factor/cosine floor, numeric checkpoint tolerance/tie-break, reviewed jitter amplitudes và actual LR sau P0 review, trước P1. Patience và official horizon cần bounded P1 learning curves, không throughput-only pilot. Không đồng nhất P1 bounded budget với final training horizon.

Official search vẫn 4 LR ×1 tuning seed/architecture, rồi selected LR ×3 final seeds. Chưa chọn four-LR values, tuning seed hoặc final seed IDs. Pilot **3406 != split 406**, và **3406 không tự động thuộc official seed set**. Không test/validation-score-driven config selection ở task này; không E3 implementation.

## 8. Exact pretrained hash và readiness

Luồng bắt buộc: **file exists → sha256sum → user reviews digest → copy cùng 64-char full digest vào mọi P0/P1 runtime config**. Được thực hiện trên machine có file thật; không cần GPU để hash. Ví dụ hai command kiểm file, không auto-accept digest:

```bash
test -f "$CS406_WEIGHTS_PATH"
sha256sum "$CS406_WEIGHTS_PATH"
```

Tất cả templates/examples vẫn `expected_weight_sha256=null`. Không tải weights hoặc invent digest ở task này. Dataset root, weight path và UIT conformance report cũng còn null tới deployment. Conformance chạy được độc lập với weights sau khi có target dataset/environment; không cần sửa examples để run conformance.

**Còn cần user review/đầu vào thật:**

1. First exploratory dùng explicit measurement-only, không còn numeric-threshold blocker. Legitimate thresholds vẫn phải được review/freeze sau first measurements và trước fresh strict comparable runs.
2. Xác nhận actual logical GPU capacity, CUDA/package lock, CPU allocation ≥4, RAM/walltime envelope và target conformance report; numeric VRAM threshold theo capacity rule cần review.
3. Lấy file pretrained thật, review full hash và pin cùng digest vào mọi runtime config; resolve dataset/weight/report paths.
4. Xác nhận initial A→B batch/window policy trước submit. Không review numeric resolution fallback ở task này; chỉ candidate 1344×576.
5. Sau P0 mới review B_final/selected microbatches/HR feasibility và thresholds trước ba comparable runs; sau đó freeze bounded P1 fields. Không coi prepare examples là permission chạy job.

Không có architecture/training-framework changes, không GPU/Slurm/server execution, không official experiment, không classification-score tuning hoặc test access trong task chuẩn bị này.

## 9. Preparation checks

Preparation ở commit 82a9614 đã kiểm đủ 58 paths; mode patch thêm path thứ 59. Exploratory readiness check vẫn phải fail trên missing real paths/full hash/conformance report, nhưng không còn fail vì thresholds null. Comparable-final vẫn fail khi thiếu batch/thresholds. CPU regression tests kiểm missing/invalid mode, criteria constraints, measured/OOM results, mode/fingerprint forgery, HR/P1/recipe guards và existing no-test/full-SHA checks.

Mode patch chỉ thay P0 validation/assessment và shared evidence checks/collector, cùng P0 config mode fields/tài liệu/tests. P1 templates/runner, model architectures, frozen preprocessing/split và official protocol không thay đổi. Không có actual UIT conformance/P0/P1 hoặc GPU measurement mới.
