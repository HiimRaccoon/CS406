# Pilot configuration templates

Các JSON này chỉ dành cho **pilot**. Giá trị `null` là quyết định còn pending, không phải default. P0/P1 sẽ fail trước khi dựng model nếu thiếu trường bắt buộc. Không dùng chúng làm official experiment configs.

- `conformance.json`: train-only numerical checks; CLI/env phải cung cấp dataset root.
- `p0_e1.json`, `p0_e1hr.json`, `p0_e2.json`: một configuration mỗi lần chạy, benchmark warmup + bounded measurement + optional full epoch.
- `p1_e1.json`, `p1_e1hr.json`, `p1_e2.json`: finite max epochs, warmup/cosine và full validation mỗi epoch. Không auto-extend hoặc chạy LR grid.

Copy template sang `configs/pilot/local/` (ignored), review/điền numeric values trước khi chạy. Giữ nguyên `purpose`, experiment profile, preprocessing path, `.05` weight decay và E1/E2 global resolution.

Các nhóm phải khai báo:

| Nhóm | Contract |
|---|---|
| `protocol_id`, `seed` | Predetermined pilot protocol và training seed; không tự lấy split seed 406 |
| `paths` | Dataset root chứa trực tiếp thư mục labels; offline pretrained file; output root |
| `expected_weight_sha256` | **Mandatory trước P0/P1**: full 64 hex characters của file thật đã lấy/hash/review. Normalize whitespace/case thành lowercase; null, prefix hoặc invalid digest bị từ chối trước model build. Conformance không cần weight |
| `initialization` | `linear_default` hoặc `trunc_normal`; exact seed. Trunc-normal cần `std`, `bias`; trường không áp dụng có thể để null |
| `optimizer` | AdamW LR, hai betas, eps; decay `.05` cho Conv/Linear weights, bias/LN/layer-scale không decay |
| `augmentation` | Explicit enabled; nếu bật cần brightness/contrast amplitudes `[0,1]`; brightness rồi contrast, một draw mỗi original image/epoch |
| `clipping` | Explicit enabled; nếu bật cần max norm |
| `runtime` | Mọi boolean/dtype explicit; workers >0 cần prefetch factor. AMP enabled chỉ float16/bfloat16; scaler enabled cần toàn bộ scaler fields |
| P0 `budget` | Benchmark warmup updates, measured updates, full-epoch flag. Hai windows nối tiếp cùng iterator, không cycle; tổng updates phải vừa một train epoch |
| P0 `feasibility` | Reviewed peak-reserved GiB, min original-images/s, max full-epoch seconds; mọi threshold được dùng phải >0 |
| P1 `budget` | Predetermined finite max epochs, cùng budget cho ba profiles; không dừng vì đạt score mong muốn |
| P1 `scheduler` | Warmup update count, start factor, cosine min LR; horizon = updates/epoch × max epochs |
| P1 `checkpoint` | Supported validation Macro-F1 tolerance và tie-break `earlier` hoặc `nll` |
| `gates` | Passing target-environment conformance report có train+val hash audit; P1 cần ba passing full-epoch P0 summaries |

`linear_default` được ghi bằng exact PyTorch rule/version trong `model_metadata.json`: `kaiming_uniform_(a=sqrt(5))` cho weight và uniform bias trong `±1/sqrt(fan_in)`. Cùng rule/seed cho ba profiles; E2 fan-in khác theo architecture. RNG classifier được cô lập, seed ngay trước initialize rồi restore outer state. Đây là lựa chọn phải review, chưa mặc định chọn thay người dùng.

P1 scheduler theo optimizer updates: với update index `u=0..T-1`, warmup `W`, base LR `b`, start factor `s`, floor `f`:

```text
u < W:  b * (s + (1-s) * u/W)
u >= W: f + (b-f)/2 * (1 + cos(pi * (u-W)/(T-W-1)))
```

Nếu cosine chỉ có một update, dùng floor. Phải `0 <= W < T`; resume giữ nguyên `T`. Last incomplete accumulation group chia loss theo đúng số original images thực có. LR không scale thêm theo số views.

E1-HR `global_wh` chưa chọn sẵn. `[1344,576]` trong `ordered_candidates` chỉ ghi first candidate đã thảo luận. Trước P0 phải review ordered list, `criteria_id`, thresholds, aspect-ratio suitability và `resolution_policy.batch_policy`. Chỉ chạy first candidate; nếu pass thì freeze. Task này không thêm fallback numeric values.

`batch_policy.effective_batch` là common original-image batch; `batch_policy.candidate_batches` là bảng `{microbatch, accumulation}` theo đúng thứ tự `ordered_candidates`, được review **trước candidate đầu tiên**. Mỗi hàng phải có cùng effective batch và actual batch phải khớp hàng của candidate đang chạy. Có thể dùng microbatch khác nhau giữa candidates khi đã khai báo/review trước; không dùng OOM từ oversized/unapproved microbatch để reject resolution.

Fallback cần genuine OOM/criteria failures của **tất cả** candidate đứng trước, cùng `protocol_id`, ordered list/criteria ID, thresholds, seed, runtime, initialization, dataset/split/preprocessing, full pretrained SHA và reviewed batch policy. `feasibility_context.json` được ghi sau khi verify artifacts/weight/preprocessing và trước model allocation; cả criteria và OOM summaries chứa full context + fingerprint và actual batch/effective batch. Code so sánh context/fingerprint và kiểm actual predecessor batch với bảng đã review. Failure trước khi verify identity không có bound context, không được dùng làm fallback. Không chấp nhận lỗi file/package/numerical, không tự loop resolution hoặc dùng score.

P0 passing summaries dùng làm P1 gate phải cùng runtime, seed, initialization, common effective batch, weights/data/preprocessing/source/environment. Own profile phải giữ resolution/microbatch. P0 exploratory microbatch failures vẫn lưu riêng; khi chốt common effective batch, tạo ba passing P0 measurements tương ứng trước P1.

Không dùng validation Macro-F1 để refine LR grid trong pilot. Pilot LR chỉ sanity-check scale bằng divergence/NaN/exploding/no-learning; ghi mọi lần thử. Sau review mới freeze đúng 4 LR cho official search; harness không chạy official search.

Xem [server runbook](../../docs/server_pilot_runbook.md) để setup, điền configs và chạy đúng thứ tự.
