# UIT pilot harness — server runbook

Code chỉ phục vụ preprocessing conformance, P0 feasibility/timing và bounded P1. **Không phải official E1 implementation**; không có E3, LR-search launcher, final experiment hay test evaluator. Master spec giữ trạng thái **DRAFT FOR USER REVIEW — chưa frozen, chưa authorize implementation** đối với official experiments. Yêu cầu implementation pilot mới nhất chỉ cho phép chuẩn bị harness; numeric choices và việc submit job cần được người dùng review trước.

## 1. Source và server layout

Lấy toàn bộ Git repo để giữ source commit và relative paths. [Upload checklist](../SERVER_UPLOAD_CHECKLIST.md) liệt kê exact sources và những gì phải cung cấp ngoài Git. Reports visualization cũ không cần dùng làm model input.

```text
/datastore/binhvp/<my-folder>/CS406/
├── repo/                  # Git source, frozen artifacts/configs
├── data/geoguessr/         # raw dataset: Aland/, Argentina/, ...
├── pretrained/convnext_tiny-983f1562.pth
├── runs/pilot/             # separate folder per P0/P1 run
├── reports/pilot/          # conformance + collected diagnostics
├── wheelhouse/             # optional offline dependency wheels
└── env/                    # Python 3.11 venv
```

Thay `<my-folder>` bằng workspace thực tế. Chạy setup trên node được UIT cho phép; model/GPU work chỉ trong Slurm allocation. Không giả định login node có CUDA hoặc được phép chạy heavy workload.

```bash
export CS406_HOME="/datastore/binhvp/<my-folder>/CS406"
mkdir -p "$CS406_HOME"
cd "$CS406_HOME"
git clone https://github.com/HiimRaccoon/CS406.git repo
cd repo
# Nếu đã clone: git pull --ff-only
git rev-parse HEAD
export CS406_REPO="$CS406_HOME/repo"
export CS406_DATA_ROOT="$CS406_HOME/data/geoguessr"
export CS406_WEIGHTS_PATH="$CS406_HOME/pretrained/convnext_tiny-983f1562.pth"
export CS406_RUNS_DIR="$CS406_HOME/runs/pilot"
mkdir -p logs configs/pilot/local "$CS406_RUNS_DIR" "$CS406_HOME/reports/pilot"
```

Không sửa đường dẫn/membership trong manifest. Ví dụ dòng `Aland/canvas_1629480180.jpg` phải resolve thành `$CS406_DATA_ROOT/Aland/canvas_1629480180.jpg`. Mọi CLI chạy từ repo root. Các biến `CS406_*` cần export lại ở shell mới hoặc khai báo trong environment file riêng ngoài Git, không chứa secrets trong config tracked.

## 2. Python và dependencies

Python **3.11**, torch **2.8.0**, torchvision **0.23.0**, Pillow **11.3.0**, numpy **2.1.3**, pandas **2.2.3**, scikit-learn **1.6.1**, pytest **8.3.5**. Không dùng YAML/tqdm/WandB. Dùng venv làm cách chính:

```bash
python3.11 -m venv "$CS406_HOME/env"
source "$CS406_HOME/env/bin/activate"
export CS406_PYTHON="$CS406_HOME/env/bin/python"
python --version
```

UIT driver/CUDA availability quyết định wheel index; kiểm tra `nvidia-smi` trong GPU allocation và đối chiếu [official PyTorch 2.8 installation matrix](https://pytorch.org/get-started/previous-versions/#v280). **Không tự thay torch/torchvision versions hoặc lấy CUDA wheel ngẫu nhiên.** Chọn một CUDA wheel index tương thích với driver đã review, rồi:

```bash
# Đặt CS406_TORCH_INDEX thành index CUDA đã review, không phải CPU index cho P0/P1.
: "${CS406_TORCH_INDEX:?Set reviewed official CUDA wheel index}"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url "$CS406_TORCH_INDEX"
python -m pip install -r requirements.txt
python -m pip check
python -m pip freeze > "$CS406_HOME/environment-lock.txt"
```

Nếu server offline, trên máy Linux cùng Python/architecture có Internet chuẩn bị wheelhouse từ **cùng** index đã review, ví dụ:

```bash
python3.11 -m pip download -d wheelhouse torch==2.8.0 torchvision==0.23.0 --index-url "$CS406_TORCH_INDEX"
python3.11 -m pip download -d wheelhouse -r requirements.txt --find-links wheelhouse
# Chuyển wheelhouse sang server bằng phương thức UIT cho phép; không commit binaries.
```

Sau đó trên server:

```bash
python -m pip install --no-index --find-links "$CS406_HOME/wheelhouse" -r requirements.txt
python -m pip check
python -m pip freeze > "$CS406_HOME/environment-lock.txt"
```

Không dùng wheel Windows từ workspace local cho UIT Linux. Runtime ghi exact package versions/CUDA/cuDNN/driver/GPU và allocation vào mỗi run. Không đổi packages, source hoặc runtime giữa comparable P0/P1 runs; nếu đổi phải chạy lại conformance và affected gates. Module CUDA cụ thể tùy UIT, không hard-code tên module chưa xác minh.

## 3. Dataset, offline pretrained weights và CPU checks

Chuẩn bị raw dataset bằng cách giữ nguyên file bytes. Harness đọc manifest cũ, validate bốn full hashes/provenance/counts/support; test records chỉ là metadata. Không có API load test image.

Download weight trên máy có Internet từ official URL rồi chuyển file sang server nếu cần:

```bash
mkdir -p "$CS406_HOME/pretrained"
curl --fail --location https://download.pytorch.org/models/convnext_tiny-983f1562.pth \
  --output "$CS406_WEIGHTS_PATH"
sha256sum "$CS406_WEIGHTS_PATH"
```

File phải tên đúng `convnext_tiny-983f1562.pth`. Sau khi lấy file thật, tính/review **full SHA-256 thực tế** rồi **bắt buộc điền `expected_weight_sha256` trong mọi P0/P1 config trước GPU chạy**. Hash phải đủ 64 hex characters, được normalize thành lowercase. Config check fail nếu thiếu/null/invalid hash; `build_model` yêu cầu actual full digest bằng expected full digest, không dựa vào prefix `983f1562` làm integrity gate. Templates vẫn null cho tới khi hash thật được review; không có full hash giả điền sẵn. Conformance không cần pretrained weight. Model strict-load ImageNet state dictionary, không tự download và không dùng `weights.transforms()`. E1/E1-HR/E2 build độc lập từ cùng file, drop-path max `.1` theo explicit TorchVision block-linear schedule, cùng recorded initialization rule.

```bash
cd "$CS406_REPO"
python -m pytest -q
python -m src.pilot.dataset
python -m src.pilot.conformance --config configs/pilot/conformance.json \
  --data-root "$CS406_DATA_ROOT" --run-id uit-conformance-v1 \
  --verify-all-train-val-hashes
```

Conformance mặc định ghi `reports/pilot/conformance/uit-conformance-v1/{conformance_report.json,conformance_report.md}` trong repo (ignored). Command mở deterministic train samples của cả hai layouts; hash audit mở train/val bytes để xác minh integrity, **không mở test images, không score**. Full audit phải pass trước GPU pilot; không hash lại mỗi epoch. Giữ dataset immutable sau audit, chạy lại khi root/content thay đổi.

Kiểm report `status=pass`, labels124, counts34997/7497/7502, `test_images_opened=0`, audit `{train:34997,val:7497}`. Report CPU local không thay thế conformance trên target packages/root. Hash-only audit không dùng validation image content để thiết kế geometry. Không sửa v1 nếu conformance fail; xử lý bug/backend/version trước.

CPU unit tests dùng tiny synthetic models để kiểm accumulation, checkpoints/RNG và bounded control flow; không phải GeoGuessr training. GPU smoke tách riêng, chỉ chạy khi chủ động opt in trong GPU allocation:

```bash
# Chỉ điền digest đã review, giống expected_weight_sha256 trong pilot configs.
export CS406_EXPECTED_WEIGHT_SHA256="<reviewed-full-sha256>"
CS406_GPU_SMOKE=1 python -m pytest -q -m gpu
```

## 4. Review và điền configs trước P0

```bash
cp configs/pilot/p0_e1.json configs/pilot/local/p0_e1.json
cp configs/pilot/p0_e1hr.json configs/pilot/local/p0_e1hr.json
cp configs/pilot/p0_e2.json configs/pilot/local/p0_e2.json
```

Mở ba copies, điền reviewed values theo [config contract](../configs/pilot/README.md). Templates cố ý còn `null`; không có default training seed/LR/batch/epochs được chọn thay người dùng. `CS406_DATA_ROOT`, `CS406_WEIGHTS_PATH`, `CS406_RUNS_DIR` override `paths` khi load. Các path config khác relative tới repo, hoặc absolute.

Trước comparable timing, freeze ít nhất: worker count, pin/persistent/prefetch, CPU threads, AMP dtype, GradScaler policy (và numeric fields nếu enabled), TF32 flags, float32 matmul precision, cuDNN benchmark/deterministic, deterministic algorithms/warn-only và compile on/off. TF32 matmul và matmul precision phải nhất quán (`highest` với TF32 disabled; `high` với TF32 enabled trong torch version đã pin). Đây là protocol, không phải ablation. Nếu deterministic CUDA cần `CUBLAS_WORKSPACE_CONFIG`, đặt biến theo [PyTorch reproducibility documentation](https://pytorch.org/docs/2.8/notes/randomness.html) trước Python và giữ policy giống nhau; runtime ghi biến này trong environment.

Classifier phải chọn explicit method/seed cho cả ba architectures, không chỉ E3. `linear_default` log exact reset rule + torch version; `trunc_normal` cần std/bias. Không kế thừa classifier/optimizer từ E1 sang E2.

E1-HR: `1344×576` chỉ first candidate; `global_wh` chưa điền. Review ordered candidate list, suitability/aspect ratio, criteria ID, feasibility thresholds và `resolution_policy.batch_policy` **trước candidate đầu chạy**. Bảng `candidate_batches` có một `{microbatch, accumulation}` cho mỗi ordered candidate, cùng `effective_batch`; actual batch phải khớp hàng tương ứng. Không dùng oversized/unapproved microbatch rồi lấy OOM làm lý do reject resolution. Nếu first candidate pass thì giữ nó; không chọn theo validation score.

P0 ghi `feasibility_context.json` sau verify dataset/preprocessing/offline full weight digest và trước model allocation. Context + fingerprint được đưa vào cả criteria và OOM failure summaries, ràng buộc protocol ID, ordered list/criteria ID, thresholds, seed, runtime, initialization, split/preprocessing/weight identity và reviewed batch policy (cùng source/environment/device/budget). Khi fallback, validator phải khớp context/fingerprint và kiểm actual batch/effective batch của từng predecessor. Reports cũ thiếu context, lỗi trước khi identity được verify hoặc lỗi package/path/numerical không đủ để chuyển fallback. Giữ nguyên mọi failure reports; runner vẫn chạy đúng **một** declared resolution/job. Không có numeric fallback mới trong task này.

P0 LR chỉ sanity-check stability, không validation; record mọi retry. P0 không chứng minh convergence. Có thể kiểm microbatch thủ công bằng run riêng, giữ failure logs; không có automatic architecture/LR/resolution search.

```bash
python -m src.pilot.p0_feasibility --config configs/pilot/local/p0_e1.json --check-config
python -m src.pilot.p0_feasibility --config configs/pilot/local/p0_e1hr.json --check-config
python -m src.pilot.p0_feasibility --config configs/pilot/local/p0_e2.json --check-config
```

`--check-config` chỉ kiểm complete/structural contract; runtime mới kiểm files/reports/CUDA. Đặt `gates.conformance_report` trỏ report UIT đã pass. Reviewed P0 `warmup_updates+measure_updates` phải vừa số updates một epoch; code không cycle để lấp budget. Benchmark warmup khác P1 LR warmup.

## 5. Submit P0 từng profile và review kết quả

Scripts generic dùng partition `normal`, GRES `gpu:2g.35gb:1`, một GPU. Chưa có job nào được submit trong task chuẩn bị code. Các lệnh dưới chỉ dùng **sau khi người dùng approve config và việc chạy**. Điền reviewed Slurm CPU/RAM/time, ≤16 CPUs và ≤72 GiB RAM:

```bash
export CS406_CPUS=<reviewed-cpu-count>
export CS406_MEM=<reviewed-memory-request>
export CS406_WALLTIME=<reviewed-HH:MM:SS>
export CS406_CONFIG="$CS406_REPO/configs/pilot/local/p0_e1.json"
sbatch --cpus-per-task="$CS406_CPUS" --mem="$CS406_MEM" --time="$CS406_WALLTIME" \
  scripts/slurm/pilot_p0.sbatch --run-id p0-e1-reviewed
# Sau khi inspect run trước, đổi CS406_CONFIG và run-id:
# p0_e1hr.json -> p0-e1hr-reviewed
# p0_e2.json   -> p0-e2-reviewed
squeue -u "$USER"
```

`logs/` phải tồn tại **trước** sbatch vì Slurm mở stdout trước script. Không submit trên login node bằng Python trực tiếp. `CS406_PYTHON`, repo/config/data/weights/run variables phải được export cho job; script dùng exact venv interpreter, không phụ thuộc shell activation trong batch.

Mỗi P0 run có `config_resolved.json`, `environment.json`, `feasibility_context.json`, `model_metadata.json`, `timing.json`, `memory.json`, `run_summary.json`; lỗi sớm có thể chỉ config + failure summary. Không checkpoint P0. Inspect status/pass-fail, failure type, steady-state original-images/s, per-microbatch/update time, loader-wait fraction, full-epoch seconds, peak allocated/reserved bytes và thresholds. Timing sync CUDA trước/sau, reset peaks sau benchmark warmup; end-to-end gồm input/transfer/train/numerical checks, không tách riêng kernel-only FLOPs performance. Full epoch **không gồm validation/checkpoint I/O**; P1 ghi epoch walltime train+validation riêng, checkpoint I/O ngoài timer.

Nếu OOM/failure, folder giữ nguyên; sửa config theo reviewed feasibility rule rồi tạo run ID mới. Không ghi đè để che trial lỗi. Warmup/measurement là finite windows; no validation/model-quality claim. Tìm microbatch phù hợp rồi chốt common effective batch theo original images; E2 năm views vẫn một sample. Nếu changed accumulation để match common batch, tạo passing P0 summaries mới cho ba profiles. P1 gate yêu cầu full epoch measured trên cả ba.

## 6. Bounded P1 sau review P0

Chỉ bắt đầu sau khi review P0. Copy ba P1 templates sang `configs/pilot/local/`; điền cùng reviewed runtime, effective batch, initialization, seed, finite epoch budget, warmup policy, scheduler horizon, checkpoint tolerance/tie-break và augmentation amplitudes. Mỗi config `gates.p0_summaries` là list **ba absolute paths** tới passing summaries của E1/E1HR/E2; own profile giữ exact microbatch/accumulation/resolution đã pass. Những source/artifact/weight/environment identities phải trùng.

P1 budget phải review **trước** chạy, không kéo dài tới khi có score mong muốn. Runner dừng đúng `max_epochs`, không early-stop tự động; curves dùng để review/freeze common official horizon/patience về sau. Train CE unweighted/natural, full fine-tuning, AdamW, shared-image jitter; full deterministic validation mỗi epoch. Chỉ một LR đã review để kiểm stability/convergence, không refine grid theo validation Macro-F1. Cần thay LR do lỗi scale thì report/rationale trước, không tạo hidden tuning budget.

```bash
python -m src.pilot.p1_train --config configs/pilot/local/p1_e1.json --check-config
export CS406_CONFIG="$CS406_REPO/configs/pilot/local/p1_e1.json"
sbatch --cpus-per-task="$CS406_CPUS" --mem="$CS406_MEM" --time="$CS406_WALLTIME" \
  scripts/slurm/pilot_p1.sbatch --run-id p1-e1-reviewed
# Sau review, tương tự p1_e1hr và p1_e2; không loop trial search.
```

P1 outputs bổ sung: `metrics.csv`, `history.json`, `learning_curves.svg`, `checkpoint_last.pt`, `checkpoint_best.pt`, timing/memory và summary. `best` chỉ **diagnostic pilot checkpoint**, selected bằng supported validation Macro-F1 + tolerance/tie rule; không gọi final model. Không lấy Top-1 từ epoch khác làm best model metric.

### Walltime và resume

`checkpoint_last.pt` atomic tại completed epoch boundary; lưu model/optimizer/scheduler/scaler, counters, config hash, train seed, Python/NumPy/CPU/CUDA RNG, loader generators, best/history và identity. Epoch0 checkpoint cho phép recover nếu bị kill giữa epoch đầu.

Script gửi USR1 trước walltime 180 giây. Runner cố finish epoch rồi exit; nếu epoch dài hơn khoảng còn lại, Slurm có thể kill và chỉ previous completed epoch được bảo toàn. Chọn walltime từ P0 và checkpoint overhead, không coi 180 giây bảo đảm finish. Có thể request epoch-boundary stop bằng signal trên job step, không gửi message tới người khác.

```bash
export CS406_CONFIG="$CS406_REPO/configs/pilot/local/p1_e1.json"
sbatch --cpus-per-task="$CS406_CPUS" --mem="$CS406_MEM" --time="$CS406_WALLTIME" \
  scripts/slurm/pilot_p1.sbatch \
  --resume "$CS406_RUNS_DIR/p1-e1-reviewed/checkpoint_last.pt"
```

Giữ cùng config paths/bytes/resolved env/source/packages/device policy; không đổi seed, microbatch, LR/horizon hoặc tăng epochs khi resume. Không chạy hai jobs vào cùng folder (OS lock từ chối). Partial epoch bị replay từ boundary đã lưu, không tiếp tục giữa batch; không tính replay thành extra completed budget. Resume không reset scheduler/warmup. `last` là source để resume, không dùng `best` rollback tạo thêm optimization. Bitwise resumed equivalence được test với synthetic CPU/dropout; CUDA/compile phải kiểm trên target policy trước khi coi deterministic. Persistent-worker lifecycle có thể khác khi resume, nhưng sample order và jitter của harness được key theo epoch/image/seed, không phụ thuộc worker scheduling.

## 7. Collect diagnostics và export training recipe sau review

```bash
python -m src.pilot.collect_results --runs-dir "$CS406_RUNS_DIR" \
  --output "$CS406_HOME/reports/pilot/pilot_results.md"
```

Collector giữ cả pass/fail/interrupted runs, resolved configs, environment và measurements; tạo MD + JSON, không tự xếp hạng/chọn architecture/LR/resolution. Không có runs thì fail, không tạo kết quả giả. Chưa tự freeze numeric values chỉ vì pilot hoàn tất.

Sau review, tạo **user-supplied** JSON contract ngoài repo với các keys sau (không copy null templates rồi coi là frozen):

```text
source_runs: absolute run directories của passing P0 + completed P1 cho cả ba profiles
seeds: {tuning: <one numeric ID>, final: [<three distinct numeric IDs>]}
lr_candidates: [<exactly four positive distinct reviewed values>]
runtime, optimizer, scheduler, batch, augmentation, initialization
e1hr_resolution_wh, checkpoint
numeric_value_rationales: lý do cho từng numeric value được freeze, kèm pilot evidence
```

`scheduler` phải ghi common max epochs/horizon/warmup/floor; `checkpoint` phải ghi official patience/tolerance/tie rule được review; `batch` ghi effective batch và từng profile microbatch/accumulation. `numeric_value_rationales` bao phủ từng giá trị vừa chốt. Recipe không cấp phép official run và không chốt E3 protocol thay người dùng.

```bash
python -m src.pilot.collect_results --runs-dir "$CS406_RUNS_DIR" \
  --output "$CS406_HOME/reports/pilot/pilot_results.md" \
  --reviewed-recipe "$CS406_HOME/reviewed_training_recipe.json" \
  --recipe-id training_recipe_v1 --recipes-dir configs/training
```

Export chỉ serialize một reviewed contract hoàn chỉnh và evidence hashes, không suy ra values từ score; tạo `configs/training/training_recipe_v1.json` + `.sha256`, từ chối ghi đè version cũ. Review/upload Git recipe/report riêng sau; master spec chỉ reference recipe, không duplicate numeric contract. **Task hiện tại chưa tạo frozen training recipe hoặc GPU pilot result.** Official comparison về sau vẫn `4 LR × 1 tuning seed / architecture`, sau đó best LR ×3 final seeds; harness không thực hiện bước đó.

## 8. Troubleshooting và giới hạn

| Lỗi | Xử lý |
|---|---|
| Pending config values | Điền/review fields được liệt kê; code không chọn default thay người dùng |
| Missing image | Root phải chứa trực tiếp label folders, giữ manifest nguyên byte |
| Artifact/preprocessing hash mismatch | Restore pinned files; không regenerate split hoặc sửa v1 |
| Conformance gate fail | Chạy lại target environment + full train/val hash audit, giữ dataset immutable |
| CUDA unavailable / CPU wheel | Kiểm trong allocated job, đúng venv và CUDA wheel đã review |
| TF32/matmul conflict | Review consistent explicit runtime flags, không âm thầm override |
| Deterministic CUDA failure | Kiểm CUBLAS workspace/runtime support; record policy, không tắt riêng architecture |
| OOM | Ghi failure; review microbatch/accumulation. HR fallback chỉ theo ordered feasibility rule |
| P1 P0 identity mismatch | Kiểm packages/source/GPU/weights/seeds/batch; rerun affected gates, không bypass |
| Numerical failure | Stop/report LR scale/AMP/gradient issue; không dùng test, không hidden LR search |
| Timeout | Resume từ same run last completed epoch, unchanged budget; review job walltime |
| Existing run ID | Dùng ID mới hoặc explicit resume, không overwrite |

Optional GPU smoke dùng synthetic inputs; actual pretrained strict loading và full-size VRAM/timing chưa được xác minh trên UIT trong task local. Local tests/conformance chỉ là bằng chứng CPU. E3 pilot, official E1 framework, final metrics/test, alternative architectures/augmentation và geometry search nằm ngoài harness này.
