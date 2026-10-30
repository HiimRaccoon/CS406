# Pilot harness — implementation handoff

Task này chỉ implement pilot harness theo yêu cầu mới, không chuyển master draft thành frozen hoặc official E1 implementation. Source mới ngoài preprocessing nằm hết dưới `src/pilot/`; không có `src/data/`, `src/models/` hoặc `src/training/`.

## Ranh giới và cách hoạt động

- `src/preprocessing/geoguessr_v1.py`: implementation reusable của frozen UI/fill/letterbox/raw local geometry, tensor bilinear antialias, ImageNet normalize. Config/hash v1 không đổi.
- `src/pilot/dataset.py`: exact manifest/provenance/support checks, train/val-only image API, natural epoch sampler và per-image/epoch jitter; không regenerate split.
- `models.py`: offline shared ConvNeXt-Tiny builder, 768-d E1/E1HR hoặc concat global/mean-local 1536-d E2, linear124 head, explicit init/optimizer groups. Không inheritance GeoGuessr checkpoint giữa profiles.
- `config.py`, `runtime.py`, `engine.py`: chỉ các mechanics cần cho pilot: fail-on-pending config, identity/gates, finite update loops, AMP/accumulation, warmup/cosine, diagnostics. Không registry/search/distributed framework.
- `conformance.py`, `p0_feasibility.py`, `p1_train.py`: ba CLI tách biệt; P0 không score/claim convergence, P1 bounded train+validation và không tự extend.
- `metrics.py`, `checkpoint.py`, `collect_results.py`: supported-GT Macro-F1 + full124 Top-k/Weighted-F1, atomic epoch-boundary resume và diagnostic collection. Export recipe chỉ từ explicit reviewed values/evidence, không tự chọn numeric contract.
- Hai generic Slurm scripts và [runbook](server_pilot_runbook.md) có setup/offline weights, gates, timing, resume và collection.

`docs/master_spec.md` và `docs/pilot_plan.md` trong commit cũng mang các patch tài liệu đã được yêu cầu ở bước review trước: scheduler/seed pending và P0/P1/runtime/head protocol. Không rewrite architecture hoặc freeze numeric training choices. Những dòng deferred/no-code của draft trước ghi lại stage review tài liệu; authorization mới chỉ cấp phép **pilot harness code**, không cấp phép chạy GPU jobs hay official experiments.

## Kiểm tra local đã thực hiện

Python 3.11.18, torch 2.8.0+cpu, torchvision 0.23.0+cpu; không có CUDA runtime khả dụng trong local env.

- **28 CPU tests pass, 1 opt-in GPU test skipped**: exact manifest/tamper/test-access guards, preprocessing hai layout, metrics đối chiếu scikit-learn, actual random-weight ConvNeXt architecture/gradients, initialization, optimizer grouping, uneven accumulation, RNG/scheduler resume, synthetic bounded P0/P1 control flow và gates.
- Actual TorchVision conformance trên deterministic train samples của cả hai raw layouts: pass; exact fill/padding, shape, repeated determinism và crop trực tiếp processed raw.
- Full SHA-256 image audit: **34.997 train + 7.497 validation pass**, **0 test images opened**. Validation chỉ dùng để integrity-hash bytes; không dùng content/score để thiết kế preprocessing.
- Chín protected hashes (frozen preprocessing config/checksum, exact split artifacts/provenance, preprocessing spec, grillme) giữ nguyên.
- Python compile và Bash syntax của cả hai Slurm scripts pass.

Local conformance outputs nằm ở `reports/pilot/conformance/local-cpu-v1-audited/` (ignored). Đây là numerical/integrity QA, **không phải GPU pilot result**, không thay conformance trên UIT environment/root. Không có `pilot_results.md` giả hoặc frozen training recipe được tạo.

Reproduce CPU smoke từ repo root sau khi cài requirements:

```bash
python -m pytest -q
python -m src.pilot.dataset
python -m src.pilot.conformance --config configs/pilot/conformance.json \
  --data-root /path/to/raw/geoguessr --verify-all-train-val-hashes
```

Trong workspace Windows sandbox hiện tại, pytest temp directory mặc định bị hạn chế; command đã chạy dùng `--basetemp=.tools/pytest-pilot-3`. Các synthetic test seeds/mini-budgets không phải research seed IDs hoặc reviewed training parameters.

## Bàn giao UIT và các giá trị còn pending

[SERVER_UPLOAD_CHECKLIST.md](../SERVER_UPLOAD_CHECKLIST.md) là exact upload manifest: Git source/config/artifacts/scripts/tests/docs/dependency files; dataset, binary pretrained weights, environment và outputs phải tồn tại riêng trên server.

[Server runbook](server_pilot_runbook.md) có recommended layout `/datastore/binhvp/<my-folder>/CS406/{repo,data/geoguessr,pretrained,runs/pilot,reports/pilot,env}`, exact first clone/setup/conformance commands, config review rồi từng P0/P1 sbatch/resume command. Dependencies pinned trong [requirements.txt](../requirements.txt); CUDA wheel index cần phù hợp UIT driver, không chọn thay người dùng. Raw data/weights/runs/logs/secrets không được commit.

Còn pending trước GPU chạy: exact pilot/training seed IDs, head init method/parameters/seed policy, reviewed E1-HR ordered feasibility policy và chosen candidate, microbatch/accumulation/effective batch, runtime/AMP/scaler/TF32/compile/determinism, AdamW LR/betas/eps, benchmark windows/criteria, bounded P1 epochs, warmup/cosine floor, checkpoint tolerance/tie-break, clipping và jitter amplitudes. Official 4 LR values, final seeds và early-stopping patience chỉ freeze sau pilot/review phù hợp. Full pretrained SHA-256 chỉ ghi sau khi có file thật.

Chưa download/strict-load pretrained file thật, chưa kiểm CUDA/UIT numerical behavior, VRAM, throughput hoặc walltime. Chưa chạy Slurm, official LR search, final E1/E1HR/E2, E3 hoặc test evaluation. Không có `e1_spec.md` hoặc permanent experiment framework mới. Promotion/refactor chỉ sau pilot và review E1 spec.

## Toàn bộ file mới/sửa trong bàn giao

Danh sách dưới đây bao gồm các reviewed document patches đã có, không bao gồm files frozen hoặc local environment/output binaries không đổi.

```text
.gitattributes
.gitignore
.python-version
README.md
SERVER_UPLOAD_CHECKLIST.md
configs/pilot/README.md
configs/pilot/conformance.json
configs/pilot/p0_e1.json
configs/pilot/p0_e1hr.json
configs/pilot/p0_e2.json
configs/pilot/p1_e1.json
configs/pilot/p1_e1hr.json
configs/pilot/p1_e2.json
docs/master_spec.md
docs/pilot_implementation_review.md
docs/pilot_plan.md
docs/server_pilot_runbook.md
pyproject.toml
requirements.txt
scripts/slurm/pilot_p0.sbatch
scripts/slurm/pilot_p1.sbatch
src/__init__.py
src/pilot/__init__.py
src/pilot/checkpoint.py
src/pilot/collect_results.py
src/pilot/config.py
src/pilot/conformance.py
src/pilot/dataset.py
src/pilot/engine.py
src/pilot/metrics.py
src/pilot/models.py
src/pilot/p0_feasibility.py
src/pilot/p1_train.py
src/pilot/runtime.py
src/preprocessing/__init__.py
src/preprocessing/geoguessr_v1.py
tests/conftest.py
tests/test_checkpoint_engine.py
tests/test_collection.py
tests/test_config.py
tests/test_gates.py
tests/test_gpu_smoke.py
tests/test_manifest.py
tests/test_metrics.py
tests/test_models.py
tests/test_pilot_runners.py
tests/test_preprocessing.py
```
