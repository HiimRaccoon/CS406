# UIT pilot — source/upload checklist

**Pilot harness only.** Master specification vẫn draft; official E1/E2/E3 chưa implement. Không có GPU pilot/result hoặc final/test evaluation được thực hiện trong task chuẩn bị code.

## Source phải có từ Git

Dùng `git clone https://github.com/HiimRaccoon/CS406.git repo` hoặc `git pull --ff-only`, không copy từng Python file.

- `src/__init__.py`
- `src/preprocessing/{__init__.py,geoguessr_v1.py}`
- `src/pilot/` — config, dataset, models, metrics, conformance, P0, P1, checkpoint, runtime/engine helpers, result collector.
- `configs/preprocessing/{geoguessr_v1.json,geoguessr_v1.sha256}` — frozen, giữ nguyên.
- `configs/pilot/{README.md,conformance.json,p0_e1.json,p0_e1hr.json,p0_e2.json,p1_e1.json,p1_e1hr.json,p1_e2.json}`.
- `configs/pilot/reviewed/{README.md,*.example.json}` — preparation inputs, không phải ready-to-submit runtime configs; hash/threshold/path placeholders phải được review/resolve.
- `artifacts/baseline124/{manifest.csv,summary.json,labels.json,class_support.csv,provenance.json,README.md}` — exact artifacts, không regenerate.
- `scripts/slurm/{pilot_p0.sbatch,pilot_p1.sbatch}`.
- `tests/`, `requirements.txt`, `.python-version`, `pyproject.toml`, `.gitattributes`, `.gitignore`.
- `docs/{master_spec.md,pilot_plan.md,preprocessing_v1.md,grillme.md,server_pilot_runbook.md,pilot_config_review.md}`; `README.md` và checklist này.

## Không nằm trong Git — cần có trên server

- Raw GeoGuessr JPEG dataset, giữ nguyên bytes; dataset root chứa `Aland/`, `Argentina/`, … như manifest relative paths.
- **Offline** `convnext_tiny-983f1562.pth` từ đúng ImageNet1K V1 checkpoint; file exists → hash thực tế → user review → pin cùng full SHA-256 vào mọi P0/P1 runtime config, không dùng hash đoán.
- Python 3.11 environment và packages/CUDA wheels phù hợp UIT driver; wheelhouse nếu server không có Internet.
- Reviewed, filled P0/P1 runtime configs trong `configs/pilot/local/` hoặc path ngoài repo; environment variables.
- Runs, logs, checkpoints, conformance/pilot reports; package freeze, outputs và credentials/secrets.

## Trước khi submit bất kỳ GPU job nào

- [ ] Đọc [runbook](docs/server_pilot_runbook.md); record source commit.
- [ ] Chuẩn bị env, dataset root và offline weight file.
- [ ] Tests CPU pass; exact artifact checks pass.
- [ ] Conformance trên target environment pass, gồm train+val image hash audit, không mở test images.
- [ ] Review/fill explicit P0 mode, seed, initialization, optimizer/runtime policy, batch và bounded budgets. Full reviewed SHA/conformance vẫn mandatory ở cả hai mode.
- [ ] First diagnostics dùng `measurement_only` (thresholds có thể null); sau review measurements mới freeze legitimate thresholds và chạy fresh `criteria` full epochs. Không promote measured reports thành comparable passes hoặc dùng measurement-only OOM cho HR resolution fallback.
- [ ] E1-HR reviewed ordered list/selection rule; không tự thêm fallback hoặc chọn theo score.
- [ ] P0: một reviewed config mỗi job. P1 chỉ sau ba passing full-epoch P0 summaries và budget review.

Không upload raw data, pretrained binary, checkpoints, runs/logs hoặc secrets lên Git. Không có `src/data/`, `src/models/`, `src/training/` ở stage này. Promotion thành official framework chỉ sau pilot và review `e1_spec.md`.
