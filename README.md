# CS406 — GeoGuessr country classification

ConvNeXt-Tiny benchmark trên fixed split 124 quốc gia/vùng lãnh thổ. Main scope: E1 full-image, E1-HR higher-resolution full-image, E2 shared-encoder global/local, E3 natural head retraining vs class-uniform cRT trên frozen E1/E2 features.

- [Master specification — DRAFT FOR USER REVIEW](docs/master_spec.md): objective, ASCII workflows, experiment matrix, FIXED/PENDING recipe và [project checklist](docs/master_spec.md#project-checklist); chưa frozen, chưa authorize implementation.
- [Grillme đã review](docs/grillme.md): biên bản 34 câu hỏi và các quyết định nguồn cho master draft.
- [Preprocessing v1 — frozen](docs/preprocessing_v1.md), [config](configs/preprocessing/geoguessr_v1.json) và [checksum](configs/preprocessing/geoguessr_v1.sha256).
- [Fixed split artifacts và provenance](artifacts/baseline124/README.md); không regenerate từ seed 406.
- [Train-only UI QA: 297 ảnh, 124 labels](reports/preprocessing_qa_v1/qa_review_vi.md).
- [Crop geometry visual review](reports/crop_geometry_preview/geometry_review_vi.md).
- [UIT pilot plan — DRAFT, deferred](docs/pilot_plan.md).
- [Server pilot runbook](docs/server_pilot_runbook.md) và [upload checklist](SERVER_UPLOAD_CHECKLIST.md): setup, offline weights, conformance, P0/P1, resume và collection.
- [Pilot implementation handoff](docs/pilot_implementation_review.md): ranh giới code, validation evidence, pending values và toàn bộ file mới/sửa.

Geometry E2 v1: global 896×384; bốn raw local crops 384×384 hoặc 402×402 theo layout → 320×320; navigation arrows được chủ động giữ. Geometry không mở lại thành search branch. Candidate 416/448 chỉ là visual design alternatives lịch sử.

Master spec vẫn là draft, chưa frozen; preprocessing/geometry và fixed split có version riêng đã freeze. Theo yêu cầu mới, repo có **pilot harness self-contained**, chưa chạy GPU pilot hoặc official experiments. Numeric pilot configs còn pending và cần review trước khi submit. Không có official E1/E2/E3 implementation.

```text
src/
├── preprocessing/   # frozen v1 implementation, reusable
└── pilot/           # dataset/models/metrics/P0/P1/checkpoint/collection, pilot only
```

Không có permanent `src/data/`, `src/models/` hoặc `src/training/`. Sau pilot và review `e1_spec.md`, reusable components mới có thể được promote có chủ đích.

CPU checks từ repo root, sau khi cài dependencies:

```bash
python -m pytest -q
python -m src.pilot.dataset
python -m src.pilot.conformance --config configs/pilot/conformance.json --data-root /path/to/geoguessr
```

P0/P1 cần CUDA, offline pretrained file và reviewed configs; xem [config contract](configs/pilot/README.md). Không có automatic LR/resolution search, test evaluator, distributed training hoặc E3 code. Pilot code không cấp phép chạy official experiments hay chuyển master spec thành frozen.
