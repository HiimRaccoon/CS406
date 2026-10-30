# CS406 — GeoGuessr country classification

ConvNeXt-Tiny benchmark trên fixed split 124 quốc gia/vùng lãnh thổ. Main scope: E1 full-image, E1-HR higher-resolution full-image, E2 shared-encoder global/local, E3 natural head retraining vs class-uniform cRT trên frozen E1/E2 features.

- [Master specification — DRAFT FOR USER REVIEW](docs/master_spec.md): objective, ASCII workflows, experiment matrix, FIXED/PENDING recipe và [project checklist](docs/master_spec.md#project-checklist); chưa frozen, chưa authorize implementation.
- [Grillme đã review](docs/grillme.md): biên bản 34 câu hỏi và các quyết định nguồn cho master draft.
- [Preprocessing v1 — frozen](docs/preprocessing_v1.md), [config](configs/preprocessing/geoguessr_v1.json) và [checksum](configs/preprocessing/geoguessr_v1.sha256).
- [Fixed split artifacts và provenance](artifacts/baseline124/README.md); không regenerate từ seed 406.
- [Train-only UI QA: 297 ảnh, 124 labels](reports/preprocessing_qa_v1/qa_review_vi.md).
- [Crop geometry visual review](reports/crop_geometry_preview/geometry_review_vi.md).
- [UIT pilot plan — DRAFT, deferred](docs/pilot_plan.md).

Geometry E2 v1: global 896×384; bốn raw local crops 384×384 hoặc 402×402 theo layout → 320×320; navigation arrows được chủ động giữ. Geometry không mở lại thành search branch. Candidate 416/448 chỉ là visual design alternatives lịch sử.

Master spec đã được viết lại để người dùng review; vẫn là draft, không phải frozen specification. Preprocessing/geometry và fixed split có version riêng đã freeze. Pilot vẫn deferred; không tạo experiment-specific specs/training code hoặc chạy pilot/model/training trong task viết tài liệu. Các bước tiếp theo phải theo workflow và được người dùng review/yêu cầu riêng.
