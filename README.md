# CS406 — GeoGuessr country classification

ConvNeXt-Tiny benchmark trên fixed split 124 quốc gia/vùng lãnh thổ. Main scope: E1 full-image, E1-HR higher-resolution full-image, E2 shared-encoder global/local, E3 natural head retraining vs class-uniform cRT trên frozen E1/E2 features.

- [Master experiment specification — DRAFT, chưa frozen](docs/master_spec.md).
- [Preprocessing v1 — frozen](docs/preprocessing_v1.md), [config](configs/preprocessing/geoguessr_v1.json) và [checksum](configs/preprocessing/geoguessr_v1.sha256).
- [Fixed split artifacts và provenance](artifacts/baseline124/README.md); không regenerate từ seed 406.
- [Train-only UI QA: 297 ảnh, 124 labels](reports/preprocessing_qa_v1/qa_review_vi.md).
- [Crop geometry visual review](reports/crop_geometry_preview/geometry_review_vi.md).
- [UIT pilot plan — DRAFT, deferred](docs/pilot_plan.md).

Geometry E2 v1: global 896×384; bốn raw local crops 384×384 hoặc 402×402 theo layout → 320×320; navigation arrows được chủ động giữ. Geometry không mở lại thành search branch. Candidate 416/448 chỉ là visual design alternatives lịch sử.

Phạm vi công việc hiện tại chỉ gồm preprocessing/geometry và fixed-split reproducibility. Master spec và pilot plan là draft chờ người dùng review lại sau khi toàn bộ grill/preprocessing review kết thúc. Không tự động mở rộng master spec, tạo experiment-specific specs/training code hoặc chuyển sang implementation/pilot khi chưa có yêu cầu tiếp theo.
