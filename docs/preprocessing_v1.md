# Preprocessing và geometry v1 — frozen

Nguồn chuẩn duy nhất: [geoguessr_v1.json](../configs/preprocessing/geoguessr_v1.json), ID `geoguessr-ui-global-local-v1`, kèm [SHA-256](../configs/preprocessing/geoguessr_v1.sha256). Ngày freeze theo thời gian người dùng: 2026-10-30.

Đây là specification cho implementation sắp tới, không phải tuyên bố đã có training pipeline hoặc đã chạy TorchVision trên UIT. File `recommended_geometry.json` trong report chỉ còn là bản tương thích/ghi nhận quyết định và trỏ về config này. Không dùng status hay renderer của report để quyết định preprocessing lúc train.

## Fixed split và coordinate convention

Đọc bốn artifact tại [artifacts/baseline124](../artifacts/baseline124/README.md), kiểm tra [provenance](../artifacts/baseline124/provenance.json), không regenerate split. Validation có tên `val` trong manifest.

Mọi kích thước trong config có thứ tự `[width,height]`; tham số `size` của TorchVision nhận `[height,width]`. Rectangle là `[x,y,width,height]`, gốc trên trái, khoảng nửa mở. Layout được nhận dạng từ kích thước pixel decoded, không từ nhãn hay split. Chỉ nhận hai layout đã biết; layout khác hoặc ảnh lỗi phải báo lỗi, không tự rescale template và không bỏ mẫu.

Decode bằng `Pillow.Image.open(...).convert("RGB")`, không tự xoay theo EXIF. Chuyển bằng `torchvision.transforms.functional.pil_to_tensor(rgb).to(torch.float32).div(255.0)`: tensor CHW, RGB, CPU float32, khoảng [0,1]. `pil_to_tensor` giữ dtype nguồn nên phép đổi float và chia 255 phải explicit. [TorchVision API](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.pil_to_tensor.html)

## UI preprocessing đã chốt

Crop **14 pixel sát đáy**, giữ nguyên gốc tọa độ trên trái. Sau crop, hai frame có kích thước 1536×648 và 1030×689. Mask các rectangle sau:

| Layout raw | Mask | x | y | width | height |
|---|---|---:|---:|---:|---:|
| 1536×662 | Score | 1246 | 8 | 266 | 72 |
| 1536×662 | Controls/compass | 6 | 366 | 76 | 282 |
| 1536×662 | Minimap + Guess | 1250 | 440 | 262 | 208 |
| 1030×703 | Score | 740 | 8 | 266 | 72 |
| 1030×703 | Controls/compass | 6 | 407 | 76 | 282 |
| 1030×703 | Minimap + Guess | 827 | 547 | 180 | 142 |

**Navigation arrows được chủ động giữ lại.** Không thêm central road mask hoặc arrow detector. Watermark chìm của Street View có thể còn; không mô tả preprocessing là loại sạch mọi overlay.

UI fill và letterbox fill dùng **RGB float32 chính xác `[0.485,0.456,0.406]` trước normalization**. Không chuyển thành `(124,116,104)` hoặc làm tròn qua uint8. Màu uint8 trong PNG cũ chỉ thuộc implementation visualization lịch sử. Vùng padding và nội thất vùng mask hằng sẽ gần/đúng zero sau normalization; các biên mask đi qua antialias có thể trộn với scene.

Với training photometric augmentation, dùng valid-scene mask để loại UI khỏi thống kê, lấy một bộ brightness/contrast parameters cho một ảnh gốc và áp dụng trước khi dựng view. Thứ tự brightness rồi contrast; dùng thống kê scene đã chỉnh brightness. Sau đó fill UI bằng mean, bảo đảm fill không bị jitter. Công thức, clamp và grayscale weights được ghi trong config. Biên độ jitter thuộc training recipe, cần freeze trước LR search; không phải một lựa chọn geometry mới.

## Resize, letterbox và normalization

Training contract: `torchvision.transforms.functional.resize` trên **CPU float32 tensor**, `interpolation=InterpolationMode.BILINEAR`, `antialias=True`, `max_size=None`, size là hai số `[H,W]`. Không dùng PIL resize, System.Drawing hoặc `weights.transforms()` cho model input. Tensor resize hỗ trợ bilinear và antialias; khai báo explicit tránh phụ thuộc default phiên bản. [TorchVision resize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.resize.html)

Với full-image target `(Wt,Ht)`, tính `s=min(Wt/W,Ht/H)`. Content width/height dùng `floor(value+0.5)` và giới hạn vào target; không dùng Python `round()` với cách xử lý tie khác. Center content bằng floor cho left/top, pixel dư nằm ở right/bottom. Resize một lần rồi đặt lên canvas mean-filled. Không quantize giữa các bước và không reapply hard binary mask sau resize: giữ cạnh antialias nhất quán.

Với target 896×384:

| Layout raw | Content sau resize | Padding left/top/right/bottom |
|---|---|---|
| 1536×662 | 896×378 | 0 / 3 / 0 / 3 |
| 1030×703 | 574×384 | 161 / 0 / 161 / 0 |

Cuối cùng dùng `torchvision.transforms.functional.normalize(..., mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], inplace=False)`. Normalize sau resize/padding. [TorchVision normalize](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.normalize.html)

## Geometry E2 v1 đã chốt

Global 896×384; bốn local 320×320, crop từ cùng UI-processed raw tensor, không từ resized global hoặc letterboxed canvas. Vị trí tương đối theo frame sau footer crop: `(0.20W,0.60H)`, `(0.40W,0.60H)`, `(0.60W,0.60H)`, `(0.80W,0.50H)`.

Các integer rectangle dưới đây là nguồn chuẩn lúc chạy. Không tính lại crop side từ resolution của E1-HR hay từ một quy tắc tự thích nghi khác.

| Layout | Crop | x | y | side |
|---|---|---:|---:|---:|
| 1536×662 | 1 | 115 | 197 | 384 |
| 1536×662 | 2 | 422 | 197 | 384 |
| 1536×662 | 3 | 730 | 197 | 384 |
| 1536×662 | 4 | 1037 | 132 | 384 |
| 1030×703 | 1 | 5 | 212 | 402 |
| 1030×703 | 2 | 211 | 212 | 402 |
| 1030×703 | 3 | 417 | 212 | 402 |
| 1030×703 | 4 | 623 | 144 | 402 |

Mỗi square crop được resize trực tiếp về 320×320 bằng cùng tensor backend. Geometry giống nhau ở train/val/test và E3 cache. Không random crop, detector, content-based rejection hay đổi vị trí theo prediction. Candidate 416/448 chỉ là visual design alternatives đã xem, không phải main experiments hoặc LR-search options.

## Shared policy và versioning

E1, E1-HR, E2 cùng decode/UI/fill/resize/normalization. E1-HR chỉ truyền full-image target từ run config được freeze sau UIT pilot; resolution E1-HR chưa chốt và không cần compute-match E2. E2 crop geometry không phụ thuộc lựa chọn đó.

Version v1 freeze deterministic UI, geometry, fill, backend, interpolation, antialias, rounding và normalization. Jitter amplitudes, batch/accumulation, LR candidates và training horizon là training parameters riêng cần chốt ở pilot, không làm mở geometry search.

Thay đổi bất kỳ trường deterministic nào phải tạo version preprocessing mới cùng fingerprint mới, không sửa v1 trong im lặng. Mỗi run và E3 feature cache lưu cả config ID, full config SHA-256, dataset fingerprints và package versions.

## QA và giới hạn xác nhận

[UI QA v1](../reports/preprocessing_qa_v1/qa_review_vi.md) dùng 297 ảnh train từ 124 nhãn: 256 layout lớn và toàn bộ 41 layout nhỏ. Hash, decoded dimensions và cả 13 contact sheets đã được kiểm tra. Không phát hiện UI mục tiêu vượt template trong sample; không đổi UI rectangle hoặc geometry từ draft đã review.

Đây là QA sample, không chứng minh template đúng trên mọi ảnh train. System.Drawing chỉ dựng ảnh kiểm tra UI. Chưa chạy numerical conformance bằng TorchVision tại local vì Python launcher không khả dụng. Các conformance checks trong [pilot plan](pilot_plan.md) bắt buộc chạy trên UIT trước khi training pilot; không được âm thầm thay backend nếu môi trường chưa hỗ trợ contract.
