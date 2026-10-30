# Review crop geometry E2 v1 trên ảnh train

## Phạm vi và trạng thái

**Geometry 384/402 và UI preprocessing đã frozen thành v1 sau khi người dùng review/chốt và hoàn tất UI QA train-only.** Nguồn chuẩn hiện tại là [versioned config](../../configs/preprocessing/geoguessr_v1.json) và [preprocessing specification](../../docs/preprocessing_v1.md). Phần dưới ghi lại quá trình visual design lịch sử; các mức 416/448 không phải main experiments. Không train model, không dùng validation/test score và không mở thêm experiment branch.

- Dùng đúng 65 ảnh `split == train` đã chọn từ fixed manifest: 24 ảnh layout 1536 × 662 và toàn bộ 41 ảnh train layout 1030 × 703.
- Đã kiểm tra lại membership, SHA-256 manifest và SHA-256 của 65 file local trước khi tạo preview.
- Manifest SHA-256: `9cbd8b70e8522b5aa4a20fba4fa7b2ecea855cf377f98e41ebb162a07f8a8040`.
- Không sửa ảnh raw. Các ảnh PNG là tài liệu phân tích: màu xám biểu thị vùng UI/padding trong proposal, đường màu và số crop chỉ là annotation trên hình overview/global.
- Local tiles trong preview được lấy từ raw sau draft UI preprocessing, không lấy từ global đã resize. Không dùng jitter trong tài liệu preview.

## Đề xuất ưu tiên: 384 / 402 raw pixels → 320 × 320

| Layout raw | Sau draft footer crop | Cạnh crop raw | Local output | Scale local / global |
|---|---|---:|---|---:|
| 1536 × 662 | 1536 × 648 | 384 | 320 × 320 | 1,429× |
| 1030 × 703 | 1030 × 689 | 402 | 320 × 320 | 1,428× |

Global vẫn là 896 × 384, giữ aspect ratio và letterbox. Nội dung ảnh global lần lượt khoảng 896 × 378 và 574 × 384; đây là kích thước sau draft UI crop, không phải kích thước raw.

Mức zoom trên là tỷ số scale tuyến tính của cùng chi tiết ảnh, không phải zoom quang học và không phải bằng chứng về improvement classification. Local vẫn downsample từ raw, chỉ downsample ít hơn global. Sai khác nhỏ giữa hai layout đến từ làm tròn cạnh crop về pixel nguyên.

Lý do nghiêng về 384: trong các mẫu so sánh, crop vẫn giữ ngữ cảnh đường phố, đồng thời chữ, phương tiện, vạch đường và chi tiết kiến trúc lớn hơn trên input so với candidate 416/448. Candidate 448 chỉ tăng khoảng 22% scale so với global; candidate 384 tăng khoảng 43%, phù hợp hơn với vai trò local detail đã thống nhất. Đây là nhận xét định tính từ preview, chưa chứng minh crop nào đạt score tốt nhất.

## Vị trí crop

Bốn crop phân bố từ trái sang phải. Tâm tính theo ảnh sau draft footer crop:

- C1: `(0,20 W; 0,60 H)`.
- C2: `(0,40 W; 0,60 H)`.
- C3: `(0,60 W; 0,60 H)`.
- C4: `(0,80 W; 0,50 H)`, nâng cao hơn để giảm phần giao với minimap mask.

Tọa độ dưới đây dùng gốc trên trái; khoảng crop là `[x, x + side)` và `[y, y + side)`.

| Layout | Crop | x | y | Side |
|---|---|---:|---:|---:|
| 1536 × 662 | C1 | 115 | 197 | 384 |
| 1536 × 662 | C2 | 422 | 197 | 384 |
| 1536 × 662 | C3 | 730 | 197 | 384 |
| 1536 × 662 | C4 | 1037 | 132 | 384 |
| 1030 × 703 | C1 | 5 | 212 | 402 |
| 1030 × 703 | C2 | 211 | 212 | 402 |
| 1030 × 703 | C3 | 417 | 212 | 402 |
| 1030 × 703 | C4 | 623 | 144 | 402 |

Quy tắc đổi cạnh crop theo layout: tính `s_global = min(896/W, 384/H)` sau UI crop; lấy tỷ số scale mục tiêu từ layout lớn với side 384, rồi tính `side = round(320 / (target_zoom * s_global))`. Không scale cạnh crop chỉ theo width raw. Tọa độ trái/trên được làm tròn; nếu cần thì giới hạn trong frame. Với proposal 384/402 hiện tại, cả bốn crop đều nằm trong frame mà không cần clipping.

## Overlap, coverage và vùng mask

Overlap ở đây là diện tích giao chia cho diện tích một crop, không phải IoU hay phần trăm cạnh.

| Layout | Overlap C1–C2 | C2–C3 | C3–C4 | Mask lớn nhất trong một crop |
|---|---:|---:|---:|---:|
| 1536 × 662 | 20,1% | 19,8% | 16,7% | 8,8% |
| 1030 × 703 | 48,8% | 48,8% | 40,5% | 9,7% |

Layout nhỏ có nhiều overlap hơn vì giữ mức zoom tương đối so với global trong một frame hẹp hơn. Đây là đánh đổi của proposal, không phải bốn vùng độc lập hoàn toàn. Union của bốn crop phủ khoảng 50,9% / 59,7% diện tích frame sau footer crop; phần này bao gồm cả vùng UI mask nằm trong crop. Global giữ toàn cảnh còn lại.

## UI template v1 và implementation preview lịch sử

Các tọa độ sau đã được giữ nguyên khi freeze v1 sau QA 297 train images. Crop 14 pixel sát đáy, sau đó mask các hình chữ nhật sau; không crop mạnh toàn ảnh và không che navigation arrows bằng một vùng giữa ảnh.

| Layout | Thành phần | x | y | width | height |
|---|---|---:|---:|---:|---:|
| 1536 × 662 | Score panel | 1246 | 8 | 266 | 72 |
| 1536 × 662 | Controls/compass | 6 | 366 | 76 | 282 |
| 1536 × 662 | Minimap + Guess | 1250 | 440 | 262 | 208 |
| 1030 × 703 | Score panel | 740 | 8 | 266 | 72 |
| 1030 × 703 | Controls/compass | 6 | 407 | 76 | 282 |
| 1030 × 703 | Minimap + Guess | 827 | 547 | 180 | 142 |

- Màu fill trong PNG preview lịch sử là RGB `(124,116,104)`. Training v1 dùng chính xác `[0.485,0.456,0.406]` trong RGB float32 [0,1], không làm tròn uint8, theo versioned config.
- PNG preview dùng `System.Drawing HighQualityBilinear`. Training v1 dùng `torchvision.transforms.functional.resize` trên CPU float32 tensor, bilinear và explicit `antialias=True`; không tuyên bố renderer PNG có output số học giống backend này.
- Cùng một ảnh đã xử lý UI cấp raw được dùng cho global và mọi local tile. Annotation đường màu/số crop không được đưa vào model input.
- Navigation arrows được chủ động giữ lại theo quyết định của người dùng. Các watermark chìm của Street View vẫn có thể còn trong scene; không mô tả pipeline là loại sạch mọi overlay.

## Quan sát và giới hạn

- Ở mẫu Vietnam, crop bên phải vẫn giữ biển hiệu và mặt tiền; crop giữa giữ vạch qua đường, curb và phương tiện.
- Ở mẫu Japan đô thị, crop giữ mặt đường, tường rào, phương tiện, dây điện và chi tiết công trình. Một số phần cao của cột/nhà nằm ngoài local và vẫn thuộc global.
- Ở mẫu Kenya, crop trái chủ yếu là thực vật; ở mẫu Iceland, crop trái chứa nhiều mặt nước/cảnh quan. Deterministic crop không bảo đảm mỗi view có signage hoặc manh mối đặc trưng mạnh.
- Ở mẫu France chụp sát tường, nhiều crop vẫn chủ yếu là tường. Tăng/giảm side trong phạm vi đã preview không tạo ra thông tin vốn không có trong ảnh.
- Chưa có thống kê định lượng về tỷ lệ sky/road/vegetation cho toàn bộ train. Không suy diễn visual review thành một phép đo semantic coverage toàn dataset.
- Chỉ xem 24/34.956 ảnh train layout lớn; chưa chứng minh UI template bao phủ mọi biến thể. Layout nhỏ đã xem đủ 41 ảnh train, không mở ảnh validation/test.
- Không thêm content-aware crop rejection, detector hoặc crop selection theo prediction vào main scope.

## Preview để review

So sánh ba kích thước, mỗi dòng là bốn tile output 320 × 320:

- [Vietnam, layout lớn](compare_canvas_1629512178.png).
- [Japan, layout lớn](compare_canvas_1629430411.png).
- [Japan, layout nhỏ](compare_canvas_1630265548.png).
- [Sweden, layout nhỏ](compare_canvas_1630265669.png).

Overview proposal 384/402 trên toàn bộ 65 mẫu:

- Layout lớn: [trang 1](overview384_1536x662_1.png), [trang 2](overview384_1536x662_2.png).
- Layout nhỏ: [trang 1](overview384_1030x703_1.png), [trang 2](overview384_1030x703_2.png), [trang 3](overview384_1030x703_3.png), [trang 4](overview384_1030x703_4.png).

Các ví dụ khó trong proposal 384/402:

- [Kenya: vegetation](detail384_canvas_1629398983.png).
- [Iceland: landscape](detail384_canvas_1629397451.png).
- [France: ảnh sát tường](detail384_canvas_1630265536.png).
- [France: đường giữa cây](detail384_canvas_1630265841.png).

Metadata có thể kiểm tra lại: [recommended_geometry.json](recommended_geometry.json), [draft_geometry.json](draft_geometry.json), [geometry_checks.json](geometry_checks.json).

Geometry và UI đã freeze; xem [UI QA mở rộng](../preprocessing_qa_v1/qa_review_vi.md). Các phương án không chọn chỉ là tài liệu preview, không tự động trở thành thí nghiệm huấn luyện hoặc ablation. [Master spec](../../docs/master_spec.md) và [pilot plan](../../docs/pilot_plan.md) chỉ là draft, chưa frozen và không được tự động triển khai. Công việc hiện tại chỉ gồm preprocessing/geometry/split reproducibility; mọi bước tiếp theo chờ người dùng review và yêu cầu riêng. Không mở geometry search.
