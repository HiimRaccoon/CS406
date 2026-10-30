# UI preprocessing QA v1 — train only

Kết quả phục vụ freeze template UI, không dùng classification score và không tune crop geometry. Specification chuẩn: [preprocessing v1](../../docs/preprocessing_v1.md) và [versioned config](../../configs/preprocessing/geoguessr_v1.json).

## Mẫu và kiểm tra

- 297 train images thuộc đủ 124 labels: 256 layout 1536×662 và toàn bộ 41 train images layout 1030×703.
- Deterministic rank: `SHA256(UTF8("ui-qa-v1:406|" + relative_path))` tăng dần. Bao gồm một ảnh layout lớn mỗi nhãn nếu có, 24 ảnh layout lớn đã preview, rồi điền đến 256 theo rank; thêm toàn bộ layout nhỏ train. Seed/rank này chỉ chọn QA sample, không tạo lại split.
- Đã đối chiếu membership với manifest nguồn, SHA-256 từng file và decoded dimensions của mọi ảnh trong sample.
- Visual review đủ 13 contact sheets, không mở ảnh validation/test. Red rectangle là biên mask dự kiến trên raw UI crop; đường đỏ ở footer là vị trí cắt 14px. Không mask/fill lên nội dung raw trong contact sheets để có thể nhìn UI ở hai phía boundary.
- Không thấy score panel, controls/compass, minimap hoặc Guess vượt khỏi template trong sample đã xem. Footer attribution của mẫu nằm phía bị crop. Không cần sửa template từ draft; không đổi vị trí/cạnh local crop.

## Kết luận freeze

Giữ footer 14px, các score/controls/minimap rectangles đã công bố, navigation-arrow exception và geometry 384/402 → 320×320. UI template áp dụng chung E1/E1-HR/E2 và E3 cache.

Training implementation được khai báo riêng: exact ImageNet mean fill trên float32 tensor, TorchVision bilinear tensor resize với `antialias=True`. Contact sheets dùng System.Drawing chỉ để visual QA; chúng không chứng minh numerical equivalence với tensor backend và không được dùng làm model input.

Đây là kiểm tra sample, không bảo đảm bao phủ mọi biến thể UI trong toàn dataset. Navigation arrows và embedded watermarks có thể còn. TorchVision conformance trên UIT vẫn là gate bắt buộc đầu pilot; local Python launcher không khả dụng nên chưa chạy gate này tại local.

## Bằng chứng

- [Sample manifest và SHA-256 từng ảnh](sample_manifest.json).
- [Decoded dimensions](decode_results.json).
- [QA summary](summary.json).

Contact sheets:

| Trang | Chỉ số sample |
|---|---|
| [01](ui_qa_01.png) | 1–24 |
| [02](ui_qa_02.png) | 25–48 |
| [03](ui_qa_03.png) | 49–72 |
| [04](ui_qa_04.png) | 73–96 |
| [05](ui_qa_05.png) | 97–120 |
| [06](ui_qa_06.png) | 121–144 |
| [07](ui_qa_07.png) | 145–168 |
| [08](ui_qa_08.png) | 169–192 |
| [09](ui_qa_09.png) | 193–216 |
| [10](ui_qa_10.png) | 217–240 |
| [11](ui_qa_11.png) | 241–264 |
| [12](ui_qa_12.png) | 265–288 |
| [13](ui_qa_13.png) | 289–297 |
