# src/attachments/

Đọc và lưu tệp người dùng gửi trong chat. Tệp này **không** vào kho tài liệu chung.

## Các file

| File | Chức năng |
| :-- | :-- |
| [reader.py](reader.py) | `read_file()`: chuyển tệp thành các đoạn chữ, giữ số trang hoặc tên sheet |
| [vision.py](vision.py) | Chuẩn hóa ảnh và gọi model thị giác; có model dự phòng (chuyển đổi mới thử bằng test giả lập) |
| [store.py](store.py) | `AttachmentStore`: lưu, kiểm tra chủ sở hữu, gắn với hội thoại, dọn tệp cũ |

## Cách đọc

| Loại | Cách đọc |
| :-- | :-- |
| PDF có chữ | Lấy lớp chữ từng trang. Hình trên trang có chữ không được đọc |
| PDF scan (trang dưới 30 ký tự) | Chuyển trang thành ảnh cho model thị giác, tối đa `ATTACHMENT_MAX_SCAN_PAGES` trang |
| Ảnh | Chép chữ và mô tả; khi hỏi, File Agent xem lại ảnh theo câu hỏi |
| `.docx`, `.xlsx`, `.pptx` | Lấy chữ, bảng, ghi chú slide |
| HTML, văn bản, mã nguồn | Giải mã UTF-8, UTF-16, hoặc cp1258 |

## Vòng đời

1. Tải lên: đọc ngay, lưu chữ và bản thu nhỏ của ảnh. **Không lưu tệp gốc.**
2. Gửi câu hỏi thành công: tệp gắn với hội thoại, các câu sau dùng tiếp được.
3. Xóa hội thoại: tệp bị xóa theo. Tệp không dùng bị dọn sau 24 giờ.

## An toàn

- Giới hạn dung lượng, số trang, số điểm ảnh, dung lượng sau giải nén.
- Ảnh bị xóa metadata (kể cả GPS). Khi đăng nhập tắt, ai có id tệp đều dùng được tệp đó.
- Nội dung tệp được gửi tới NVIDIA NIM để đọc ảnh và trả lời.

Thêm định dạng mới: sửa `reader.py` và `ATTACH_TYPES` trong `frontend/src/lib/constants.js` (có test kiểm tra hai nơi khớp nhau).
