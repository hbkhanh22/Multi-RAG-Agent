# frontend/src/

| Mục | Chức năng |
| :-- | :-- |
| [main.jsx](main.jsx) | Điểm vào |
| [App.jsx](App.jsx) | Màn hình chat: hội thoại, gửi câu hỏi, đọc SSE, dừng, thử lại, cuộn, phím tắt |
| [styles.css](styles.css) | Toàn bộ kiểu dáng và biến màu |
| [components/](components/README.md) | Thành phần giao diện |
| [lib/](lib/README.md) | Hằng số, hook, hàm tiện ích |

## Luồng gửi câu hỏi

1. Tệp được tải lên ngay khi thêm vào ô nhập (`useAttachments`).
2. Gửi `POST /api/query/stream` kèm id tệp và `conversation_id`.
3. `token` có `draft: true` hiện trong khung "Bản nháp"; `reset` giữ bản nháp cũ tới khi có chữ mới.
4. `final` thay bản nháp bằng câu trả lời chính thức và cuộn về đầu câu trả lời.

## Đăng nhập

`AuthGate` hỏi `/api/auth/config`: nếu server tắt đăng nhập thì vào thẳng chat, nếu bật thì khôi phục phiên hoặc hiện form. Access token chỉ nằm trong bộ nhớ và tự làm mới; refresh token là cookie JavaScript không đọc được. Phiên hết hạn thì hiện form, đăng nhập lại vẫn còn đoạn chat.

`conversation_id`, giao diện và tùy chọn được lưu ở `localStorage`.
