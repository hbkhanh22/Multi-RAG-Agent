# frontend/src/lib/

| File | Chức năng |
| :-- | :-- |
| [api.js](api.js) | Phiên đăng nhập và `apiFetch` (gắn token, tự làm mới). Mọi lời gọi API cần quyền phải dùng hàm này |
| [useTheme.js](useTheme.js) | Giao diện sáng, tối và media query |
| [constants.js](constants.js) | Địa chỉ API, nhãn route và bước xử lý, giới hạn và loại tệp, câu hỏi gợi ý |
| [useAttachments.js](useAttachments.js) | Hook quản lý tệp: kiểm tra, tải lên, trạng thái, bỏ tệp |
| [useCopy.js](useCopy.js) | Hook sao chép vào clipboard |
| [utils.js](utils.js) | Định dạng, `localStorage` an toàn, nhóm hội thoại, lọc nguồn |

Khi backend thêm route, bước xử lý hoặc loại tệp, cập nhật `constants.js`.
