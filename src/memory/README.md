# src/memory/

| File | Chức năng |
| :-- | :-- |
| [store.py](store.py) | `ConversationStore`: tạo, đọc, liệt kê, xóa hội thoại; lưu một lượt hỏi đáp trong một transaction |
| [history.py](history.py) | Chọn lịch sử vừa ngân sách, nhận biết câu hỏi nối tiếp, prompt viết lại câu hỏi |

- Lịch sử vào prompt giới hạn bởi `MEMORY_MAX_TURNS` và `MEMORY_MAX_CHARS`. Thẻ `[Source N]` cũ bị bỏ để không lẫn với lượt mới.
- Tin nhắn kèm tệp được ghi thêm tên tệp trong lịch sử, để hiểu câu hỏi sau.
- Khi bật đăng nhập, mỗi người chỉ thấy hội thoại của mình; khi tắt, mọi hội thoại dùng chung.
- Nhận biết câu nối tiếp dựa trên độ dài và từ khóa, rồi mới gọi model nhỏ để viết lại.
