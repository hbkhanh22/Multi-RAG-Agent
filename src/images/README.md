# src/images/

| File | Chức năng |
| :-- | :-- |
| [generator.py](generator.py) | Gọi FLUX trên NVIDIA NIM, chuyển sang model dự phòng khi lỗi, kiểm tra ảnh trả về |
| [store.py](store.py) | Lưu ảnh trong database theo chủ sở hữu; ảnh của lượt không được lưu bị xóa sau một ngày |

Việc duyệt nội dung nằm ở [image_agent.py](../agents/image_agent.py): không gọi thẳng `generate()` với câu của người dùng.
