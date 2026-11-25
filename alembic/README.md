# alembic/

Migration schema, dùng cho PostgreSQL và SQLite. Cấu hình: [alembic.ini](../alembic.ini).

| File | Chức năng |
| :-- | :-- |
| [env.py](env.py) | Lấy URL từ `settings`, chạy migration |
| [script.py.mako](script.py.mako) | Mẫu migration mới |
| [versions/0001_initial.py](versions/0001_initial.py) | Hội thoại, tài liệu, hàng đợi, đoạn pgvector |
| [versions/0002_auth.py](versions/0002_auth.py) | Tài khoản, refresh token |
| [versions/0003_attachments.py](versions/0003_attachments.py) | Tệp đính kèm |
| [versions/0004_images.py](versions/0004_images.py) | Ảnh do agent vẽ |

```bash
alembic upgrade head                       # áp dụng
alembic revision -m "mo ta thay doi"       # tạo mới
```

Docker Compose tự chạy `upgrade head` qua dịch vụ `migrate`. Migration cần chạy lại được nhiều lần (kiểm tra bảng đã có chưa), vì ứng dụng có thể đã tự tạo bảng. Đổi schema phải sửa cả [src/storage/db.py](../src/storage/db.py).
