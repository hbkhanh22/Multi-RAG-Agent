# src/storage/

| File | Chức năng |
| :-- | :-- |
| [db.py](db.py) | Schema SQLAlchemy và kết nối (SQLite hoặc PostgreSQL) |
| [objects.py](objects.py) | Kho tài liệu gốc: `LocalStorage` (`data/raw`); `S3Storage` mới thử với S3 giả lập |
| [jobs.py](jobs.py) | Hàng đợi job trên bảng `ingest_jobs` |

## Các bảng

| Bảng | Nội dung |
| :-- | :-- |
| `users`, `refresh_tokens` | Tài khoản và phiên đăng nhập |
| `conversations`, `messages` | Hội thoại |
| `attachments` | Nội dung tệp đính kèm |
| `generated_images` | Ảnh do agent vẽ, xóa cùng hội thoại |
| `documents` | Tài liệu đã nạp và trạng thái, theo từng index |
| `ingest_jobs` | Hàng đợi |
| `chunks` | Đoạn và vector, chỉ có trên PostgreSQL |

## Cần biết

- SQLite tự tạo bảng. Với PostgreSQL, thay đổi schema phải có migration trong [alembic/](../../alembic/README.md).
- Hàng đợi dùng `FOR UPDATE SKIP LOCKED`, nhiều worker không lấy trùng job. Job lỗi thử lại tối đa `INGEST_MAX_ATTEMPTS`; job treo quá 30 phút được xếp lại khi worker khởi động.
- Tên tệp tải lên được làm sạch; thao tác xóa local chỉ trong `data/raw`.
