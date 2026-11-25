# docker/

| File | Chức năng |
| :-- | :-- |
| [entrypoint.sh](entrypoint.sh) | Điểm vào của image backend |

| Lệnh | Việc làm |
| :-- | :-- |
| `api` | Chạy Uvicorn với `WEB_CONCURRENCY` worker |
| `worker` | Chạy `python -m src.worker` |
| `migrate` | Chạy `alembic upgrade head` |
| khác | Chạy nguyên lệnh đó |

Trước đó script kéo mô hình từ S3 nếu có `MODELS_S3_PREFIX` (chưa kiểm), và dọn thư mục metric của Prometheus. Image không chứa torch và không đóng gói mô hình: gắn `data/models/onnx` vào container.
