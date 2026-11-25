# deploy/

| Mục | Chức năng |
| :-- | :-- |
| [prometheus/prometheus.yml](prometheus/prometheus.yml) | Thu metric của API mỗi 15 giây |
| [grafana/dashboards/multi-rag.json](grafana/dashboards/multi-rag.json) | Dashboard Grafana. **Chưa chạy được ở đây** (không tải được image) |
| [grafana/provisioning/](grafana/provisioning) | Khai báo datasource và dashboard cho Grafana. Chưa kiểm |

Liên quan: [docker-compose.yml](../docker-compose.yml), [Dockerfile](../Dockerfile), [docker/](../docker/README.md), [CI](../.github/workflows/ci.yml).

## Vận hành bằng Docker Compose

```bash
docker compose up -d --build                  # khởi động, http://localhost:8080
docker compose ps                             # api phải "healthy", migrate "Exited (0)"
docker compose logs -f worker                 # theo dõi nạp tài liệu
docker compose --profile monitoring up -d     # Prometheus :9090; Grafana :3000 chưa kiểm
docker compose stop                           # dừng, giữ dữ liệu
docker compose down                           # xóa container, giữ dữ liệu
docker compose down -v                        # XÓA CẢ DATABASE
```

Thứ tự: `postgres` sẵn sàng, `migrate` chạy xong, rồi `api` và `worker`. Lần đầu nạp tài liệu mất vài phút.

## Dev và production

| | Chạy trực tiếp | Docker Compose |
| :-- | :-- | :-- |
| Database | SQLite | PostgreSQL |
| Vector | Chroma | pgvector |
| Mô hình | ONNX nếu đã xuất, không thì torch | ONNX int8 |
| Worker | Luồng trong API | Container riêng |
| MCP | Bật | Tắt |

## Tùy chỉnh

| Nhu cầu | Cách làm |
| :-- | :-- |
| Không vào được Docker Hub | `POSTGRES_IMAGE=ghcr.io/tensorchord/vchord-postgres:pg17-v0.5.3`, `WEB_DIST=prebuilt` |
| Đổi cổng, số worker | `WEB_PORT`, `WEB_CONCURRENCY` (mỗi worker nạp riêng mô hình, khoảng 1.3 GB RAM) |

Chưa kiểm: S3, MinIO, Postgres quản lý sẵn trên cloud.

Chưa nên mở ra Internet. Cần trước: xử lý các vấn đề trong [src/auth](../src/auth/README.md#vấn-đề-chưa-sửa), bật `AUTH_ENABLED`, đặt `JWT_SECRET`, `ADMIN_TOKEN`, dùng HTTPS với `COOKIE_SECURE=true`, và sao lưu Postgres.

## CI

[ci.yml](../.github/workflows/ci.yml): test không cần mô hình, migration chạy hai lần trên pgvector, build image (đẩy lên GHCR ở nhánh `main`), kiểm tra file compose. Job test và build của commit `5e8eb30` đã chạy xanh; thay đổi chưa commit chưa qua CI.
