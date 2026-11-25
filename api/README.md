# api/

Lớp HTTP (FastAPI): nhận yêu cầu, kiểm tra quyền, gọi lõi ở [src/](../src/README.md). Không chứa logic agent. Tài liệu tương tác: `/docs`.

## Các file

| File | Chức năng |
| :-- | :-- |
| [main.py](main.py) | Tạo ứng dụng; request id và metric cho mọi yêu cầu; khi khởi động: xếp job đồng bộ tài liệu, nạp mô hình ở nền, kết nối MCP; `/health/*`, `/metrics` |
| [routes.py](routes.py) | Hỏi đáp, stream, hủy, hội thoại, trạng thái |
| [images.py](images.py) | `GET /api/images/{id}`: ảnh đã vẽ, chỉ chủ sở hữu xem được |
| [attachments.py](attachments.py) | Tải lên và xóa tệp đính kèm trong chat |
| [documents.py](documents.py) | Quản lý kho tài liệu (quyền admin) |
| [auth.py](auth.py) | Đăng ký, đăng nhập, làm mới phiên, đăng xuất. Khi `AUTH_ENABLED=false`, các endpoint này trả 404, trừ `config`, `me` và `logout` |
| [deps.py](deps.py) | Giới hạn lượt gọi, kiểm tra `ADMIN_TOKEN`, xác định người dùng |
| [schemas.py](schemas.py) | Kiểu dữ liệu vào và ra |

## Endpoint

| Nhóm | Endpoint |
| :-- | :-- |
| Hỏi đáp | `POST /api/query`, `/api/query/stream`, `/api/query/batch`, `/api/chat` (như `/query`), `/api/query/{request_id}/cancel`, `/api/route` (chỉ xem định tuyến) |
| Hội thoại | `GET /api/conversations`, `GET` và `DELETE /api/conversations/{id}` |
| Tệp đính kèm | `POST /api/attachments`, `DELETE /api/attachments/{id}` (chỉ khi chưa dùng), `GET /api/attachments/types` |
| Tài liệu (cần `ADMIN_TOKEN`) | `GET` và `POST /api/documents`, `DELETE /api/documents/{id}`, `POST /api/ingest/sync`, `GET /api/jobs`, `GET /api/jobs/{id}` |
| Vận hành | `/health/live`, `/health/ready` (503 khi chưa sẵn sàng), `/metrics`, `/api/health`, `/api/stats`, `/api/domain/{domain}/stats`, `/api/mcp/status` |

Khi bật đăng nhập, gửi `Authorization: Bearer <access token>`; nhóm tài liệu cần tài khoản admin hoặc `$ADMIN_TOKEN`. Vẫn công khai: `/api/health`, `/api/stats`, `/api/mcp/status`, `/api/attachments/types`, `/docs`, `/metrics` (nginx chặn `/metrics`).

Khi tắt đăng nhập, mọi endpoint trừ nhóm tài liệu gọi được tự do; để trống `ADMIN_TOKEN` thì nhóm đó cũng mở.

Giới hạn lượt gọi chỉ áp dụng cho `/query*`, `/chat` và tải tệp, tính riêng từng tiến trình.

## Ví dụ

```bash
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -d '{"query": "Model Context Protocol là gì?", "include_metadata": true}'

# hỏi về một tệp: tải lên trước, rồi gửi id
curl -X POST http://localhost:8000/api/attachments -F file=@hoa-don.jpg
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" \
  -d '{"query": "Tổng tiền là bao nhiêu?", "attachment_ids": ["<id>"]}'

# thêm tài liệu vào kho (202: worker nạp ở nền)
curl -X POST http://localhost:8000/api/documents -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F domain=tech -F file=@tai-lieu.pdf
```

Gửi `conversation_id` để tiếp tục hội thoại; bỏ trống thì server tạo mới và trả id.

## Sự kiện của `/api/query/stream`

| `type` | Ý nghĩa |
| :-- | :-- |
| `start` | `request_id` (dùng để hủy), `conversation_id` |
| `stage` | Bước đang chạy: `understanding`, `routing`, `retrieving`, `reading_files`, `synthesizing`, `evaluating`, `refining` |
| `route` | Agent được chọn |
| `tool` | Công cụ MCP bắt đầu hoặc kết thúc |
| `token` | Chữ đang sinh. `draft: true` nghĩa là bản nháp, có thể bị thay |
| `evaluation` | Kết quả chấm điểm |
| `reset` | Bản nháp sắp được thay bằng bản viết lại |
| `final` | Toàn bộ câu trả lời, nguồn và số liệu. **Đây là câu trả lời chính thức** |
| `cancelled`, `error` | Thay cho `final` khi bị dừng hoặc lỗi |

Đóng kết nối cũng dừng việc sinh câu trả lời.

## Quy ước

- Handler gọi LLM là `def` thường, chạy trong thread pool, nên không chặn event loop.
- Lỗi nội bộ chỉ trả mã tra cứu, chi tiết nằm trong log.
- Chỉ lượt hỏi đáp thành công được lưu; câu bị chặn không lưu.
