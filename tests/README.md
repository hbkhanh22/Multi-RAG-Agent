# tests/

```bash
python -m pytest -q                         # toàn bộ
python -m pytest -q -m "not models"         # không cần mô hình cục bộ (CI)
python -m pytest -q tests/test_api.py       # một file
TEST_PG_URL=postgresql+psycopg://... python -m pytest -q tests/test_infra.py   # thêm test Postgres
```

| File | Kiểm tra |
| :-- | :-- |
| [conftest.py](conftest.py) | Fixture dùng chung: LLM giả, database tạm, web giả |
| [fake_mcp_server.py](fake_mcp_server.py) | MCP server giả |
| [test_api.py](test_api.py) | Endpoint, SSE, ẩn lỗi nội bộ |
| [test_images.py](test_images.py) | Tạo ảnh: kiểm duyệt, dự phòng, quyền xem, định tuyến |
| [test_attachments.py](test_attachments.py) | Đọc tệp, quyền sở hữu, vòng đời, File Agent |
| [test_contextual.py](test_contextual.py) | Ngữ cảnh trang, phiên bản index, dịch câu hỏi |
| [test_evaluation.py](test_evaluation.py) | Vòng chấm và viết lại, độ đo |
| [test_infra.py](test_infra.py) | Hàng đợi, đồng bộ, S3, pgvector, hủy, giới hạn lượt gọi |
| [test_mcp.py](test_mcp.py) | Kết nối MCP, Tool Agent |
| [test_memory.py](test_memory.py) | Hội thoại, câu hỏi nối tiếp |
| [test_multilingual.py](test_multilingual.py) | Câu hỏi Việt tìm tài liệu Anh |
| [test_parallel_expansion.py](test_parallel_expansion.py) | Mở rộng câu hỏi song song |
| [test_security.py](test_security.py) | Đăng nhập, tách dữ liệu, kiểm duyệt, che dữ liệu cá nhân |
| [test_supervisor.py](test_supervisor.py) | Chạy song song agent |
| [test_units.py](test_units.py) | Logic thuần: router, cạnh, synthesizer, SSRF |
| [test_web_search.py](test_web_search.py) | Tavily, dự phòng, đọc trang |
| [test_workflow.py](test_workflow.py) | Đồ thị đầu cuối |

LLM luôn là giả. Test dùng fixture `modules` chạy với embedding, reranker và kho vector thật, nên cần mô hình và `data/raw/tech` đã nạp; chúng được gắn nhãn `models`. Test không đụng `data/memory.db`.
