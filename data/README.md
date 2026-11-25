# data/

| Mục | Nội dung | Trên git |
| :-- | :-- | :-- |
| `raw/sport/` | Luật bóng đá IFAB, FIFA, VFF (có bản quyền) | Không |
| `raw/tech/` | Tài liệu công nghệ, AI agent | Có |
| `raw/general/` | Kiến thức chung | Chưa có tài liệu |
| `models/` | Mô hình tải sẵn; `models/onnx/` là bản int8 | Không |
| `vectorstores/` | Chroma, chỉ dùng khi dev | Không |
| `memory.db*` | SQLite, chỉ dùng khi dev | Không |
| `mcp_workspace/` | Thư mục MCP `filesystem` được đọc | Không |

Thêm tài liệu: chép file `.pdf`, `.txt`, `.md`, `.docx` vào `raw/<miền>/`. File `.md` đặt trong `raw/` cũng bị nạp, nên **không để README trong đó**.

Với Docker, dữ liệu hội thoại và vector nằm trong volume `pgdata`, không nằm ở đây.
