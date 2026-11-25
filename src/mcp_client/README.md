# src/mcp_client/

Kết nối các MCP server khai báo trong [config/mcp_servers.json](../../config/mcp_servers.json) và cho agent gọi công cụ.

| File | Chức năng |
| :-- | :-- |
| [manager.py](manager.py) | `MCPManager`: một event loop nền giữ mọi kết nối, cung cấp hàm đồng bộ `tool_specs()`, `call_tool()`, `status()` |

- Kết nối diễn ra lúc API khởi động. Server lỗi bị bỏ qua, không ảnh hưởng server khác.
- Chỉ công cụ trong `allowed_tools` được đưa cho model; `filesystem` chỉ có công cụ đọc và giới hạn trong `data/mcp_workspace`.
- Tham số `url` trỏ tới địa chỉ nội bộ bị chặn. Kết quả bị cắt ở `MCP_MAX_OUTPUT_CHARS`.
- Cần `npx` hoặc `uvx` trong PATH. Docker Compose đặt `MCP_ENABLED=false`.
