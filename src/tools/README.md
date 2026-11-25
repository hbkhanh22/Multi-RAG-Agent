# src/tools/

Tìm kiếm web cho Web Agent và cho bước dự phòng của RAG Agent.

| File | Chức năng |
| :-- | :-- |
| [web_search.py](web_search.py) | `WebSearchTool`: Tavily, dự phòng `ddgs`, có thử lại. Hàm chặn URL nội bộ dùng chung với MCP |
| [page_reader.py](page_reader.py) | Tải trang kết quả, lấy phần chữ liên quan tới câu hỏi |

- `[]` nghĩa là đã tìm nhưng không có kết quả; `SearchUnavailable` nghĩa là không nhà cung cấp nào trả lời. Hai trường hợp này được báo cho người dùng khác nhau.
- `ddgs` hay bị chặn, nên đặt `TAVILY_API_KEY`.
- **Chống SSRF:** chỉ tải địa chỉ công khai, kiểm tra lại ở mỗi lần chuyển hướng và ở địa chỉ thực tế đã kết nối.
- Trang dựng bằng JavaScript cho ít hoặc không có chữ.
