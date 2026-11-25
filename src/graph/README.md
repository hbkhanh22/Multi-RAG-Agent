# src/graph/

Đồ thị LangGraph điều phối toàn bộ việc trả lời một câu hỏi. Sơ đồ ở [README gốc](../../README.md#4-luồng-xử-lý-một-câu-hỏi).

## Các file

| File | Chức năng |
| :-- | :-- |
| [state.py](state.py) | `AgentState`: dữ liệu đi qua các bước |
| [nodes.py](nodes.py) | Hàm của từng bước; tạo một lần các agent dùng chung |
| [edges.py](edges.py) | Chọn bước kế tiếp sau router và sau evaluator |
| [workflow.py](workflow.py) | Dựng đồ thị; `run_workflow()` (một lần) và `stream_workflow()` (sự kiện) |

## Các bước

| Bước | Việc làm |
| :-- | :-- |
| `condense` | Viết lại câu hỏi nối tiếp thành câu độc lập. Bỏ qua khi không có lịch sử hoặc vừa gửi tệp mới |
| `router` | Chạy song song định tuyến, kiểm duyệt và mở rộng câu hỏi. NemoGuard đang quá hạn nên bước này chờ thêm tới `GUARD_TIMEOUT` (5 giây) |
| `sport_agent`, `tech_agent`, `general_agent`, `web_agent`, `tool_agent`, `file_agent` | Gọi một agent và stream kết quả |
| `supervisor` | Chạy nhiều agent song song |
| `synthesizer` | Gộp kết quả |
| `evaluator`, `refine` | Chấm điểm, viết lại khi chưa đạt |
| `direct_chat_agent` | Trò chuyện thường |
| `blocked` | Từ chối yêu cầu bị chặn (hiện chỉ do mẫu jailbreak) |

## Quy tắc cần nhớ

- Mỗi bước chỉ trả về **khóa nó thay đổi**. Trả cả state sẽ làm reducer cộng dồn dữ liệu.
- Tin nhắn vừa kèm tệp luôn có `files`; agent khác chỉ tham gia khi router chọn `multi`.
- `_run_agent` gắn `draft: true` cho chữ còn có thể bị thay (khi bật viết lại, hoặc với Tool Agent).
- State chỉ chứa thông tin mô tả tệp; File Agent tự đọc nội dung từ database.
- `evaluator` bỏ qua câu trả lời không có ngữ cảnh đối chiếu. Judge lỗi không chặn câu trả lời.

## Thêm một agent

1. Viết agent trong [agents/](../agents/README.md).
2. Tạo instance và hàm bước trong `nodes.py`; thêm vào `supervisor_agent` nếu cho phép chạy song song.
3. Khai báo bước và cạnh trong `workflow.py`, `edges.py`.
4. Thêm tên miền và mô tả vào prompt của `router_agent.py`.
5. Thêm nhãn trong `frontend/src/lib/constants.js`.
