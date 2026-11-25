# src/agents/

## Các file

| File | Lớp | Chức năng |
| :-- | :-- | :-- |
| [base_agent.py](base_agent.py) | `BaseAgent` | Lớp cha: gọi LLM qua NVIDIA NIM, stream, kiểm tra hủy, đếm token |
| [router_agent.py](router_agent.py) | `RouterAgent`, `fast_route` | Phân loại câu hỏi ra JSON. Câu chào đi đường tắt regex |
| [rag_agent.py](rag_agent.py) | `RAGAgent` | Một instance cho mỗi miền: truy xuất và trả lời |
| [web_agent.py](web_agent.py) | `WebAgent`, `SearchQueryWriter` | Viết câu tìm kiếm, tra web, tổng hợp |
| [file_agent.py](file_agent.py) | `FileAgent`, `PartReader` | Trả lời từ tệp đính kèm |
| [tool_agent.py](tool_agent.py) | `ToolAgent` | Vòng lặp gọi công cụ MCP |
| [supervisor_agent.py](supervisor_agent.py) | `SupervisorAgent` | Chạy song song các agent, không gọi LLM |
| [synthesizer_agent.py](synthesizer_agent.py) | `SynthesizerAgent` | Gộp câu trả lời và nguồn |

Agent trò chuyện (`ChatAgent`) nằm trong [graph/nodes.py](../graph/nodes.py); Judge và Refiner nằm trong [evaluation/judge.py](../evaluation/judge.py).

## Giao diện chung

`run(input_data)` nhận `query` và tùy agent: `history`, `expanded_queries`, `attachments`, `on_token`, `on_event`. Agent trả lời trả về:

| Khóa | Ý nghĩa |
| :-- | :-- |
| `success`, `answer` | Kết quả |
| `sources` | Nguồn được trích dẫn trong câu trả lời (Web Agent: mọi kết quả tìm được) |
| `contexts` | Mọi đoạn LLM đã thấy, đánh số như `[Source N]`, để Judge đối chiếu |
| `retrieved_count`, `reranked_count` | Số liệu truy xuất |

## Cách hoạt động

**Router.** JSON lỗi thì dùng `general`. Miền `tools` chỉ có khi MCP đã kết nối, miền `files` chỉ có khi hội thoại có tệp.

**RAG.** Mở rộng câu hỏi, tìm hybrid, rerank, lọc theo ngưỡng. Không có đoạn liên quan thì tìm miền khác, rồi Internet; vẫn không có thì trả lời bằng kiến thức chung và nói rõ điều đó.

**Web.** Phân biệt "không có kết quả" với "dịch vụ tìm kiếm không khả dụng". Prompt cấm suy đoán tỷ số, giá, ngày.

**File.**

| Trường hợp | Cách xử lý |
| :-- | :-- |
| Vừa ngân sách `ATTACHMENT_CONTEXT_CHARS` | Đưa toàn bộ cho LLM |
| Dài, câu hỏi tổng quan | Mỗi phần được rút gọn song song rồi gộp |
| Dài, câu hỏi cụ thể | BM25 chọn đoạn, luôn kèm đoạn đầu |
| Có ảnh | Model thị giác xem lại ảnh theo câu hỏi |

**Tool.** Tối đa `MCP_MAX_ITERATIONS` vòng; kết quả công cụ là dữ liệu không tin cậy.

**Synthesizer.** Một agent thành công: giữ nguyên câu trả lời. Từ hai agent: đánh lại số nguồn cho khỏi trùng, rồi gộp bằng một lần gọi LLM.
