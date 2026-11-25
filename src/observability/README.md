# src/observability/

| File | Chức năng |
| :-- | :-- |
| [metrics.py](metrics.py) | Metric Prometheus tại `/metrics`; gộp được nhiều worker |
| [logging.py](logging.py) | Log kèm request id, dạng chữ hoặc JSON, che dữ liệu cá nhân |
| [tracing.py](tracing.py) | Langfuse, **chưa cấu hình, chưa chạy thật** |

| Nhóm metric | Ví dụ |
| :-- | :-- |
| HTTP | `rag_http_requests_total`, `rag_http_request_seconds` |
| Câu hỏi | `rag_queries_total`, `rag_query_seconds`, `rag_time_to_first_token_seconds`, `rag_active_streams` |
| LLM | `rag_llm_calls_total`, `rag_llm_tokens_total`, `rag_llm_call_seconds` |
| Chất lượng | `rag_self_eval_total` |
| An toàn | `rag_guardrail_total`, `rag_prompt_injection_total`, `rag_auth_total`, `rag_rate_limited_total` |
| Nạp tài liệu, web | `rag_ingest_jobs_total`, `rag_ingested_documents_total`, `rag_web_search_total` |

Request id trong log trùng với header `X-Request-ID`. Lỗi ở lớp này không làm hỏng câu trả lời.
