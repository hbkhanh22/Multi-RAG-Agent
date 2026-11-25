# config/

| File | Chức năng |
| :-- | :-- |
| [settings.py](settings.py) | Toàn bộ cấu hình (Pydantic), đọc từ biến môi trường và `.env`. Đường dẫn tương đối tính từ thư mục gốc dự án |
| [mcp_servers.json](mcp_servers.json) | Các MCP server và danh sách công cụ được phép dùng |

Dùng trong code: `from config.settings import settings`. Biến môi trường là tên trường viết hoa. Mẫu: [.env.example](../.env.example).

Bảng dưới là các biến hay dùng; danh sách đầy đủ nằm trong `settings.py`.

## LLM

| Biến | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `NVIDIA_API_KEY` | trống | Bắt buộc |
| `DEFAULT_LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Sinh, tổng hợp, chấm câu trả lời |
| `FAST_LLM_MODEL`, `ROUTER_LLM_MODEL`, `CHAT_LLM_MODEL` | `nvidia/nemotron-3.5-bolt-e4b` | Việc nhỏ, định tuyến, trò chuyện |
| `VISION_LLM_MODEL`, `VISION_FALLBACK_MODEL` | `nemotron-3.5-super-vl-120b-a12b`, `nemotron-3-nano-omni-30b-a3b-reasoning` | Đọc ảnh và trang scan |
| `LLM_ENABLE_THINKING` | `false` | Bật có thể khiến câu trả lời rỗng vì hết ngân sách token |

## Truy xuất

| Biến | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `EMBEDDING_MODEL`, `RERANKER_MODEL` | `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3` | Mô hình cục bộ |
| `MODEL_BACKEND` | `auto` | `onnx` (đã kiểm), `torch` (chưa kiểm lại), hoặc `auto` |
| `VECTOR_BACKEND` | `chroma` | `chroma` hoặc `pgvector` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | `512`, `64` | Tính theo ký tự |
| `TOP_K_RETRIEVAL`, `RERANK_MAX_CANDIDATES`, `TOP_K_RERANK` | `10`, `8`, `4` | Số đoạn qua từng bước |
| `VECTOR_WEIGHT`, `BM25_WEIGHT` | `0.6`, `0.4` | Trọng số khi hợp nhất |
| `RELEVANCE_THRESHOLD` | `0.05` | Điểm rerank tối thiểu |
| `QUERY_EXPANSION_CROSSLINGUAL`, `PARALLEL_EXPANSION` | `true` | Dịch Việt và Anh; chạy song song với router |
| `INDEX_VERSION`, `CONTEXTUAL_CHUNKS` | trống, `false` | Dựng index song song; gắn ngữ cảnh trang vào đoạn (đang tắt) |

## Tệp đính kèm, web, MCP

| Biến | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `ATTACHMENT_MAX_MB`, `ATTACHMENT_MAX_FILES` | `20`, `5` | Mỗi tệp; mỗi tin nhắn |
| `IMAGE_ENABLED`, `IMAGE_MODEL`, `IMAGE_FALLBACK_MODEL` | `true`, FLUX.2 klein, FLUX.1 schnell | Tạo ảnh |
| `IMAGE_LIMIT_PER_HOUR` | `20` | Số ảnh mỗi giờ mỗi người |
| `ATTACHMENT_MAX_SCAN_PAGES` | `10` | Số trang scan được đọc mỗi tệp |
| `ATTACHMENT_MAX_CHARS`, `ATTACHMENT_CONTEXT_CHARS` | `200000`, `24000` | Chữ giữ lại mỗi tệp; chữ gửi cho LLM mỗi câu hỏi |
| `TAVILY_API_KEY`, `TAVILY_SEARCH_DEPTH` | trống, `basic` | Trống thì chỉ dùng `ddgs` |
| `WEB_READ_PAGES`, `LOCAL_UTC_OFFSET` | `3`, `7` | Số trang đọc thêm; múi giờ của "hôm nay" |
| `MCP_ENABLED`, `MCP_MAX_ITERATIONS` | `true`, `5` | Bật MCP; số vòng gọi công cụ |

## Tự đánh giá và bộ nhớ

| Biến | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `SELF_EVAL_ENABLED` | `true` | Chấm câu trả lời |
| `SELF_EVAL_MIN_SCORE`, `SELF_EVAL_MAX_ROUNDS` | `4`, `1` | Ngưỡng đạt (1–5); số vòng viết lại |
| `JUDGE_LLM_MODEL` | trống | Trống thì dùng `DEFAULT_LLM_MODEL` |
| `MEMORY_MAX_TURNS`, `MEMORY_MAX_CHARS` | `6`, `4000` | Lịch sử đưa vào prompt |

## Hạ tầng

| Biến | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `DATABASE_URL` | trống | Trống thì dùng SQLite `data/memory.db` |
| `DOCUMENT_STORAGE` | `local` | `local` (`data/raw`). `s3` mới thử với S3 giả lập |
| `INGEST_WORKER` | `inline` | `inline` (trong API) hoặc `external` (tiến trình riêng) |
| `WATCH_DOCUMENTS` | `true` | Tự nạp lại khi `data/raw` đổi |
| `RATE_LIMIT_PER_MINUTE`, `MAX_UPLOAD_MB` | `30`, `50` | `0` là tắt giới hạn; cỡ tài liệu tải lên kho |
| `CORS_ORIGINS` | localhost | Danh sách origin, cách nhau bằng dấu phẩy |
| `LOG_JSON`, `METRICS_ENABLED` | `false`, `true` | Log JSON; metric Prometheus |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | trống | Bật tracing khi có cả hai. `LANGFUSE_HOST` phải đúng vùng của tài khoản |

## Bảo mật (xem [src/auth](../src/auth/README.md))

| Biến | Mặc định | Ý nghĩa |
| :-- | :-- | :-- |
| `AUTH_ENABLED` | `false` | Bật tài khoản, màn hình đăng nhập và phân quyền |
| `JWT_SECRET` | trống | Cần đặt khi bật đăng nhập, nếu không phiên mất khi khởi động lại |
| `ADMIN_EMAILS`, `ADMIN_TOKEN` | trống | `ADMIN_EMAILS`: thành admin khi đăng nhập Google, không tự đăng ký bằng mật khẩu được. `ADMIN_TOKEN`: token dịch vụ. Tắt đăng nhập và để trống `ADMIN_TOKEN` thì API tài liệu mở |
| `ALLOW_REGISTRATION`, `COOKIE_SECURE` | `true`, `false` | Tự đăng ký; đặt `true` khi chạy HTTPS |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | trống | Đăng nhập Google, chưa chạy thật |
| `GUARDRAILS_ENABLED`, `GUARD_TIMEOUT` | `true`, `5` | Chặn jailbreak theo mẫu và gọi NemoGuard (hiện luôn quá hạn); thời gian chờ (giây) |
| `PII_REDACTION` | `true` | Che dữ liệu cá nhân trong log và trace |

## Thêm MCP server

Thêm mục vào `servers` trong `mcp_servers.json` với `command`, `args`, `env`, `allowed_tools`, `enabled`, rồi khởi động lại API. `${WORKSPACE}` và `${BIEN_MOI_TRUONG}` được thay thế. Bỏ `allowed_tools` nghĩa là cho phép **mọi** công cụ của server, nên luôn khai báo với server có quyền ghi.
