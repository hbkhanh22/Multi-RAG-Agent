# Hệ Thống Multi-RAG Agent (Multi-Agent Retrieval-Augmented Generation)

> Hệ thống Multi-RAG Agent thông minh điều phối nhiều Agent chuyên biệt theo từng miền tri thức (*Sport*, *Tech*, *General*), kết hợp **LangGraph**, **Chroma Vector Store**, **Hybrid Retrieval (Vector + BM25)**, **Cross-Encoder Reranker**, và **LLMs (NVIDIA NIM)** để giải quyết các truy vấn phức tạp với độ chính xác và khả năng trích dẫn nguồn cao.

---

## 📑 Mục lục
1. [Tổng Quan Kiến Trúc (Architecture Overview)](#1-tổng-quan-kiến-trúc-architecture-overview)
2. [Chi Tiết Pipeline Xử Lý Dữ Liệu (Ingestion Pipeline)](#2-chi-tiết-pipeline-xử-lý-dữ-liệu-ingestion-pipeline)
3. [Luồng Hoạt Động Của Hệ Thống RAG (RAG Execution Workflow)](#3-luồng-hoạt-động-của-hệ-thống-rag-rag-execution-workflow)
4. [Bên Trong Từng Thành Phần RAG (Internal RAG Mechanics)](#4-bên-trong-từng-thành-phần-rag-internal-rag-mechanics)
   - [Router Agent & Direct Chat Routing](#41-router-agent--direct-chat-routing)
   - [Cơ Chế Nội Bộ Pipeline RAG (Expansion, Retrieval, Reranking)](#42-cơ-chế-nội-bộ-của-pipeline-rag-retrieval-augmented-generation-pipeline)
   - [Supervisor Agent & Synthesizer Agent (Đa tác tử & Tối ưu hóa)](#43-supervisor-agent--synthesizer-agent-phối-hợp-đa-miền--tối-ưu-hóa)
   - [Web Search Tool & Live Internet Agent (Thông tin thời gian thực)](#44-web-search-tool--live-internet-agent-thông-tin-thời-gian-thực)
   - [Đánh Giá Chất Lượng & Vòng Lặp Tự Sửa](#45-đánh-giá-chất-lượng--vòng-lặp-tự-sửa-evaluation--self-correction)
5. [Cấu Trúc Thư Mục Dự Án](#5-cấu-trúc-thư-mục-dự-án)
6. [Cấu Hình Hệ Thống (Configuration)](#6-cấu-hình-hệ-thống-configuration)
7. [Hướng Dẫn Cài Đặt & Khởi Chạy](#7-hướng-dẫn-cài-đặt--khởi-chạy)
8. [API Endpoints & Cách Sử Dụng](#8-api-endpoints--cách-sử-dụng)
9. [Khởi Chạy Trực Tiếp Qua Python Script](#9-khởi-chạy-trực-tiếp-qua-python-script)
10. [Tiến Độ Dự Án](#10-tiến-độ-dự-án)

---

## 1. Tổng Quan Kiến Trúc (Architecture Overview)

Hệ thống được thiết kế theo mô hình **Multi-Agent Directed Acyclic Graph (DAG)** xây dựng trên nền tảng **LangGraph**. Thay vì sử dụng một kho vector store đơn lẻ khổng lồ dễ gây nhiễu, hệ thống phân tách tri thức thành các miền độc lập (*Sport*, *Tech*, *General*).

```mermaid
flowchart LR
    U(["👤 Người dùng"]) --> FE["React + Vite UI<br/>(SSE streaming, lịch sử chat)"]
    FE -->|"POST /api/query/stream"| API["FastAPI<br/>api/routes.py"]
    API --> MEM[("SQLite<br/>data/memory.db")]
    API --> G

    subgraph G["LangGraph workflow"]
        direction TB
        C["condense"] --> R["Router Agent"]
        R --> A1["Sport / Tech / General<br/>RAG Agents"]
        R --> A2["Web Agent"]
        R --> A3["Tool Agent"]
        R --> SV["Supervisor"]
        R --> DC["Direct Chat"]
        A1 & A2 & A3 & SV --> SY["Synthesizer"]
    end

    A1 --> VS[("ChromaDB + BM25<br/>sport / tech / general")]
    A1 --> RR["Reranker<br/>bge-reranker-v2-m3"]
    A2 --> DDG["DuckDuckGo"]
    A3 --> MCP["MCP servers<br/>filesystem, fetch"]
    G -.->|"gọi LLM"| NIM["NVIDIA NIM<br/>Nemotron"]
    ING["Auto-ingest<br/>(luồng nền khi khởi động)"] --> VS
```

---

## 2. Chi Tiết Pipeline Xử Lý Dữ Liệu (Ingestion Pipeline)

Quy trình nạp tài liệu biến đổi dữ liệu thô (PDF, DOCX, TXT, Markdown) thành các vector embedding và lưu trữ vào ChromaDB theo từng domain.

```mermaid
flowchart TD
    RAW["data/raw/sport · tech · general<br/>(.pdf .txt .md .docx)"] --> SCAN["Quét file khi server khởi động"]
    SCAN --> CHK{"So với manifest<br/>(mtime, size)"}
    CHK -->|"không đổi"| SKIP["Bỏ qua"]
    CHK -->|"mới / đã sửa"| LOAD["Đọc file<br/>PyPDF · Text · Docx2txt"]
    LOAD --> CHUNK["Chia chunk<br/>512 ký tự, overlap 64"]
    CHUNK --> DEL["Xóa chunk cũ của file<br/>(nếu là bản sửa)"]
    DEL --> EMB["Embedding bge-m3<br/>(1024 chiều, chuẩn hóa)"]
    EMB --> STORE[("Chroma<br/>data/vectorstores/&lt;domain&gt;_store/bge-m3")]
    EMB --> MAN["Cập nhật manifest"]
    SCAN -->|"file đã bị xóa khỏi đĩa"| RM["Xóa chunk + mục manifest"]
    RM --> STORE
```

### Các bước xử lý trong Ingestion:
1. **Quét tự động (Auto-Scan & Incremental Delta Check):** Khi khởi động server FastAPI (`lifespan`), `src/ingestion/auto_ingest.py` tự động quét các thư mục `data/raw/{domain}`.
2. **Theo dõi Manifest (`.ingested_manifest.json`):** Lưu trữ đường dẫn, thời gian sửa đổi (`mtime`) và kích thước (`size`) của từng file. Nếu file chưa từng nạp hoặc đã bị chỉnh sửa, hệ thống mới tiến hành nạp, giúp tránh trùng lặp dữ liệu và tiết kiệm thời gian khởi động.
3. **Phân rã văn bản (Chunking):** Sử dụng `RecursiveCharacterTextSplitter` với:
   - `chunk_size = 512`: Độ dài tối ưu ngữ nghĩa cho các mô hình embedding dạng nhỏ/nhanh.
   - `chunk_overlap = 64` (hoặc `128`): Giữ tính liên tục của ngữ cảnh giữa các đoạn cắt liền kề.
   - `add_start_index = True`: Lưu vị trí index để phục vụ việc trích xuất toạ độ dữ liệu.
4. **Vector Embedding:** Sử dụng mô hình `BAAI/bge-m3` (đa ngôn ngữ, dimension = 1024, vector chuẩn hóa) qua HuggingFace để mã hóa nội dung chunk thành vector không gian.
5. **ChromaDB Multi-Collection:** Mỗi domain sở hữu một thư mục lưu trữ độc lập:
   - Sport: `data/vectorstores/sport_store` (collection: `sport_collection`)
   - Tech: `data/vectorstores/tech_store` (collection: `tech_collection`)
   - General: `data/vectorstores/general_store` (collection: `general_collection`)

---

## 3. Luồng Hoạt Động Của Hệ Thống RAG (RAG Execution Workflow)

Mỗi khi người dùng gửi một câu hỏi qua API `/api/query`, luồng xử lý diễn ra chi tiết qua các giai đoạn:

```mermaid
flowchart TD
    START(["Câu hỏi + conversation_id"]) --> HIS["Nạp lịch sử hội thoại<br/>(SQLite)"]
    HIS --> CON{"condense:<br/>câu hỏi nối tiếp?"}
    CON -->|"có"| RW["Viết lại thành câu độc lập<br/>(model nhỏ)"]
    CON -->|"không"| RT
    RW --> RT["router"]
    RT -->|"direct_chat"| DC["direct_chat_agent"] --> END1(["Trả lời, stream token"])
    RT -->|"sport"| SP["sport_agent"]
    RT -->|"tech"| TE["tech_agent"]
    RT -->|"general"| GE["general_agent"]
    RT -->|"web"| WE["web_agent"]
    RT -->|"tools"| TO["tool_agent"]
    RT -->|"multi"| SV["supervisor<br/>(chạy song song)"]
    SP & TE & GE & WE & TO & SV --> SY["synthesizer"]
    SY --> EV{"evaluator<br/>(LLM judge)"}
    EV -->|"chưa đạt, còn lượt"| RF["refine<br/>(gửi nhận xét cho LLM)"]
    RF --> EV
    EV -->|"đạt / hết lượt / không có context"| SAVE["Lưu lượt hỏi đáp<br/>(chỉ khi thành công)"]
    SAVE --> END2(["final: câu trả lời + nguồn + themes + evaluation"])
```

Trong lúc chạy, server gửi các sự kiện SSE `stage` (understanding / routing / retrieving / synthesizing / evaluating / refining), `route`, `tool`, `token`, `evaluation`, `reset` (bỏ bản nháp đã stream trước khi stream bản đã sửa) và cuối cùng là `final` hoặc `error`.

---

## 4. Bên Trong Từng Thành Phần RAG (Internal RAG Mechanics)

### 4.1. Router Agent & Direct Chat Routing

```mermaid
flowchart TD
    Q["Câu hỏi"] --> FAST{"Regex câu chào / cảm ơn /<br/>'bạn là ai' (≤ 60 ký tự)?"}
    FAST -->|"khớp"| DC["direct_chat<br/>(không gọi LLM)"]
    FAST -->|"không"| LLM["LLM nhỏ phân loại<br/>→ JSON"]
    LLM --> PARSE{"Parse JSON<br/>(thẳng / trong khối code / trích đoạn JSON)"}
    PARSE -->|"lỗi"| GEN["general (mặc định)"]
    PARSE -->|"ok"| ROUTE{"route"}
    ROUTE -->|"sport · tech · general · web"| ONE["1 agent"]
    ROUTE -->|"tools"| MCPQ{"MCP đã kết nối?"}
    MCPQ -->|"có"| TOOLS["tool_agent"]
    MCPQ -->|"không"| WEB["đổi sang web"]
    ROUTE -->|"multi"| MULTI["lọc domain hợp lệ<br/>≥ 2 → supervisor · 1 → agent đó"]
```

**Output JSON mẫu của Router:**
```json
{
  "route": "tech",
  "primary_domain": "tech",
  "all_domains": ["tech"],
  "domain_scores": {"tech": 0.95, "sport": 0.0, "general": 0.05, "direct_chat": 0.0},
  "confidence": 0.95,
  "reasoning": "Câu hỏi hỏi về kiến trúc Microservices và REST API"
}
```

### 4.2. Cơ Chế Nội Bộ Của Pipeline RAG (Retrieval-Augmented Generation Pipeline)

```mermaid
flowchart TD
    Q["Câu hỏi"] --> EXP["Multi-query expansion<br/>(gốc + 2 biến thể, model nhỏ)"]
    EXP --> HY["Hybrid search mỗi biến thể<br/>vector k=10 + BM25 → weighted RRF"]
    HY --> DD["Xen kẽ thứ hạng, khử trùng<br/>giữ tối đa 8 ứng viên"]
    DD --> RR["Rerank bge-reranker-v2-m3<br/>lấy top 4"]
    RR --> T1{"điểm ≥ 0.05?"}
    T1 -->|"có"| GEN["Sinh câu trả lời<br/>kèm [Source N]"]
    T1 -->|"không"| OD["Tìm ở 2 domain còn lại<br/>(không mở rộng câu hỏi)"]
    OD --> T2{"rerank: điểm ≥ 0.05?"}
    T2 -->|"có"| GEN
    T2 -->|"không"| WEB["Tìm web (DuckDuckGo)"]
    WEB --> GEN
```

Quy trình RAG bao gồm 4 giai đoạn nối tiếp nhau được tinh chỉnh để tối đa hoá độ chính xác:
1. **Query Expansion (Mở rộng câu hỏi):**
   - **Multi-Query:** LLM sinh thêm 2 biến thể câu hỏi (mặc định chỉ bật Multi-Query) tiếp cận từ nhiều góc độ khác nhau.
   - **Cross-lingual (`QUERY_EXPANSION_CROSSLINGUAL`, mặc định bật):** biến thể đầu tiên là bản dịch Việt ↔ Anh của câu hỏi, dùng thuật ngữ chuẩn của lĩnh vực. Kho tài liệu có cả tiếng Anh (IFAB, FIFA, tài liệu tech) lẫn tiếng Việt (VFF, ĐH Duy Tân).
   - **Synonym Expansion (có sẵn, mặc định tắt để giảm số lần gọi LLM):** Bổ sung thuật ngữ đồng nghĩa theo chuyên ngành thể thao và công nghệ.
   - **HyDE (Hypothetical Document Embeddings, có sẵn nhưng mặc định tắt):** LLM sinh một đoạn giả định chứa câu trả lời khả dĩ và dùng embedding của đoạn này để truy xuất tài liệu, giúp thu hẹp khoảng cách ngữ nghĩa giữa câu hỏi ngắn và đoạn văn bản dài.
2. **Hybrid Retrieval & Deduplication (ChromaDB + BM25):**
   - **Hybrid Search:** Mỗi truy vấn chạy đồng thời *vector similarity* (k = 10) và *BM25*, rồi hợp nhất bằng *Weighted Reciprocal Rank Fusion* (`VECTOR_WEIGHT` / `BM25_WEIGHT`).
   - **Deduplication:** Khử trùng các chunk trùng lặp theo (nguồn, trang, nội dung).
   - **Relevance Threshold & Fallback:** Chunk chỉ được coi là liên quan nếu điểm rerank ≥ `RELEVANCE_THRESHOLD`. Nếu domain chính không có chunk liên quan, hệ thống thử các domain nội bộ khác; nếu vẫn không có, chuyển sang tìm kiếm web.
3. **Cross-Encoder Re-ranking:**
   - Sử dụng mô hình `BAAI/bge-reranker-v2-m3` (đa ngôn ngữ, điểm sigmoid 0..1).
   - Tính toán điểm tương đồng ngữ nghĩa sâu (deep cross-attention) giữa câu hỏi và từng chunk.
   - **Rerank cùng ngôn ngữ ([src/retrieval/lang.py](src/retrieval/lang.py)):** mỗi chunk được chấm với phiên bản câu hỏi cùng ngôn ngữ với nó (câu gốc hoặc bản dịch), nên số lần gọi model không tăng.
     - Lý do: cặp "câu hỏi tiếng Việt – đoạn tiếng Anh" chỉ được khoảng 0.02–0.05 điểm, dưới ngưỡng liên quan, kể cả khi đoạn đó chứa đúng đáp án. Cùng đoạn ấy chấm với câu hỏi tiếng Anh được 0.6–0.99.
     - Kết quả trên 24 câu eval: hit rate 0.79 → 0.96, MRR 0.75 → 0.90.
   - Lọc từ tối đa **8 ứng viên** (`RERANK_MAX_CANDIDATES`) xuống **Top 4** (`TOP_K_RERANK`), rồi bỏ các chunk dưới ngưỡng liên quan. Trên bộ eval, 6, 8 hay 10 ứng viên cho cùng hit rate; mỗi ứng viên tốn khoảng 0.28s CPU.
4. **LLM Generation:**
   - Đưa câu hỏi kèm tối đa 4 chunks ngữ cảnh tinh chọn vào LLM để sinh câu trả lời kèm thẻ trích dẫn `[Source N]`.

> **Đa ngôn ngữ:** embedding và reranker đều hỗ trợ tiếng Việt/Anh, nên câu hỏi tiếng Việt truy xuất được tài liệu tiếng Anh (điểm liên quan ≥ 0.15, câu hỏi không liên quan ≤ 0.01; ngưỡng mặc định 0.05). Mỗi embedding model có thư mục vectorstore và manifest riêng (`data/vectorstores/<domain>_store/<model>/`), đổi model sẽ tự nạp lại. Đặt model đã tải sẵn vào `data/models/<tên model>/` để chạy offline. Lần nạp đầu với CPU mất ~14 phút cho 630 chunk.

---

### 4.3. Supervisor Agent & Synthesizer Agent (Phối Hợp Đa Miền & Tối Ưu Hóa)

```mermaid
sequenceDiagram
    participant R as Router
    participant S as Supervisor
    participant A as Agent A (vd: sport)
    participant B as Agent B (vd: web)
    participant Y as Synthesizer
    participant L as LLM

    R->>S: route = multi, agents = [sport, web]
    par Thread pool
        S->>A: run(query)
        S->>B: run(query)
    end
    A-->>S: answer + [Source 1..n]
    B-->>S: answer + [Source 1..m]
    S->>Y: agent_responses
    alt Chỉ 1 agent thành công
        Y-->>R: trả nguyên câu trả lời (không gọi LLM)
    else Từ 2 agent trở lên
        Y->>Y: đánh lại số nguồn: B dùng [Source n+1..n+m]
        Y->>L: 1 lần gọi → JSON {answer, themes}
        L-->>Y: câu trả lời gộp + 3–5 key themes
    end
```

#### Cơ Chế Hoạt Động Cụ Thể:
- **Supervisor Agent (Thực thi song song đa miền):**
  - **Parallel Execution:** Router chọn nhiều agent (ví dụ `tech` + `web`); Supervisor chạy các agent này đồng thời bằng thread pool, không tự gọi LLM.
  - **Collect:** Thu thập câu trả lời và nguồn trích dẫn của từng agent, giao cho Synthesizer.
- **Synthesizer Agent:**
  - **Một agent:** trả nguyên câu trả lời và trích dẫn của agent đó (không gọi thêm LLM, không cắt nội dung).
  - **Nhiều agent:** đánh lại số `[Source N]` để duy nhất trên toàn bộ câu trả lời, rồi gộp bằng **một** lần gọi LLM (kèm Key Themes).
  - **Trước đây (Cách cũ):** Thực hiện 2 cuộc gọi LLM nối tiếp (gọi trích xuất Key Themes, sau đó gọi tổng hợp bài viết), tổng thời gian mất ~30 giây.
  - **Hiện tại (Cách tối ưu):** Gộp thành **1 cuộc gọi LLM duy nhất** yêu cầu định dạng trả về bao gồm cả Key Themes, câu trả lời tổng hợp toàn diện và danh sách trích dẫn.

---

### 4.4. Web Search Tool & Live Internet Agent (Thông Tin Thời Gian Thực)

Hệ thống tích hợp **Web Search Tool** (sử dụng thư viện `ddgs` - DuckDuckGo Search API) và **Web Scraper Tool** (sử dụng `httpx` + `BeautifulSoup`), mang lại khả năng xử lý các truy vấn nằm ngoài kho tài liệu nội bộ:

#### 1. Cơ Chế Hoạt Động Của Web Agent:
- **Định tuyến chủ động từ Router Agent:** Khi Router phát hiện câu hỏi mang tính chất thời sự, sự kiện mới diễn ra (vd: *"tin tức hôm nay"*, *"kết quả trận đấu tối qua"*), hoặc yêu cầu rõ ràng về tìm kiếm Internet, hệ thống sẽ chuyển ngay sang `web_agent_node`.
- **Cơ chế Fallback thông minh (2 cấp độ):**
  1. *Cấp 1:* Nếu tài liệu không tìm thấy trong domain chính, thử tìm trên các domain nội bộ còn lại (`Sport`, `Tech`, `General`).
  2. *Cấp 2:* Nếu cả 3 domain nội bộ đều không có dữ liệu phù hợp, hệ thống tự động kích hoạt `WebSearchTool` tìm kiếm trên DuckDuckGo để bổ sung ngữ cảnh, giảm nguy cơ mô hình tự bịa câu trả lời (hallucination).
- **Trích dẫn nguồn trực tiếp:** Kết quả trả về kèm đầy đủ tiêu đề trang web, đường dẫn URL (`https://...`) và đoạn trích dẫn (snippet), cho phép người dùng nhấp trực tiếp vào liên kết trên giao diện người dùng.

### 4.5. Đánh Giá Chất Lượng & Vòng Lặp Tự Sửa (Evaluation & Self-Correction)

Hệ thống đánh giá ở hai tầng:

**a) Online, trong mỗi request (LLM-as-judge + reflection):**

1. Các agent trả về `contexts`: toàn bộ chunk đã đưa cho LLM, đánh số giống các thẻ `[Source N]` trong câu trả lời. Khi gộp nhiều agent, Synthesizer đánh lại số cho khớp.
2. `evaluator` (`AnswerJudge`) chấm câu trả lời theo thang 1-5:
   - **faithfulness**: mọi khẳng định đều có trong context, trích dẫn đúng nguồn.
   - **relevance**: trả lời đúng câu hỏi.
   - **completeness**: dùng hết thông tin hữu ích có trong context.

   Judge cũng trả về danh sách lỗi cụ thể và hướng sửa.
3. Nếu có tiêu chí dưới `SELF_EVAL_MIN_SCORE`, `refine` (`AnswerRefiner`) gửi cho LLM: câu hỏi, context, câu trả lời cũ, điểm và nhận xét. LLM viết lại câu trả lời, sau đó câu trả lời mới được chấm lại. Tối đa `SELF_EVAL_MAX_ROUNDS` vòng.
4. Bỏ qua khi không có context để đối chiếu (Tool Agent, chat thường). Nếu judge lỗi hoặc trả về JSON hỏng, câu trả lời vẫn được giữ nguyên, không bao giờ bị chặn.
5. API trả về trường `evaluation` (điểm, lỗi, số vòng sửa, lịch sử từng vòng). UI hiển thị badge "Chất lượng x/5".

Chi phí: thêm 1 lệnh gọi LLM cho mỗi câu hỏi có tài liệu, và thêm 2 lệnh gọi nếu phải sửa.

**b) Offline, trên bộ câu hỏi chuẩn (`evals/datasets/*.jsonl`):**

```bash
python -m src.evaluation.run --dataset evals/datasets/sport.jsonl evals/datasets/tech.jsonl
python -m src.evaluation.run --dataset evals/datasets/sport.jsonl --no-refine     # baseline: chỉ chấm, không sửa
python -m src.evaluation.run --only sport-gk-8s sport-subs                         # chạy vài câu
python -m src.evaluation.run --dataset evals/datasets/*.jsonl --ragas              # thêm độ đo RAGAS
python -m src.evaluation.run --rescore evals/results/<lần chạy>.json               # chấm RAGAS lại kết quả cũ, không chạy pipeline
```

Mỗi câu hỏi có đáp án mẫu (`reference`) và nguồn và trang cần truy xuất (`evidence`). Script đo các nhóm chỉ số sau:

- **Định tuyến:** route accuracy.
- **Truy xuất:** hit rate, MRR, precision của các chunk được đưa cho LLM.
- **Câu trả lời:** faithfulness, relevance, completeness và **correctness so với đáp án mẫu**.
- **Vòng tự sửa:** điểm của bản nháp đầu so với điểm sau khi sửa.
- **Độ trễ.**

Kết quả lưu ở `evals/results/<thời điểm>.json` và `.md`, thư mục này được gitignore.

**c) Độ đo RAGAS (`--ragas`, [src/evaluation/ragas_eval.py](src/evaluation/ragas_eval.py)):**

Khác với judge thang 1-5, RAGAS tách câu trả lời thành từng khẳng định nhỏ rồi kiểm tra riêng từng cái. Điểm từ 0 đến 1, so sánh được với các benchmark RAG đã công bố.

| Độ đo | Ý nghĩa | Cần đáp án mẫu |
| :-- | :-- | :-- |
| `faithfulness` | Tỷ lệ khẳng định trong câu trả lời được context hỗ trợ | Không |
| `answer_relevancy` | Sinh lại câu hỏi từ câu trả lời, rồi so độ tương đồng với câu hỏi gốc (dùng LLM và embedding) | Không |
| `context_precision` | Các chunk hữu ích có được xếp hạng đầu không | Có |
| `context_recall` | Tỷ lệ ý trong đáp án mẫu tìm thấy trong context | Có |
| `factual_correctness` | F1 theo từng khẳng định giữa câu trả lời và đáp án mẫu | Có |

- LLM được gọi qua endpoint tương thích OpenAI của NVIDIA NIM.
- Embedding dùng lại `bge-m3` đã nạp sẵn, không tải thêm bản thứ hai.
- Telemetry của ragas bị tắt (`RAGAS_DO_NOT_TRACK=true`).
- Mỗi câu hỏi tốn khoảng 10 lệnh gọi LLM, chạy song song theo `RAGAS_CONCURRENCY`.

---

## 5. Cấu Trúc Thư Mục Dự Án

```
multi-rag-agent/
├── api/                           # FastAPI Backend
│   ├── main.py                    # Khởi tạo app, CORS, lifespan nạp dữ liệu tự động
│   ├── routes.py                  # /query, /query/stream, /query/batch, /route, /health, /stats, /conversations, /mcp/status
│   └── schemas.py                 # Pydantic schemas cho Request & Response
├── config/                        # Cấu hình hệ thống
│   ├── settings.py                # Pydantic BaseSettings đọc biến môi trường (.env)
│   └── mcp_servers.json           # Khai báo MCP server (filesystem, fetch) + allow-list tool
├── data/                          # Dữ liệu
│   ├── raw/                       # Thư mục chứa tài liệu gốc theo miền
│   │   ├── sport/                 # Luật thi đấu, lịch sử giải đấu, chiến thuật, khoa học thể thao
│   │   ├── tech/                  # Tài liệu kỹ thuật, kiến trúc phần mềm, AI/ML
│   │   └── general/               # Tri thức tổng hợp khác
│   └── vectorstores/              # Nơi lưu trữ cơ sở dữ liệu vector ChromaDB
│       ├── .ingested_manifest.json# Manifest theo dõi trạng thái hash của tài liệu
│       ├── sport_store/           # Chroma collection: sport
│       ├── tech_store/            # Chroma collection: tech
│       └── general_store/         # Chroma collection: general
├── frontend/                      # Ứng dụng giao diện người dùng React + Vite
│   ├── src/                       # Components, Chat UI, styles
│   └── package.json
├── src/                           # Mã nguồn cốt lõi (Core Engine)
│   ├── agents/                    # Định nghĩa các Agents
│   │   ├── base_agent.py          # Lớp cơ sở tích hợp ChatNVIDIA
│   │   ├── router_agent.py        # Agent phân loại ý định & routing
│   │   ├── rag_agent.py           # Specialist Agent thực hiện chu trình RAG
│   │   ├── web_agent.py           # Tìm kiếm web (DuckDuckGo)
│   │   ├── tool_agent.py          # Vòng lặp tool-calling qua MCP (đọc file, fetch URL)
│   │   ├── supervisor_agent.py    # Chạy song song nhiều agent cho câu hỏi đa miền
│   │   └── synthesizer_agent.py   # Tổng hợp phản hồi & trích dẫn nguồn
│   ├── graph/                     # Xây dựng đồ thị LangGraph
│   │   ├── state.py               # TypedDict AgentState
│   │   ├── nodes.py               # Định nghĩa các hàm Node trong đồ thị
│   │   ├── edges.py               # Điều kiện rẽ nhánh (Conditional Edges)
│   │   └── workflow.py            # Biên dịch StateGraph và hàm run_workflow()
│   ├── ingestion/                 # Pipeline xử lý dữ liệu
│   │   ├── auto_ingest.py         # Quét và nhúng tài liệu gia tăng tự động
│   │   ├── loader.py              # Bộ đọc PDF/Text/DOCX
│   │   ├── chunker.py             # Bộ cắt văn bản đệ quy (Recursive Splitter)
│   │   ├── contextualizer.py      # (tùy chọn) Ngữ cảnh theo trang: tiêu đề › mục + tóm tắt, gắn vào đầu chunk
│   │   └── embedder.py            # VectorStoreBuilder
│   ├── retrieval/                 # Công cụ truy xuất RAG
│   │   ├── vector_store.py        # VectorStoreManager: 3 collection Chroma, vector/BM25/hybrid search
│   │   ├── reranker.py            # Cross-Encoder Reranker (chấm theo biến thể câu hỏi cùng ngôn ngữ)
│   │   ├── lang.py                # Nhận diện tiếng Việt và chọn biến thể câu hỏi cho rerank
│   │   └── query_expansion.py     # Kỹ thuật HyDE & Multi-Query Expansion
│   ├── evaluation/                # Đánh giá chất lượng
│   │   ├── judge.py               # AnswerJudge (LLM-as-judge) + AnswerRefiner (tự sửa theo nhận xét)
│   │   ├── metrics.py             # hit rate / MRR / precision truy xuất, tổng hợp kết quả
│   │   ├── ragas_eval.py          # Độ đo RAGAS (faithfulness, answer relevancy, context precision/recall, factual correctness)
│   │   └── run.py                 # CLI chạy eval offline: python -m src.evaluation.run
│   ├── mcp_client/manager.py      # MCPManager: kết nối stdio, allow-list, chặn SSRF, timeout
│   ├── memory/                    # Bộ nhớ hội thoại SQLite + viết lại câu hỏi nối tiếp
│   ├── tools/                     # WebSearchTool, WebScraperTool (calculator/search_tools chưa nối vào graph)
│   └── utils/                     # Công cụ ghi log và bộ đo thời gian (Timer)
├── evals/datasets/                # Bộ câu hỏi chuẩn (JSONL): câu hỏi, đáp án mẫu, nguồn + trang cần truy xuất
├── tests/                         # pytest (LLM giả, kho vector + reranker thật)
├── requirements.txt               # Thư viện phụ thuộc Python
└── README.md                      # Tài liệu hướng dẫn dự án
```

---

## 6. Cấu Hình Hệ Thống (Configuration)

Mọi cấu hình được tập trung tại file [config/settings.py](file:///e:/AI%20Agent/ai-agent/multi-rag-agent/config/settings.py) và có thể nạp đè thông qua file `.env`:

| Biến môi trường / Cấu hình | Giá trị mặc định | Mô tả |
| :--- | :--- | :--- |
| `NVIDIA_API_KEY` | `""` | Khóa API NVIDIA NIM (dùng cho ChatNVIDIA) |
| `DEFAULT_LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | LLM sinh câu trả lời RAG và tổng hợp |
| `FAST_LLM_MODEL` | `nvidia/nemotron-3.5-bolt-e4b` | LLM nhỏ cho query expansion |
| `ROUTER_LLM_MODEL` / `CHAT_LLM_MODEL` | `nvidia/nemotron-3.5-bolt-e4b` | Router (JSON ngắn) và hội thoại thường; model không suy luận nên phản hồi ~1-2s. `gpt-oss-20b` là model suy luận, mất 50-300s cho cùng việc |
| `POWERFUL_LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Mô hình mạnh (hiện chưa agent nào dùng) |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Mô hình nhúng đa ngôn ngữ cục bộ (1024 dimensions) |
| `CHUNK_SIZE` | `512` | Số ký tự tối đa trên mỗi chunk tài liệu |
| `CHUNK_OVERLAP` | `64` | Số ký tự gối đầu giữa 2 chunk liên tiếp |
| `TOP_K_RETRIEVAL` | `10` | Số lượng tài liệu thô truy xuất từ vector store |
| `TOP_K_RERANK` | `4` | Số tài liệu tinh chọn sau khi chạy Cross-Encoder |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Mô hình Cross-Encoder để tái chấm điểm |
| `USE_RERANKER` | `True` | Bật/tắt tầng Reranking trong pipeline |
| `VECTOR_WEIGHT` / `BM25_WEIGHT` | `0.6` / `0.4` | Trọng số hợp nhất vector và BM25 |
| `RELEVANCE_THRESHOLD` | `0.05` | Điểm rerank (0..1) tối thiểu để một chunk được coi là liên quan |
| `PARALLEL_EXPANSION` | `True` | Chạy query expansion song song với router LLM (expansion không phụ thuộc route) |
| `QUERY_EXPANSION_CROSSLINGUAL` | `True` | Một biến thể của query expansion là bản dịch Việt ↔ Anh, dùng cho cả truy xuất lẫn rerank |
| `INDEX_VERSION` | `""` | Phiên bản index. Mỗi phiên bản có thư mục store riêng (`<model>-<version>`) và manifest riêng, nên có thể dựng index mới song song với index cũ để so sánh |
| `CONTEXTUAL_CHUNKS` | `False` | Gọi LLM một lần mỗi trang để sinh "Tên tài liệu › Mục" và tóm tắt một câu, rồi gắn vào đầu mỗi chunk (có cache). Tắt mặc định vì không cải thiện trên bộ eval hiện tại |
| `SELF_EVAL_ENABLED` | `True` | Bật LLM judge chấm câu trả lời sau bước tổng hợp |
| `JUDGE_LLM_MODEL` | `""` (= `DEFAULT_LLM_MODEL`) | Model dùng làm giám khảo |
| `SELF_EVAL_MIN_SCORE` | `4` | Tiêu chí nào có điểm (1-5) thấp hơn ngưỡng này thì câu trả lời bị coi là chưa đạt |
| `SELF_EVAL_MAX_ROUNDS` | `1` | Số vòng sửa tối đa (0 = chỉ chấm, không sửa) |
| `RAGAS_LLM_MODEL` | `""` (= judge / default) | Model dùng để chấm RAGAS offline |
| `RAGAS_CONCURRENCY` | `2` | Số lệnh gọi LLM song song khi chấm RAGAS (gói NIM miễn phí khoảng 40 request/phút) |
| `CORS_ORIGINS` | `localhost:5173,...` | Danh sách origin được phép (phân tách bằng dấu phẩy) |

---

## 7. Hướng Dẫn Cài Đặt & Khởi Chạy

### Bước 1: Chuẩn bị môi trường Python
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
source .venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Thiết lập biến môi trường
Tạo file `.env` từ file mẫu:
```bash
cp .env.example .env
```
Mở `.env` và điền API Key tương ứng:
```env
NVIDIA_API_KEY=nvapi-your-key-here
```

### Bước 3: Thêm tài liệu vào kho dữ liệu
Đặt các tài liệu cần tra cứu vào thư mục thích hợp:
- Thể thao: `data/raw/sport/` (ví dụ: luật thi đấu, lịch sử giải đấu, tài liệu chiến thuật, huấn luyện)
- Công nghệ: `data/raw/tech/` (ví dụ: whitepaper, tài liệu kiến trúc hệ thống, API docs)
- Tổng hợp: `data/raw/general/` (ví dụ: nội quy, chính sách, kiến thức chung)

*Định dạng hỗ trợ:* `.pdf`, `.txt`, `.md`, `.docx`.

### Bước 4: Khởi chạy Backend Server
> **Lưu ý trên Windows:** Đảm bảo đã kích hoạt môi trường ảo `.venv` trước khi chạy hoặc chỉ định đường dẫn trực tiếp từ `.venv`.

- **Cách 1: Kích hoạt .venv rồi chạy:**
  ```powershell
  # Windows PowerShell:
  .venv\Scripts\Activate.ps1
  uvicorn api.main:app --reload --port 8000
  ```
- **Cách 2: Chạy trực tiếp từ .venv (không cần activate):**
  ```powershell
  .\.venv\Scripts\uvicorn.exe api.main:app --reload --port 8000
  ```
- **Trên Linux/macOS:**
  ```bash
  source .venv/bin/activate
  uvicorn api.main:app --reload --port 8000
  ```
> Khi khởi động, hệ thống quét `data/raw/` ở luồng nền (API trả lời được ngay), chunking và embedding file mới/đã sửa vào ChromaDB, đồng thời xóa chunk của file đã bị xóa hoặc thay đổi. Bạn có thể theo dõi tiến trình này ngay trên màn hình terminal console.

- Tài liệu tương tác API Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Bước 5: Khởi chạy React Chat UI (Frontend)
Mở một terminal mới:
```bash
cd frontend
npm install
npm run dev
```
Truy cập giao diện chat tại: http://localhost:5173

---

## 8. API Endpoints & Cách Sử Dụng

### MCP client (công cụ ngoài qua Model Context Protocol)
- Agent `tools` gọi công cụ của các MCP server khai báo trong `config/mcp_servers.json` (mặc định: **filesystem** chỉ đọc trong `data/mcp_workspace/` và **fetch** tải nội dung URL công khai). Router chỉ đưa route `tools` ra khi có server đã kết nối; câu như "Đọc file ghi_chu.txt", "Tóm tắt trang https://..." sẽ đi vào đây.
- Vòng lặp tool-calling gốc của model (tối đa `MCP_MAX_ITERATIONS`=5 vòng): model chọn tool → chạy qua MCP → trả kết quả lại cho model → câu trả lời cuối được stream. Giao diện hiện "Đang dùng công cụ ..." và danh sách công cụ đã dùng dưới mỗi câu trả lời. `GET /api/mcp/status` xem server và tool đang có.
- Thêm server: thêm mục vào `servers` (`command`, `args`, `env`, `allowed_tools`, `enabled`); `${WORKSPACE}` và `${BIEN_MOI_TRUONG}` được thay thế. Cần `npx` (Node) hoặc `uvx` (uv) trong PATH; lần đầu có thể mất thời gian tải gói. Server lỗi chỉ bị bỏ qua, không làm hỏng các server khác. Tắt hẳn bằng `MCP_ENABLED=false`.
- **An toàn:** chỉ tool nằm trong `allowed_tools` được đưa cho model (filesystem: chỉ các tool đọc, không có ghi/xóa); filesystem bị giới hạn trong workspace; tham số `url` bị chặn nếu trỏ tới địa chỉ nội bộ (SSRF); kết quả tool được coi là dữ liệu không tin cậy và bị cắt ở `MCP_MAX_OUTPUT_CHARS`. Khi thêm server có khả năng ghi (ghi file, DB, gửi mail...) hãy cân nhắc kỹ `allowed_tools`, vì nội dung web/file có thể chứa chỉ dẫn độc hại (prompt injection).
- Kết nối MCP chỉ thực hiện lúc khởi động; đổi cấu hình cần khởi động lại server.

### Conversation memory (bộ nhớ hội thoại)
- Hội thoại lưu ở SQLite `data/memory.db` (bảng `conversations`, `messages`). Gửi `conversation_id` trong `/api/query` hoặc `/api/query/stream` để tiếp tục; bỏ trống thì server tạo mới và trả `conversation_id` trong kết quả (`final`).
- Câu hỏi nối tiếp ("giải thích kỹ hơn về nó") được node `condense` viết lại thành câu độc lập bằng model nhỏ trước khi route/truy xuất; chỉ chạy khi có lịch sử và câu hỏi trông như câu nối tiếp (ngắn hoặc có đại từ), câu chào bỏ qua.
- Prompt sinh câu trả lời nhận tối đa `MEMORY_MAX_TURNS` (6) lượt gần nhất trong ngân sách `MEMORY_MAX_CHARS` (4000); thẻ `[Source N]` cũ bị loại khỏi lịch sử để không lẫn với đánh số nguồn của lượt mới.
- Chỉ lưu câu trả lời thành công. Endpoint: `GET /api/conversations`, `GET /api/conversations/{id}`, `DELETE /api/conversations/{id}`. Frontend lưu id ở localStorage, khôi phục khi tải lại trang và có danh sách lịch sử ở sidebar.
- Chưa có xác thực: ai biết `conversation_id` (UUID ngẫu nhiên) đều đọc được hội thoại đó.
- LLM suy luận (nemotron 3.5) mặc định tắt chế độ nghĩ (`LLM_ENABLE_THINKING=false`); bật lên có thể làm câu trả lời rỗng vì hết ngân sách token.

### 0. Streaming (`POST /api/query/stream`) — frontend dùng endpoint này
Trả về Server-Sent Events: `stage` (routing/retrieving/synthesizing), `route`, nhiều `token` (chữ hiện dần), rồi một `final` chứa `QueryResponse` đầy đủ (nguồn, themes...) hoặc `error`. Câu chào/cảm ơn/"bạn là ai" đi đường tắt regex, không tốn lần gọi LLM phân loại. Route một agent stream trực tiếp; route nhiều agent chỉ stream sau bước tổng hợp (không stream được vì đầu ra là JSON).

### 1. Truy vấn câu hỏi (`POST /api/query`)
Xử lý câu hỏi đầy đủ qua LangGraph workflow:

**Trên Linux / macOS (hoặc Git Bash):**
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Kiến trúc Model Context Protocol (MCP) là gì?",
    "include_metadata": true
  }'
```

**Trên Windows PowerShell:**
```powershell
# Sử dụng curl.exe (curl nguyên bản của Windows):
curl.exe -X POST "http://localhost:8000/api/query" -H "Content-Type: application/json" -d "{\"query\": \"Kien truc Model Context Protocol (MCP) la gi?\", \"include_metadata\": true}"

# Hoặc sử dụng Invoke-RestMethod (chuẩn PowerShell):
Invoke-RestMethod -Uri "http://localhost:8000/api/query" -Method Post -ContentType "application/json" -Body '{"query": "Kiến trúc Model Context Protocol (MCP) là gì?", "include_metadata": true}'
```

**Ví dụ phản hồi:**
```json
{
  "query": "Kiến trúc Model Context Protocol (MCP) là gì?",
  "success": true,
  "final_answer": "Model Context Protocol (MCP) là một chuẩn giao tiếp mở cho phép... [Source 1]",
  "route": "tech",
  "confidence": 0.95,
  "key_themes": [
    "Khái niệm MCP",
    "Kiến trúc Client - Server",
    "Giao thức tích hợp công cụ AI"
  ],
  "sources": [
    {
      "source_id": 1,
      "source": "data/raw/tech/mcp_overview.pdf",
      "page": 2
    }
  ],
  "agent_responses": { ... },
  "metadata": {
    "processing_time": 3.42,
    "status": "complete"
  }
}
```

### 2. Kiểm tra quyết định định tuyến (`POST /api/route`)
Xem trước Router Agent sẽ phân loại câu hỏi vào domain nào:
```bash
curl -X POST "http://localhost:8000/api/route" \
  -H "Content-Type: application/json" \
  -d '{"query": "Luật việt vị trong bóng đá được áp dụng thế nào?"}'
```

### 3. Kiểm tra trạng thái hệ thống (`GET /api/health` & `GET /api/stats`)
- Kiểm tra tính sẵn sàng của backend:
  ```bash
  curl http://localhost:8000/api/health
  ```
- Xem số lượng vector và tài liệu trong từng domain:
  ```bash
  curl http://localhost:8000/api/stats
  ```

---

### 4. Chạy test
```bash
python -m pytest -q
```
Test dùng LLM giả nhưng kho vector, embedding và reranker thật (cần đã nạp `data/raw/tech`).

Đánh giá chất lượng với LLM thật (xem mục 4.5): `python -m src.evaluation.run`.

---

## 9. Khởi Chạy Trực Tiếp Qua Python Script

Bạn cũng có thể gọi trực tiếp hệ thống trong mã nguồn Python mà không cần mở server API:

```python
from src.graph import run_workflow

query = "Lịch sử và thể thức thi đấu của FIFA World Cup?"
result = run_workflow(query, debug=True)

print("=== CÂU TRẢ LỜI CUỐI CÙNG ===")
print(result["final_answer"])

print("\n=== CÁC CHỦ ĐỀ CHÍNH ===")
print(result["key_themes"])

print("\n=== NGUỒN TRÍCH DẪN ===")
for source in result["sources"]:
    print(f"- Nguồn: {source.get('source')}, Trang: {source.get('page')}")
```

---

## 10. Tiến Độ Dự Án

> Cập nhật: 20/11/2026 · Test: **112/112 passed** (`python -m pytest -q`, ~2 phút 30 giây)

### Các mốc đã hoàn thành (theo lịch sử commit)

| # | Commit | Mốc |
| :-- | :-- | :-- |
| 1 | `29105e0` | Khởi tạo dự án: LangGraph, các agent, ingestion, ChromaDB, FastAPI, React UI |
| 2 | `e37eea4` | Dọn repo: bỏ theo dõi `node_modules`, build, vectorstore, log |
| 3 | `413ab2e` | Sửa workflow chạy đúng; truy xuất và nạp tài liệu nhanh hơn |
| 4 | `cfb7a49` | Gia cố API (ẩn lỗi nội bộ, trace id, giới hạn batch), cập nhật frontend và tài liệu |
| 5 | `220eea1` | Bộ test pytest (LLM giả, truy xuất thật) |
| 6 | `bf51270` | Embedding đa ngôn ngữ `bge-m3` và reranker `bge-reranker-v2-m3` |
| 7 | `3944c52` | Đường tắt câu chào (không gọi LLM), model nhỏ cho router/chat, streaming SSE |
| 8 | `fe4f366` | Bộ nhớ hội thoại SQLite, viết lại câu hỏi nối tiếp, lịch sử ở sidebar |
| 9 | `950f876` | MCP client và Tool Agent (filesystem chỉ đọc + fetch URL) |
| 10 | `9f343e6` | Đổi domain finance sang sport, sơ đồ README bằng Mermaid |
| 11 | `3b1df32` | Evaluation: LLM judge, vòng tự sửa, eval offline, RAGAS |
| 12 | `5a3b328` | Truy xuất xuyên ngôn ngữ: dịch Việt ↔ Anh trong expansion, rerank cùng ngôn ngữ |
| 13 | (commit này) | Giảm độ trễ (router gọn, expansion song song, 8 ứng viên rerank); Supervisor truyền lịch sử hội thoại |

### Trạng thái theo thành phần

| Thành phần | Trạng thái | Ghi chú |
| :-- | :-- | :-- |
| Ingestion tự động (PDF/TXT/MD/DOCX, manifest, xóa chunk cũ) | ✅ Hoàn thành | Chạy nền khi khởi động |
| Hybrid retrieval (vector + BM25, weighted RRF) | ✅ Hoàn thành | |
| Cross-encoder rerank + ngưỡng liên quan | ✅ Hoàn thành | |
| Fallback 3 tầng (domain chính → domain khác → web) | ✅ Hoàn thành | |
| Query expansion | 🟡 Một phần | Multi-query bật; HyDE, synonyms, domain-specific có code nhưng tắt |
| Router (regex fast path + LLM JSON) | ✅ Hoàn thành | Route `tools` chỉ xuất hiện khi MCP đã kết nối |
| RAG agent theo domain (Sport / Tech / General) | 🟡 Một phần | Tech: 3 PDF (630 chunk); Sport: 4 PDF luật bóng đá, FIFA, VFF (880 chunk, không đưa lên git); General đang trống |
| Web Agent (DuckDuckGo) | ✅ Hoàn thành | `WebScraperTool` có sẵn nhưng Web Agent chưa dùng |
| Supervisor (song song) + Synthesizer (đánh số nguồn toàn cục) | ✅ Hoàn thành | Agent con nhận lịch sử hội thoại và context của luồng gọi (stream event, tracing) |
| Tool Agent qua MCP | ✅ Hoàn thành | Chỉ đọc; chặn SSRF; tối đa 5 vòng |
| Bộ nhớ hội thoại | ✅ Hoàn thành | Chưa có xác thực người dùng |
| Streaming SSE + React UI | ✅ Hoàn thành | Theme sáng/tối, lịch sử hội thoại, trạng thái API |
| Kiểm thử tự động | ✅ 112 test | API, workflow, MCP, memory, đa ngôn ngữ, evaluation, contextual/rerank, unit |
| Đánh giá chất lượng (LLM judge + tự sửa + eval offline + RAGAS) | ✅ Hoàn thành | 24 câu chuẩn (18 sport, 6 tech), xem bảng kết quả bên dưới |
| Truy xuất xuyên ngôn ngữ (Việt ↔ Anh) | ✅ Hoàn thành | Expansion có bản dịch, rerank cùng ngôn ngữ. Contextual chunk đã thử, không cải thiện nên mặc định tắt |

### Kết quả đánh giá (20/11/2026)

Cấu hình mặc định, 24 câu, LLM `nemotron-3.5-lightning-30b-a3b`. Thang điểm judge là 1-5, RAGAS là 0-1.

**Truy xuất (eval chỉ đo truy xuất, A/B):**

| Cấu hình | Hit rate | MRR | Precision |
| :-- | :-- | :-- | :-- |
| Ban đầu (chunk 512, không dịch) | 0.79 | 0.75 | 0.47 |
| + dịch Việt ↔ Anh, rerank cùng ngôn ngữ (**mặc định**) | **0.96** | **0.90** | **0.56** |
| + contextual chunk (index v2, chunk 1000) | 0.92 | 0.87 | 0.51 |

**Toàn pipeline:**
- Route accuracy 100%.
- Judge: faithfulness 4.88, relevance 5.0, correctness 5.0; pass rate 0.96.
- Vòng tự sửa: 2 câu được sửa, cả 2 đạt sau khi sửa.
- RAGAS: faithfulness 0.93, answer relevancy 0.84, context precision 0.95, context recall 1.00, factual correctness 0.71.
- Độ trễ: xem bảng bên dưới.

**Độ trễ (Phase 3, cùng bộ câu hỏi, judge bật):**

| | Trước | Sau |
| :-- | :-- | :-- |
| Router | 2.0s (31/32 route đúng) | 0.7s (32/32): chỉ yêu cầu JSON gọn `route` / `all_domains` / `confidence` |
| Query expansion | chạy sau router (khoảng 0.8s) | chạy song song với router |
| Rerank | 2.75s (10 ứng viên) | khoảng 2.3s (8 ứng viên, hit rate không đổi) |
| Thời gian tới token đầu tiên (median, 6 câu) | khoảng 7.0s | **4.0s** |
| Trung bình mỗi câu trong eval 24 câu | 16.3s | **9.7s** |

Sau tối ưu, chất lượng giữ nguyên: route 100%, hit rate 0.96, MRR 0.90, faithfulness 4.96/5, pass rate 1.00.

### Việc còn tồn đọng / đề xuất tiếp theo

- **Dữ liệu:** bổ sung tài liệu cho `data/raw/general`; mở rộng bộ eval (hiện 24 câu, nên lên 50–100 câu).
- **Truy xuất:** `sport-kickoff-goal` vẫn trượt ("quả giao bóng" bị hiểu thành *goal kick*).
- **Judge:** điểm của LLM judge dao động giữa các lần chạy; nên dùng model judge mạnh hơn hoặc chấm nhiều lần lấy trung bình cho eval offline.
- **Code chưa nối vào hệ thống:** `src/tools/calculator_tool.py`, `src/tools/search_tools.py`, `DocumentLoader`, `VectorStoreBuilder`, `CacheManager`, `POWERFUL_LLM_MODEL` (có thể nối thành tool cho agent hoặc xóa bớt).
- **Route đa miền:** câu trả lời của các agent chạy song song không được stream; chỉ bản tổng hợp cuối cùng được gửi về.
- **API:** trường `domain` trong `QueryRequest` chưa được dùng; `/api/stats` trả `uptime = 0` và danh sách agent chưa có `tools`.
- **Bảo mật / vận hành:** chưa có xác thực, ai có `conversation_id` đều đọc được hội thoại; cấu hình MCP chỉ nạp lúc khởi động.
- **Chưa có:** upload tài liệu qua API/UI, Docker/CI, tracing (Langfuse/LangSmith).
