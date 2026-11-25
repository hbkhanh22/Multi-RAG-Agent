import { Code2, Globe, Layers, Trophy } from "lucide-react";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
export const REQUEST_TIMEOUT_MS = 180000;
export const HEALTH_POLL_MS = 30000;

/** Chat attachments: keep in sync with src/attachments/reader.py and ATTACHMENT_* settings
    (the API is the one that enforces them). */
export const ATTACH_MAX_FILES = 5;
export const ATTACH_MAX_MB = 20;
export const ATTACH_TYPES =
  ".pdf .png .jpg .jpeg .webp .gif .bmp .tif .tiff .docx .xlsx .pptx .txt .md .csv .tsv .json .jsonl .log .xml .html .htm .yaml .yml .toml .ini .sql .py .js .jsx .ts .tsx .css .java .c .cpp .h .cs .go .rs .php .rb .sh .ps1 .ipynb .tex .srt".split(
    " "
  );

/** Route ids returned by the router agent -> human label + color tone. */
export const ROUTES = {
  tech: { label: "Tech", tone: "green" },
  sport: { label: "Sport", tone: "amber" },
  web: { label: "Web", tone: "blue" },
  tools: { label: "Công cụ MCP", tone: "violet" },
  files: { label: "Tệp đính kèm", tone: "violet" },
  multi: { label: "Đa miền", tone: "violet" },
  general: { label: "Tổng hợp", tone: "neutral" },
  direct_chat: { label: "Trò chuyện", tone: "neutral" },
  error: { label: "Lỗi", tone: "red" },
};

export const routeInfo = (route) => ROUTES[route] || { label: route, tone: "neutral" };

/** Pipeline stages streamed by the backend (`{type: "stage"}` events). */
export const STAGES = {
  understanding: "Hiểu ngữ cảnh hội thoại",
  routing: "Phân loại câu hỏi",
  reading_files: "Đọc tệp đính kèm",
  retrieving: "Tra cứu tài liệu & nguồn",
  synthesizing: "Tổng hợp câu trả lời",
  evaluating: "Tự đánh giá chất lượng",
  refining: "Cải thiện câu trả lời",
};

export const EVAL_CRITERIA = {
  faithfulness: "Bám sát nguồn",
  relevance: "Đúng trọng tâm",
  completeness: "Đầy đủ",
  correctness: "Chính xác",
};

export const HEALTH_LABELS = {
  checking: "Đang kiểm tra kết nối",
  online: "Hệ thống sẵn sàng",
  degraded: "Hoạt động một phần",
  offline: "Mất kết nối API",
};

export const SUGGESTIONS = [
  {
    icon: Code2,
    tone: "green",
    label: "Giao thức MCP là gì?",
    prompt: "Model Context Protocol (MCP) là gì, vì sao nó quan trọng và các tính năng cốt lõi?",
  },
  {
    icon: Trophy,
    tone: "amber",
    label: "Luật việt vị & VAR",
    prompt: "Luật việt vị trong bóng đá được quy định thế nào và VAR hỗ trợ ra sao?",
  },
  {
    icon: Globe,
    tone: "blue",
    label: "Tin AI agent hôm nay",
    prompt: "Tìm kiếm trên web tin tức mới nhất hôm nay về các mô hình AI agent",
  },
  {
    icon: Layers,
    tone: "violet",
    label: "Vector vs Hybrid Search",
    prompt: "So sánh ưu nhược điểm giữa Vector Search thông thường và Hybrid Search kết hợp Reranker?",
  },
];
