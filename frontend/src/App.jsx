import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  AlertCircle,
  Bot,
  BrainCircuit,
  Check,
  ChevronDown,
  Clock3,
  Code2,
  Copy,
  Cpu,
  Database,
  ExternalLink,
  FileText,
  Globe,
  Layers,
  Loader2,
  Moon,
  Plus,
  Send,
  Sparkles,
  Sun,
  TrendingUp,
  User,
  Zap,
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const REQUEST_TIMEOUT_MS = 180000;
const HEALTH_POLL_MS = 30000;

// crypto.randomUUID only exists in secure contexts (https / localhost); fall back for LAN http access.
const uid = () =>
  globalThis.crypto?.randomUUID?.() ?? `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

const SAMPLE_QUESTIONS = [
  {
    icon: Globe,
    domain: "Live Web",
    label: "Tin tức trực tuyến",
    prompt: "Tìm kiếm trên web tin tức mới nhất hôm nay về các mô hình AI agent",
  },
  {
    icon: Code2,
    domain: "Tech",
    label: "Giao thức MCP",
    prompt: "Model Context Protocol (MCP) là gì, vì sao nó quan trọng và các tính năng cốt lõi?",
  },
  {
    icon: TrendingUp,
    domain: "Finance",
    label: "Tài chính doanh nghiệp",
    prompt: "Các phương pháp định giá cổ phiếu phổ biến và cách đọc báo cáo lưu chuyển tiền tệ?",
  },
  {
    icon: Layers,
    domain: "RAG Architecture",
    label: "Kiến trúc Multi-RAG",
    prompt: "So sánh ưu nhược điểm giữa Vector Search thông thường và Hybrid Search kết hợp Reranker?",
  },
];

const STAGE_LABELS = {
  routing: "Đang phân loại câu hỏi...",
  retrieving: "Đang tra cứu tài liệu & tìm kiếm...",
  synthesizing: "Đang tổng hợp câu trả lời...",
};

const STATUS_LABELS = {
  checking: "Đang kiểm tra",
  online: "Sẵn sàng",
  degraded: "Hoạt động một phần",
  offline: "Mất kết nối",
};

const INITIAL_MESSAGES = [
  {
    id: "welcome-msg",
    role: "assistant",
    content:
      "### 👋 Xin chào! Tôi là Multi-RAG Autonomous Agent\nTôi có thể hỗ trợ bạn tra cứu và phân tích sâu về **Kỹ thuật công nghệ (Tech)**, **Tài chính doanh nghiệp (Finance)** từ kho tài liệu nội bộ, đồng thời **Tìm kiếm thông tin thời sự trực tiếp trên Internet (Web Search)**.\n\n*Hãy nhập câu hỏi bên dưới hoặc chọn gợi ý nhanh để bắt đầu!*",
    meta: {
      route: "ready",
      confidence: 1,
    },
  },
];

export default function App() {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const [stage, setStage] = useState("routing");
  const [streamingId, setStreamingId] = useState(null);
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [debug, setDebug] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("rag_theme") || "dark");
  const [copiedId, setCopiedId] = useState(null);
  const [apiStatus, setApiStatus] = useState("checking"); // checking | online | degraded | offline
  const abortRef = useRef(null);

  const inputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    localStorage.setItem("rag_theme", theme);
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isSending]);

  // Real backend health instead of a hard-coded "online" label
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`);
        const data = await res.json().catch(() => ({}));
        if (!cancelled) setApiStatus(res.ok ? (data.status === "healthy" ? "online" : "degraded") : "offline");
      } catch {
        if (!cancelled) setApiStatus("offline");
      }
    }
    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // Grow the textarea with its content (up to a cap)
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const canSend = input.trim().length > 0 && !isSending;

  const latestMeta = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.meta || null;
  }, [messages]);

  const handleCopy = async (id, text) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // ignore
    }
  };

  async function submitQuery(queryText) {
    const query = (queryText || input).trim();
    if (!query || isSending) return;

    const userMsg = {
      id: uid(),
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setError("");
    setIsSending(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    const assistantId = uid();
    let streamedText = "";
    let started = false;
    const patchAssistant = (patch) =>
      setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)));

    try {
      setStage("routing");
      const response = await fetch(`${API_BASE_URL}/api/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          query,
          include_metadata: includeMetadata,
          debug,
        }),
      });

      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || payload.message || "Không thể kết nối đến hệ thống Multi-RAG.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let finished = false;

      const handleEvent = (event) => {
        if (event.type === "stage") {
          setStage(event.stage);
        } else if (event.type === "token") {
          streamedText += event.text;
          if (!started) {
            started = true;
            setStreamingId(assistantId);
            setMessages((prev) => [
              ...prev,
              { id: assistantId, role: "assistant", content: streamedText, streaming: true },
            ]);
          } else {
            patchAssistant({ content: streamedText });
          }
        } else if (event.type === "final") {
          finished = true;
          const payload = event.response;
          const meta = {
            route: payload.route,
            confidence: payload.confidence,
            success: payload.success,
            sources: payload.sources || [],
            keyThemes: payload.key_themes || [],
            processingTime: payload.metadata?.processing_time,
          };
          const content = payload.final_answer || streamedText || "Agent không trả về nội dung.";
          if (started) {
            patchAssistant({ content, meta, streaming: false });
          } else {
            setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content, meta }]);
          }
        } else if (event.type === "error") {
          throw new Error(event.message || "Đã xảy ra lỗi khi xử lý câu hỏi.");
        }
      };

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const line = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (line) handleEvent(JSON.parse(line.slice(6)));
        }
      }
      if (!finished) throw new Error("Kết nối bị ngắt trước khi nhận đủ câu trả lời.");
    } catch (err) {
      const msg =
        err?.name === "AbortError"
          ? "Yêu cầu bị hủy hoặc quá thời gian chờ."
          : err instanceof Error
            ? err.message
            : "Đã xảy ra lỗi không xác định.";
      setError(msg);
      setMessages((prev) => [
        ...prev.filter((m) => m.id !== assistantId),
        {
          id: uid(),
          role: "assistant",
          content: `⚠️ **Lỗi kết nối:** ${msg}\n\n*Vui lòng kiểm tra lại trạng thái API backend hoặc kết nối mạng.*`,
          meta: { route: "error", success: false },
        },
      ]);
    } finally {
      clearTimeout(timeout);
      abortRef.current = null;
      setIsSending(false);
      setStreamingId(null);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }

  function handleFormSubmit(e) {
    e?.preventDefault();
    submitQuery();
  }

  function handleNewChat() {
    abortRef.current?.abort();
    setMessages(INITIAL_MESSAGES);
    setError("");
    setInput("");
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  const isInitialState = messages.length <= 1;

  return (
    <main className="appShell">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <BrainCircuit size={22} />
          </div>
          <div className="brandInfo">
            <h1>Multi-RAG</h1>
            <span className="brandBadge">Autonomous Agent</span>
          </div>
        </div>

        <button type="button" className="newChatBtn" onClick={handleNewChat}>
          <Plus size={18} />
          <span>Đoạn chat mới</span>
        </button>

        {/* DOMAINS INFO */}
        <section className="domainsCard" aria-label="Available Agents">
          <div className="sectionLabel">
            <Layers size={14} />
            <span>Chuyên mục Agent</span>
          </div>
          <div className="domainItems">
            <div className="domainItem tech">
              <Code2 size={16} />
              <div>
                <strong>Tech Agent</strong>
                <small>Tài liệu, API & Kiến trúc</small>
              </div>
            </div>
            <div className="domainItem finance">
              <TrendingUp size={16} />
              <div>
                <strong>Finance Agent</strong>
                <small>Báo cáo & phân tích tài chính</small>
              </div>
            </div>
            <div className="domainItem web">
              <Globe size={16} />
              <div>
                <strong>Web Agent</strong>
                <small>Tìm kiếm trực tuyến Real-time</small>
              </div>
            </div>
            <div className="domainItem router">
              <Zap size={16} />
              <div>
                <strong>Router Agent</strong>
                <small>Phân tích ý định & Điều phối</small>
              </div>
            </div>
          </div>
        </section>

        {/* SESSION MONITOR */}
        <section className="statusCard" aria-label="Session status">
          <div className="sectionLabel">
            <Sparkles size={14} />
            <span>Trạng thái phiên</span>
          </div>
          <div className="statusList">
            <div className="statusRow">
              <span className="statusKey">Trạng thái</span>
              <span className={apiStatus === "online" ? "statusOnline" : "statusValue"}>
                <span className="pulseDot" /> {STATUS_LABELS[apiStatus]}
              </span>
            </div>
            <div className="statusRow">
              <span className="statusKey">Route gần nhất</span>
              <span className={`routeTag ${latestMeta?.route || "none"}`}>
                {latestMeta?.route || "Chưa có"}
              </span>
            </div>
            <div className="statusRow">
              <span className="statusKey">Độ tin cậy</span>
              <span className="statusValue">{formatPercent(latestMeta?.confidence)}</span>
            </div>
            {typeof latestMeta?.processingTime === "number" && (
              <div className="statusRow">
                <span className="statusKey">Thời gian xử lý</span>
                <span className="statusValue">{latestMeta.processingTime.toFixed(2)}s</span>
              </div>
            )}
          </div>
        </section>

        {/* CONTROLS */}
        <section className="controlsCard" aria-label="Settings">
          <div className="sectionLabel">
            <Database size={14} />
            <span>Tùy chọn cấu hình</span>
          </div>
          <div className="controlToggles">
            <label className="toggleLabel">
              <input
                type="checkbox"
                checked={includeMetadata}
                onChange={(e) => setIncludeMetadata(e.target.checked)}
              />
              <span>Trích xuất Metadata</span>
            </label>
            <label className="toggleLabel">
              <input
                type="checkbox"
                checked={debug}
                onChange={(e) => setDebug(e.target.checked)}
              />
              <span>Chế độ Debug</span>
            </label>
          </div>

          <div className="sidebarFooter">
            <button
              type="button"
              className="iconBtn themeBtn"
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              title={theme === "dark" ? "Chuyển sang Giao diện Sáng" : "Chuyển sang Giao diện Tối"}
            >
              {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
              <span>{theme === "dark" ? "Giao diện sáng" : "Giao diện tối"}</span>
            </button>
          </div>
        </section>
      </aside>

      {/* CHAT MAIN PANEL */}
      <section className="chatPanel" aria-label="Chat container">
        {/* CHAT TOPBAR */}
        <header className="chatTopBar">
          <div className="chatTitleGroup">
            <h2>Multi-RAG Assistant</h2>
            <div className="modelBadge">
              <Cpu size={14} />
              <span>NVIDIA NIM LLM • Hybrid Vector Store</span>
            </div>
          </div>
          <div className="topBarActions">
            <span className="activeStatus">
              <span className="statusPill">API: {STATUS_LABELS[apiStatus]}</span>
            </span>
          </div>
        </header>

        {error && (
          <div className="errorBanner" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* MESSAGES LIST */}
        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="avatar" aria-hidden="true">
                {message.role === "user" ? <User size={18} /> : <Bot size={18} />}
              </div>

              <div className="bubble">
                <div className="bubbleHeader">
                  <span className="authorName">
                    {message.role === "user" ? "Bạn" : "Multi-RAG Agent"}
                  </span>
                  {message.role === "assistant" && (
                    <button
                      type="button"
                      className="copyBtn"
                      onClick={() => handleCopy(message.id, message.content)}
                      title="Sao chép câu trả lời"
                    >
                      {copiedId === message.id ? (
                        <>
                          <Check size={14} className="copiedIcon" />
                          <span>Đã chép</span>
                        </>
                      ) : (
                        <>
                          <Copy size={14} />
                          <span>Sao chép</span>
                        </>
                      )}
                    </button>
                  )}
                </div>

                <div className={`messageText ${message.role === "assistant" ? "markdown" : "userText"}`}>
                  {message.role === "assistant" ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{
                        table: ({ node, ...props }) => (
                          <div className="tableWrapper">
                            <table {...props} />
                          </div>
                        ),
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    message.content
                  )}
                </div>

                {message.meta && <MessageMeta meta={message.meta} />}
              </div>
            </article>
          ))}

          {/* THINKING STATE */}
          {isSending && !streamingId && (
            <article className="message assistant thinkingMessage">
              <div className="avatar" aria-hidden="true">
                <Bot size={18} />
              </div>
              <div className="bubble thinkingBubble">
                <div className="thinkingHeader">
                  <Loader2 size={16} className="spinIcon" />
                  <span>{STAGE_LABELS[stage] || "Multi-RAG Agent đang xử lý..."}</span>
                </div>
                <div className="thinkingBar">
                  <div className="thinkingShimmer" />
                </div>
              </div>
            </article>
          )}

          {/* SUGGESTION PROMPTS (EMPTY STATE) */}
          {isInitialState && (
            <div className="starterPrompts">
              <div className="starterHeader">
                <Sparkles size={16} />
                <span>Gợi ý câu hỏi bắt đầu nhanh:</span>
              </div>
              <div className="starterGrid">
                {SAMPLE_QUESTIONS.map((item, idx) => {
                  const Icon = item.icon;
                  return (
                    <button
                      type="button"
                      key={idx}
                      className="starterCard"
                      onClick={() => submitQuery(item.prompt)}
                    >
                      <div className="starterTop">
                        <Icon size={18} className="starterIcon" />
                        <span className="starterDomain">{item.domain}</span>
                      </div>
                      <strong>{item.label}</strong>
                      <p>{item.prompt}</p>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* COMPOSER / INPUT BAR */}
        <div className="composerWrapper">
          <form className="composer" onSubmit={handleFormSubmit}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleFormSubmit();
                }
              }}
              placeholder="Nhập câu hỏi để tra cứu tài liệu hoặc trò chuyện với Multi-RAG Agent..."
              rows={1}
            />
            <button
              type="submit"
              disabled={!canSend}
              className={`sendBtn ${canSend ? "active" : ""}`}
              aria-label="Gửi tin nhắn"
            >
              {isSending ? (
                <Loader2 size={18} className="spinIcon" />
              ) : (
                <Send size={18} />
              )}
            </button>
          </form>
          <div className="composerFooter">
            <span>Nhấn <strong>Enter</strong> để gửi, <strong>Shift + Enter</strong> để xuống dòng • Hỗ trợ hiển thị Markdown Preview & Bảng</span>
          </div>
        </div>
      </section>
    </main>
  );
}

function MessageMeta({ meta }) {
  const hasSources = Array.isArray(meta.sources) && meta.sources.length > 0;
  const hasThemes = Array.isArray(meta.keyThemes) && meta.keyThemes.length > 0;

  if (!meta.route && !hasSources && !hasThemes && typeof meta.confidence !== "number") {
    return null;
  }

  return (
    <div className="meta">
      <div className="metaRow">
        {meta.route && (
          <span className={`metaPill route-${meta.route}`}>
            <Zap size={11} /> {meta.route}
          </span>
        )}
        {typeof meta.confidence === "number" && (
          <span className="metaPill">Độ khớp: {formatPercent(meta.confidence)}</span>
        )}
        {typeof meta.processingTime === "number" && (
          <span className="metaPill withIcon">
            <Clock3 size={11} />
            {meta.processingTime.toFixed(2)}s
          </span>
        )}
      </div>

      {hasThemes && (
        <div className="chips">
          {meta.keyThemes.slice(0, 6).map((theme, i) => (
            <span key={i} className="chip">
              #{theme}
            </span>
          ))}
        </div>
      )}

      {hasSources && (
        <details className="sourcesAccordion">
          <summary>
            <FileText size={14} />
            <span>Nguồn tài liệu tham chiếu ({meta.sources.length})</span>
            <ChevronDown size={14} className="chevron" />
          </summary>
          <ul className="sourcesList">
            {meta.sources.slice(0, 8).map((source, index) => {
              const isWeb = source.agent === "web" || (source.source && source.source.startsWith("http")) || Boolean(source.url);
              const linkUrl = source.url || (source.source && source.source.startsWith("http") ? source.source : null);
              return (
                <li key={`${source.source || "source"}-${index}`}>
                  <div className="sourceHeader">
                    <span className={`sourceAgent ${source.agent === "web" ? "agent-web" : ""}`}>
                      {source.agent === "web" ? <Globe size={11} style={{ marginRight: 3, verticalAlign: 'middle' }} /> : null}
                      {source.agent || "Agent"}
                    </span>
                    {isWeb && linkUrl ? (
                      <a
                        href={linkUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="sourceLink"
                        title={source.snippet || source.title || linkUrl}
                      >
                        <span>{source.title || linkUrl}</span>
                        <ExternalLink size={12} className="extIcon" />
                      </a>
                    ) : (
                      <span className="sourceName">{source.source || "Tài liệu hệ thống"}</span>
                    )}
                    {source.page !== null && source.page !== undefined && (
                      <span className="sourcePage">Trang {source.page}</span>
                    )}
                  </div>
                  {source.snippet && isWeb && (
                    <p className="sourceSnippet">{source.snippet.slice(0, 160)}...</p>
                  )}
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </div>
  );
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  return `${Math.round(value * 100)}%`;
}
