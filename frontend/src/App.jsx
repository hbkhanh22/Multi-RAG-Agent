import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDown, PanelLeft, SquarePen, WifiOff } from "lucide-react";
import Sidebar from "./components/Sidebar.jsx";
import Composer from "./components/Composer.jsx";
import Message from "./components/Message.jsx";
import { Greeting, Suggestions } from "./components/EmptyState.jsx";
import { API_BASE_URL, HEALTH_LABELS, HEALTH_POLL_MS, REQUEST_TIMEOUT_MS, STAGES, routeInfo } from "./lib/constants.js";
import { apiFetch } from "./lib/api.js";
import { useAttachments } from "./lib/useAttachments.js";
import { useMediaQuery } from "./lib/useTheme.js";
import { storage, uid } from "./lib/utils.js";

const MOBILE_QUERY = "(max-width: 768px)";

/* ------------------------------------------------------------------ hooks */

function useHealth() {
  const [health, setHealth] = useState({ status: "checking", components: {} });
  useEffect(() => {
    let cancelled = false;
    async function check() {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`);
        const data = await res.json().catch(() => ({}));
        if (cancelled) return;
        setHealth({
          status: res.ok ? (data.status === "healthy" ? "online" : "degraded") : "offline",
          components: data.components || {},
        });
      } catch {
        if (!cancelled) setHealth({ status: "offline", components: {} });
      }
    }
    check();
    const timer = setInterval(check, HEALTH_POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);
  return health;
}

/* ---------------------------------------------------- trace (agent steps) */

const finishActive = (trace, to = "done") => trace.map((s) => (s.status === "active" ? { ...s, status: to } : s));

function enterStage(trace, stage) {
  const key = `stage-${stage}`;
  const done = finishActive(trace);
  if (done.some((s) => s.key === key)) return done.map((s) => (s.key === key ? { ...s, status: "active" } : s));
  return [...done, { key, label: STAGES[stage] || stage, status: "active" }];
}

function storedToUi(stored) {
  return stored.map((m) => ({
    id: uid(),
    role: m.role,
    content: m.content,
    status: "done",
    files: m.role === "user" ? m.meta?.attachments : undefined,
    meta:
      m.role === "assistant" && m.meta
        ? {
            route: m.meta.route,
            confidence: m.meta.confidence,
            sources: m.meta.sources || [],
            keyThemes: m.meta.key_themes || [],
            processingTime: m.meta.processing_time,
            toolCalls: m.meta.tool_calls || [],
            evaluation: m.meta.evaluation || null,
          }
        : undefined,
  }));
}

/* -------------------------------------------------------------------- app */

/** `theme`: [value, set]. `user`: the signed-in account, absent when the server has no authentication. */
export default function App({ theme: [theme, setTheme], user }) {
  const isMobile = useMediaQuery(MOBILE_QUERY);
  const health = useHealth();

  const [sidebarOpen, setSidebarOpen] = useState(() => !window.matchMedia(MOBILE_QUERY).matches && storage.get("rag_sidebar") !== "closed");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const attachments = useAttachments();
  const [includeMetadata, setIncludeMetadata] = useState(() => storage.get("rag_metadata", "true") === "true");
  const [debug, setDebug] = useState(false);
  const [conversationId, setConversationId] = useState(() => storage.get("rag_conversation_id"));
  const [conversations, setConversations] = useState([]);
  const [atBottom, setAtBottom] = useState(true);

  const inputRef = useRef(null);
  const scrollRef = useRef(null);
  const abortRef = useRef(null);
  const stoppedRef = useRef(false);
  const atBottomRef = useRef(true);
  const showAnswerTopRef = useRef(false); // the final answer just replaced a draft

  useEffect(() => storage.set("rag_metadata", String(includeMetadata)), [includeMetadata]);
  useEffect(() => {
    if (!isMobile) storage.set("rag_sidebar", sidebarOpen ? "open" : "closed");
  }, [sidebarOpen, isMobile]);
  useEffect(() => {
    if (isMobile) setSidebarOpen(false);
  }, [isMobile]);

  const persistConversationId = useCallback((id) => {
    setConversationId(id);
    storage.set("rag_conversation_id", id);
  }, []);

  const focusInput = () => requestAnimationFrame(() => inputRef.current?.focus());
  const closeSidebarOnMobile = () => isMobile && setSidebarOpen(false);

  /* ---------- scrolling: follow the stream only while the user is at the bottom */

  const scrollToBottom = useCallback((behavior = "auto") => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior });
  }, []);

  function handleScroll() {
    const el = scrollRef.current;
    const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    atBottomRef.current = bottom;
    setAtBottom(bottom);
  }

  useEffect(() => {
    if (showAnswerTopRef.current) {
      // the whole answer appears at once: show its beginning, not its end
      showAnswerTopRef.current = false;
      scrollRef.current?.querySelector(".threadInner")?.lastElementChild?.scrollIntoView({ block: "start" });
    } else if (atBottomRef.current) scrollToBottom();
  }, [messages, scrollToBottom]);

  /* ---------- conversations */

  const refreshConversations = useCallback(async () => {
    try {
      const res = await apiFetch("/api/conversations");
      if (res.ok) setConversations((await res.json()).conversations || []);
    } catch {
      // history is optional
    }
  }, []);

  const loadConversation = useCallback(
    async (id) => {
      const res = await apiFetch(`/api/conversations/${id}`);
      if (res.status === 404) {
        persistConversationId(null);
        setMessages([]);
        return;
      }
      if (!res.ok) throw new Error("load failed");
      const data = await res.json();
      atBottomRef.current = true;
      setMessages(storedToUi(data.messages || []));
      persistConversationId(id);
    },
    [persistConversationId]
  );

  // signing out unmounts the chat: do not leave an answer running in the background
  useEffect(() => () => abortRef.current?.abort(), []);

  useEffect(() => {
    refreshConversations();
    if (conversationId) loadConversation(conversationId).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function openConversation(id) {
    closeSidebarOnMobile();
    if (id === conversationId) return;
    stopGeneration();
    attachments.clear({ discard: true });
    try {
      await loadConversation(id);
    } catch {
      refreshConversations();
    }
  }

  async function deleteConversation(id) {
    try {
      await apiFetch(`/api/conversations/${id}`, { method: "DELETE" });
    } catch {
      // ignore
    }
    if (id === conversationId) startNewChat();
    refreshConversations();
  }

  function startNewChat() {
    stopGeneration();
    persistConversationId(null);
    setMessages([]);
    setInput("");
    attachments.clear({ discard: true });
    closeSidebarOnMobile();
    focusInput();
  }

  function stopGeneration() {
    if (abortRef.current) {
      stoppedRef.current = true;
      abortRef.current.abort();
    }
  }

  /* ---------- sending + streaming */

  /** `resend`: files of a failed turn being retried (default: the files in the composer). */
  async function submitQuery(text, resend) {
    if (isSending || (!resend && attachments.busy)) return;
    const files = resend || attachments.ready;
    const query = text || (files.length > 0 ? "Tóm tắt nội dung tệp đính kèm." : "");
    if (!query) return;
    if (!resend) attachments.clear();

    const assistantId = uid();
    const startedAt = performance.now();
    const patch = (fn) => setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, ...fn(m) } : m)));

    setMessages((prev) => [
      ...prev,
      { id: uid(), role: "user", content: query, files },
      { id: assistantId, role: "assistant", content: "", status: "pending", trace: [], query, files },
    ]);
    setInput("");
    setIsSending(true);
    atBottomRef.current = true;
    requestAnimationFrame(() => scrollToBottom("smooth"));

    const controller = new AbortController();
    abortRef.current = controller;
    stoppedRef.current = false;
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let streamed = ""; // definitive text
    let draft = ""; // text that may still be rewritten: shown apart from the answer
    let replaceDraft = false;
    let finished = false;

    const handleEvent = (event) => {
      switch (event.type) {
        case "stage":
          patch((m) => ({ trace: enterStage(m.trace, event.stage), stage: event.stage }));
          break;
        case "route": {
          const info = routeInfo(event.route);
          patch((m) => ({
            trace: m.trace.map((s) => (s.key === "stage-routing" ? { ...s, detail: info.label, tone: info.tone } : s)),
          }));
          break;
        }
        case "tool": {
          const key = `tool-${event.server}-${event.tool}`;
          if (event.status === "running") {
            patch((m) => ({
              trace: [
                ...m.trace.filter((s) => s.key !== key),
                { key, label: `Gọi ${event.server} · ${event.tool}`, status: "active", sub: true },
              ],
            }));
          } else {
            patch((m) => ({
              trace: m.trace.map((s) => (s.key === key ? { ...s, status: event.status === "error" ? "error" : "done" } : s)),
            }));
          }
          break;
        }
        case "reset": // answer is being revised: the old draft stays until new text arrives
          replaceDraft = true;
          break;
        case "token":
          if (event.draft) {
            draft = replaceDraft ? event.text : draft + event.text;
            replaceDraft = false;
            patch(() => ({ draft, status: "streaming" }));
            break;
          }
          streamed += event.text;
          patch(() => ({ content: streamed, status: "streaming" }));
          break;
        case "final": {
          finished = true;
          const r = event.response;
          if (r.conversation_id) {
            persistConversationId(r.conversation_id);
            refreshConversations();
          }
          if (draft && atBottomRef.current) {
            atBottomRef.current = false;
            showAnswerTopRef.current = true;
          }
          patch((m) => ({
            status: "done",
            draft: "",
            revealed: Boolean(draft),
            content: r.final_answer || streamed || "Agent không trả về nội dung.",
            trace: finishActive(m.trace),
            meta: {
              route: r.route,
              confidence: r.confidence,
              sources: r.sources || [],
              keyThemes: r.key_themes || [],
              processingTime: r.metadata?.processing_time || (performance.now() - startedAt) / 1000,
              toolCalls: r.agent_responses?.tools?.tool_calls || [],
              evaluation: r.evaluation || null,
            },
          }));
          break;
        }
        case "error":
          throw new Error(event.message || "Đã xảy ra lỗi khi xử lý câu hỏi.");
        default:
      }
    };

    try {
      const response = await apiFetch("/api/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          query,
          conversation_id: conversationId,
          attachment_ids: files.map((f) => f.serverId),
          include_metadata: includeMetadata,
          debug,
        }),
      });
      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(
          typeof payload.detail === "string" ? payload.detail : payload.message || `API trả về lỗi ${response.status}.`
        );
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
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
      if (stoppedRef.current) {
        patch((m) => ({ status: "stopped", trace: finishActive(m.trace, "done") }));
      } else {
        const text =
          err?.name === "AbortError"
            ? "Yêu cầu vượt quá thời gian chờ."
            : err instanceof TypeError
              ? `Không kết nối được tới API (${API_BASE_URL}). Kiểm tra backend đã chạy chưa.`
              : err?.message || "Đã xảy ra lỗi không xác định.";
        patch((m) => ({ status: "error", errorText: text, trace: finishActive(m.trace, "error") }));
      }
    } finally {
      clearTimeout(timeout);
      abortRef.current = null;
      stoppedRef.current = false;
      setIsSending(false);
      if (!isMobile) focusInput();
    }
  }

  function retry(assistantId) {
    const failed = messages.find((m) => m.id === assistantId);
    if (!failed?.query || isSending) return;
    // drop the failed exchange (its user turn + the error) and send the same question again
    setMessages((prev) => {
      const i = prev.findIndex((m) => m.id === assistantId);
      return prev.filter((_, idx) => idx !== i && idx !== i - 1);
    });
    submitQuery(failed.query, failed.files || []);
  }

  /* ---------- keyboard shortcuts (same as ChatGPT) */

  useEffect(() => {
    function onKey(e) {
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.shiftKey && e.key.toLowerCase() === "o") {
        e.preventDefault();
        startNewChat();
      } else if (mod && e.shiftKey && e.key.toLowerCase() === "s") {
        e.preventDefault();
        setSidebarOpen((v) => !v);
      } else if (e.key === "Escape" && isMobile && sidebarOpen) {
        setSidebarOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  /* ---------- render */

  const isEmpty = messages.length === 0;
  const offline = health.status === "offline";
  const composer = (
    <Composer
      value={input}
      onChange={setInput}
      onSubmit={submitQuery}
      onStop={stopGeneration}
      isSending={isSending}
      inputRef={inputRef}
      offline={offline}
      attachments={attachments}
    />
  );

  return (
    <div className={`app ${sidebarOpen ? "sidebarOpen" : ""}`}>
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={conversationId}
        onSelect={openConversation}
        onDelete={deleteConversation}
        onNewChat={startNewChat}
        onClose={() => setSidebarOpen(false)}
        health={health}
        user={user}
        settings={{ theme, setTheme, includeMetadata, setIncludeMetadata, debug, setDebug }}
      />
      {isMobile && sidebarOpen && <div className="scrim" onClick={() => setSidebarOpen(false)} />}

      <main className="main">
        <header className="topbar">
          {(!sidebarOpen || isMobile) && (
            <div className="topbarGroup">
              <button
                type="button"
                className="iconAction"
                onClick={() => setSidebarOpen(true)}
                aria-label="Mở thanh bên"
                data-tooltip="Mở thanh bên"
                data-tooltip-pos="bottom"
              >
                <PanelLeft size={18} />
              </button>
              <button
                type="button"
                className="iconAction"
                onClick={startNewChat}
                aria-label="Đoạn chat mới"
                data-tooltip="Đoạn chat mới"
                data-tooltip-pos="bottom"
              >
                <SquarePen size={18} />
              </button>
            </div>
          )}
          <span className="topbarTitle">Multi-RAG Agent</span>
          {health.status !== "online" && health.status !== "checking" && (
            <span className={`statusBanner is-${health.status}`} role="status">
              <WifiOff size={14} />
              {HEALTH_LABELS[health.status]}
            </span>
          )}
        </header>

        {isEmpty ? (
          <div className="home">
            <Greeting />
            {composer}
            <Suggestions onPick={(prompt) => submitQuery(prompt)} />
          </div>
        ) : (
          <>
            <div className="thread" ref={scrollRef} onScroll={handleScroll} aria-live="polite">
              <div className="threadInner">
                {messages.map((m) => (
                  <Message key={m.id} message={m} onRetry={isSending ? undefined : retry} />
                ))}
              </div>
            </div>

            <div className="dock">
              {!atBottom && (
                <button type="button" className="jumpBtn" onClick={() => scrollToBottom("smooth")} aria-label="Cuộn xuống cuối">
                  <ArrowDown size={16} />
                </button>
              )}
              {composer}
              <p className="disclaimer">Agent có thể mắc lỗi. Hãy kiểm tra lại thông tin qua các nguồn được trích dẫn.</p>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
