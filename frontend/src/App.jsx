import { useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Bot,
  BrainCircuit,
  Clock3,
  Loader2,
  Send,
  Sparkles,
  User,
} from "lucide-react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const initialMessages = [
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content: "Xin chao. Hay dat cau hoi, toi se dieu phoi den finance, tech hoac general agent.",
    meta: {
      route: "ready",
      confidence: 1,
    },
  },
];

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState("");
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [debug, setDebug] = useState(false);
  const inputRef = useRef(null);

  const canSend = input.trim().length > 0 && !isSending;

  const latestMeta = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((message) => message.role === "assistant");
    return lastAssistant?.meta || null;
  }, [messages]);

  async function submitMessage(event) {
    event?.preventDefault();
    const query = input.trim();
    if (!query || isSending) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setError("");
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          include_metadata: includeMetadata,
          debug,
        }),
      });

      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || payload.message || "Khong the ket noi den agent.");
      }

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: payload.final_answer || "Agent khong tra ve noi dung.",
          meta: {
            route: payload.route,
            confidence: payload.confidence,
            success: payload.success,
            sources: payload.sources || [],
            keyThemes: payload.key_themes || [],
            processingTime: payload.metadata?.processing_time,
          },
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Da co loi xay ra.";
      setError(message);
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: "Toi chua xu ly duoc cau hoi nay. Kiem tra backend FastAPI va thu lai.",
          meta: {
            route: "error",
            success: false,
          },
        },
      ]);
    } finally {
      setIsSending(false);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }

  function clearChat() {
    setMessages(initialMessages);
    setError("");
    setInput("");
    requestAnimationFrame(() => inputRef.current?.focus());
  }

  return (
    <main className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <BrainCircuit size={24} aria-hidden="true" />
          </div>
          <div>
            <h1>Multi-RAG Chat</h1>
            <p>Gemini agent workspace</p>
          </div>
        </div>

        <section className="statusPanel" aria-label="Agent status">
          <div className="statusHeader">
            <Sparkles size={18} aria-hidden="true" />
            <span>Session</span>
          </div>
          <dl>
            <div>
              <dt>API</dt>
              <dd>{API_BASE_URL}</dd>
            </div>
            <div>
              <dt>Route</dt>
              <dd>{latestMeta?.route || "none"}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{formatPercent(latestMeta?.confidence)}</dd>
            </div>
          </dl>
        </section>

        <section className="controls" aria-label="Request options">
          <label>
            <input
              type="checkbox"
              checked={includeMetadata}
              onChange={(event) => setIncludeMetadata(event.target.checked)}
            />
            <span>Metadata</span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={debug}
              onChange={(event) => setDebug(event.target.checked)}
            />
            <span>Debug</span>
          </label>
          <button type="button" className="secondaryButton" onClick={clearChat}>
            Clear
          </button>
        </section>
      </aside>

      <section className="chatPanel" aria-label="Agent chat">
        <header className="chatHeader">
          <div>
            <h2>Agent Chat</h2>
            <p>Hoi ve tai chinh, cong nghe hoac kien thuc tong quat.</p>
          </div>
          {isSending && (
            <div className="thinking">
              <Loader2 size={16} aria-hidden="true" />
              <span>Dang xu ly</span>
            </div>
          )}
        </header>

        {error && (
          <div className="errorBanner" role="alert">
            <AlertCircle size={18} aria-hidden="true" />
            <span>{error}</span>
          </div>
        )}

        <div className="messages" aria-live="polite">
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="avatar" aria-hidden="true">
                {message.role === "user" ? <User size={18} /> : <Bot size={18} />}
              </div>
              <div className="bubble">
                <div className="messageText">{message.content}</div>
                {message.meta && <MessageMeta meta={message.meta} />}
              </div>
            </article>
          ))}
        </div>

        <form className="composer" onSubmit={submitMessage}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submitMessage();
              }
            }}
            placeholder="Nhap cau hoi cho agent..."
            rows={1}
          />
          <button type="submit" disabled={!canSend} aria-label="Send message">
            {isSending ? <Loader2 size={20} aria-hidden="true" /> : <Send size={20} aria-hidden="true" />}
          </button>
        </form>
      </section>
    </main>
  );
}

function MessageMeta({ meta }) {
  const hasSources = Array.isArray(meta.sources) && meta.sources.length > 0;
  const hasThemes = Array.isArray(meta.keyThemes) && meta.keyThemes.length > 0;

  return (
    <div className="meta">
      <div className="metaRow">
        {meta.route && <span>{meta.route}</span>}
        {typeof meta.confidence === "number" && <span>{formatPercent(meta.confidence)}</span>}
        {typeof meta.processingTime === "number" && (
          <span className="withIcon">
            <Clock3 size={13} aria-hidden="true" />
            {meta.processingTime.toFixed(2)}s
          </span>
        )}
      </div>

      {hasThemes && (
        <div className="chips">
          {meta.keyThemes.slice(0, 5).map((theme) => (
            <span key={theme}>{theme}</span>
          ))}
        </div>
      )}

      {hasSources && (
        <details className="sources">
          <summary>Sources ({meta.sources.length})</summary>
          <ul>
            {meta.sources.slice(0, 6).map((source, index) => (
              <li key={`${source.source || "source"}-${index}`}>
                <strong>{source.agent || "agent"}</strong>
                <span>{source.source || "unknown source"}</span>
                {source.page !== null && source.page !== undefined && <em>p. {source.page}</em>}
              </li>
            ))}
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

export default App;
