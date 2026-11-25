import { memo, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  Check,
  ChevronDown,
  CircleSlash,
  Copy,
  ExternalLink,
  FileText,
  Globe,
  Info,
  Loader2,
  PencilLine,
  RotateCcw,
  Wrench,
  X,
} from "lucide-react";
import Markdown from "./Markdown.jsx";
import FileChip from "./FileChip.jsx";
import LogoMark from "./LogoMark.jsx";
import { useCopy } from "../lib/useCopy.js";
import { EVAL_CRITERIA, routeInfo } from "../lib/constants.js";
import { displayableSources, fileName, formatPercent, formatSeconds, hostname, sourceLink } from "../lib/utils.js";

function Message({ message, onRetry }) {
  return message.role === "user" ? <UserMessage message={message} /> : <AssistantMessage message={message} onRetry={onRetry} />;
}

export default memo(Message);

/* ---------------------------------------------------------------- user */

function UserMessage({ message }) {
  const [copied, copy] = useCopy();
  return (
    <div className="msg msgUser">
      {message.files?.length > 0 && (
        <ul className="fileList isSent" aria-label="Tệp đính kèm">
          {message.files.map((f) => (
            <FileChip key={f.id} file={f} />
          ))}
        </ul>
      )}
      <div className="userBubble">{message.content}</div>
      <div className="msgActions msgActionsUser">
        <IconAction label={copied ? "Đã sao chép" : "Sao chép"} onClick={() => copy(message.content)}>
          {copied ? <Check size={15} /> : <Copy size={15} />}
        </IconAction>
      </div>
    </div>
  );
}

/* ----------------------------------------------------------- assistant */

function AssistantMessage({ message, onRetry }) {
  const { status = "done", content, meta, trace } = message;
  const [panel, setPanel] = useState(null); // null | "sources" | "details"
  const [copied, copy] = useCopy();

  const running = status === "pending" || status === "streaming";
  const sources = displayableSources(meta?.sources);
  const hasDetails = Boolean(meta && (meta.route || typeof meta.confidence === "number" || meta.evaluation));
  const togglePanel = (name) => setPanel((p) => (p === name ? null : name));

  return (
    <div className="msg msgAssistant">
      <div className="msgAvatar" aria-hidden="true">
        <LogoMark size={16} />
      </div>

      <div className="msgBody">
        {trace?.length > 0 && <ProcessTrace trace={trace} status={status} elapsed={meta?.processingTime} />}

        {message.draft && <DraftPreview text={message.draft} stage={message.stage} running={running} />}

        {content && (
          <div className={`prose ${status === "streaming" ? "isStreaming" : ""} ${message.revealed ? "isRevealed" : ""}`}>
            <Markdown>{content}</Markdown>
          </div>
        )}

        {status === "stopped" && (
          <p className="msgNote">
            <CircleSlash size={14} /> Bạn đã dừng phản hồi này.
          </p>
        )}

        {status === "error" && (
          <div className="errorCard" role="alert">
            <AlertCircle size={16} />
            <div>
              <strong>Không thể hoàn tất câu trả lời</strong>
              <p>{message.errorText}</p>
            </div>
            {onRetry && (
              <button type="button" className="btnSecondary" onClick={() => onRetry(message.id)}>
                <RotateCcw size={14} /> Thử lại
              </button>
            )}
          </div>
        )}

        {!running && status !== "error" && content && (
          <>
            <div className="msgActions">
              <IconAction label={copied ? "Đã sao chép" : "Sao chép"} onClick={() => copy(content)}>
                {copied ? <Check size={15} /> : <Copy size={15} />}
              </IconAction>

              {sources.length > 0 && (
                <button
                  type="button"
                  className={`sourcesTrigger ${panel === "sources" ? "isActive" : ""}`}
                  onClick={() => togglePanel("sources")}
                  aria-expanded={panel === "sources"}
                >
                  <SourceStack sources={sources} />
                  {sources.length} nguồn
                </button>
              )}

              {hasDetails && (
                <IconAction
                  label="Chi tiết xử lý"
                  active={panel === "details"}
                  onClick={() => togglePanel("details")}
                  aria-expanded={panel === "details"}
                >
                  <Info size={15} />
                </IconAction>
              )}

              <MetaSummary meta={meta} />
            </div>

            {panel === "sources" && <SourcesPanel sources={sources} onClose={() => setPanel(null)} />}
            {panel === "details" && <DetailsPanel meta={meta} onClose={() => setPanel(null)} />}
          </>
        )}
      </div>
    </div>
  );
}

function IconAction({ label, active, children, ...rest }) {
  return (
    <button
      type="button"
      className={`iconAction ${active ? "isActive" : ""}`}
      aria-label={label}
      data-tooltip={label}
      {...rest}
    >
      {children}
    </button>
  );
}

/* -------------------------------------------------------------- draft */

/** Text that may still be rewritten: a small muted window, never styled like the answer. */
function DraftPreview({ text, stage, running }) {
  const ref = useRef(null);
  useEffect(() => {
    const el = ref.current;
    if (running && el) el.scrollTop = el.scrollHeight; // follow the latest lines
  }, [text, running]);

  let label = "Bản nháp · đang soạn";
  if (!running) label = "Bản nháp chưa hoàn tất";
  else if (stage === "evaluating") label = "Bản nháp · đang kiểm tra";
  else if (stage === "refining") label = "Bản nháp · đang chỉnh sửa";

  return (
    <section className={`draft ${running ? "isRunning" : ""}`} aria-label={label} aria-live="off">
      <header className="draftHeader">
        <PencilLine size={13} />
        <span className={running ? "shimmerText" : ""}>{label}</span>
      </header>
      <div className="draftBody prose" ref={ref}>
        <Markdown>{text}</Markdown>
      </div>
    </section>
  );
}

/* ------------------------------------------------- agent process trace */

function ProcessTrace({ trace, status, elapsed }) {
  const running = status === "pending" || status === "streaming";
  const [manual, setManual] = useState(null);
  // Open while the agent is still working with nothing to show yet; collapse once text arrives.
  const open = manual ?? status === "pending";
  const active = [...trace].reverse().find((s) => s.status === "active");
  const time = formatSeconds(elapsed);

  let title;
  if (running) title = active?.label || "Đang xử lý";
  else if (status === "stopped") title = "Đã dừng";
  else if (status === "error") title = "Xử lý thất bại";
  else title = time ? `Đã xử lý trong ${time}` : "Đã xử lý";

  return (
    <div className={`trace ${open ? "isOpen" : ""}`}>
      <button type="button" className="traceToggle" onClick={() => setManual(!open)} aria-expanded={open}>
        {running && <Loader2 size={14} className="spin" />}
        <span className={running ? "shimmerText" : ""}>{title}</span>
        <span className="traceCount">· {trace.filter((s) => !s.sub).length} bước</span>
        <ChevronDown size={14} className="chevron" />
      </button>

      {open && (
        <ol className="traceSteps">
          {trace.map((step) => (
            <li key={step.key} className={`traceStep is-${step.status} ${step.sub ? "isSub" : ""}`}>
              <span className="traceIcon">
                {step.status === "active" ? (
                  <Loader2 size={12} className="spin" />
                ) : step.status === "error" ? (
                  <X size={12} />
                ) : step.sub ? (
                  <Wrench size={11} />
                ) : (
                  <Check size={12} />
                )}
              </span>
              <span className="traceLabel">{step.label}</span>
              {step.detail && <span className={`badge tone-${step.tone || "neutral"}`}>{step.detail}</span>}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/* ------------------------------------------------------------- sources */

function SourceStack({ sources }) {
  return (
    <span className="sourceStack" aria-hidden="true">
      {sources.slice(0, 3).map((s, i) => (
        <span key={i} className="sourceDot">
          {sourceLink(s) ? <Globe size={10} /> : <FileText size={10} />}
        </span>
      ))}
    </span>
  );
}

function SourcesPanel({ sources, onClose }) {
  return (
    <section className="panel" aria-label="Nguồn tham chiếu">
      <header className="panelHeader">
        <h3>Nguồn tham chiếu</h3>
        <button type="button" className="iconAction" onClick={onClose} aria-label="Đóng">
          <X size={15} />
        </button>
      </header>
      <ol className="sourceGrid">
        {sources.map((s, i) => {
          const link = sourceLink(s);
          const title = s.title || (link ? hostname(link) : fileName(s.source));
          const origin = link ? hostname(link) : s.agent === "files" ? "Tệp đính kèm" : s.agent ? `Kho ${s.agent}` : "Tài liệu nội bộ";
          const Tag = link ? "a" : "div";
          return (
            <li key={`${link || s.source}-${i}`}>
              <Tag
                className="sourceCard"
                {...(link ? { href: link, target: "_blank", rel: "noopener noreferrer" } : {})}
              >
                <span className="sourceOrigin">
                  <span className="sourceIndex">{i + 1}</span>
                  {link ? <Globe size={12} /> : <FileText size={12} />}
                  <span className="truncate">{origin}</span>
                  {/* PyPDFLoader pages are 0-based */}
                  {typeof s.page === "number" && <span className="sourcePage">tr. {s.page + 1}</span>}
                  {link && <ExternalLink size={12} className="sourceExt" />}
                </span>
                <span className="sourceTitle">{title}</span>
                {s.snippet && <span className="sourceSnippet">{s.snippet}</span>}
              </Tag>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/* ------------------------------------------------------- meta / details */

function MetaSummary({ meta }) {
  if (!meta?.route || meta.route === "error") return null;
  const route = routeInfo(meta.route);
  const time = formatSeconds(meta.processingTime);
  return (
    <span className="metaSummary">
      <span className={`badge tone-${route.tone}`}>{route.label}</span>
      {time && <span>{time}</span>}
      {meta.evaluation?.overall != null && (
        <span className={meta.evaluation.passed ? "textOk" : "textWarn"}>Chất lượng {meta.evaluation.overall}/5</span>
      )}
    </span>
  );
}

function DetailsPanel({ meta, onClose }) {
  const route = routeInfo(meta.route);
  const evaluation = meta.evaluation;
  const scores = Object.entries(evaluation?.scores || {}).filter(([, v]) => v != null);
  const themes = meta.keyThemes || [];
  const tools = meta.toolCalls || [];

  return (
    <section className="panel" aria-label="Chi tiết xử lý">
      <header className="panelHeader">
        <h3>Chi tiết xử lý</h3>
        <button type="button" className="iconAction" onClick={onClose} aria-label="Đóng">
          <X size={15} />
        </button>
      </header>

      <dl className="statGrid">
        <div>
          <dt>Agent xử lý</dt>
          <dd>
            <span className={`badge tone-${route.tone}`}>{route.label}</span>
          </dd>
        </div>
        <div>
          <dt>Độ tin cậy định tuyến</dt>
          <dd>{formatPercent(meta.confidence)}</dd>
        </div>
        <div>
          <dt>Thời gian</dt>
          <dd>{formatSeconds(meta.processingTime) || "—"}</dd>
        </div>
        <div>
          <dt>Tự sửa câu trả lời</dt>
          <dd>{evaluation ? `${evaluation.refine_rounds || 0} lần` : "—"}</dd>
        </div>
      </dl>

      {scores.length > 0 && (
        <div className="panelSection">
          <h4>
            Đánh giá chất lượng
            <span className={evaluation.passed ? "textOk" : "textWarn"}>
              {evaluation.overall}/5 · {evaluation.passed ? "Đạt" : "Chưa đạt ngưỡng"}
            </span>
          </h4>
          <ul className="scoreList">
            {scores.map(([key, value]) => (
              <li key={key}>
                <span>{EVAL_CRITERIA[key] || key}</span>
                <span className="scoreBar" aria-hidden="true">
                  <span style={{ width: `${(value / 5) * 100}%` }} className={value >= 4 ? "isGood" : value >= 3 ? "isMid" : "isLow"} />
                </span>
                <span className="scoreValue">{value}/5</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {tools.length > 0 && (
        <div className="panelSection">
          <h4>Công cụ MCP đã gọi</h4>
          <ul className="toolList">
            {tools.map((t, i) => (
              <li key={i}>
                <span className={`toolState ${t.ok ? "isOk" : "isFail"}`}>{t.ok ? <Check size={12} /> : <X size={12} />}</span>
                <code>
                  {t.server}.{t.tool}
                </code>
                <span className="toolArgs" title={JSON.stringify(t.arguments)}>
                  {JSON.stringify(t.arguments)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {themes.length > 0 && (
        <div className="panelSection">
          <h4>Chủ đề chính</h4>
          <div className="tagList">
            {themes.slice(0, 8).map((t) => (
              <span key={t} className="tag">
                {t}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
