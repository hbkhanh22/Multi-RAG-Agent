import { useEffect, useRef, useState } from "react";
import { ArrowUp, Paperclip, Square } from "lucide-react";
import FileChip from "./FileChip.jsx";
import { ATTACH_MAX_FILES, ATTACH_MAX_MB, ATTACH_TYPES } from "../lib/constants.js";

const MAX_LEN = 2000; // matches QueryRequest.query max_length on the API

export default function Composer({ value, onChange, onSubmit, onStop, isSending, inputRef, offline, attachments }) {
  const trimmed = value.trim();
  const { items, ready, busy, notice } = attachments;
  const canSend = (trimmed.length > 0 || ready.length > 0) && !busy && !isSending && value.length <= MAX_LEN;
  const fileRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const addRef = useRef(attachments.add);
  addRef.current = attachments.add;

  // Files can be dropped anywhere in the window (and must never make the browser navigate away).
  useEffect(() => {
    const hasFiles = (e) => e.dataTransfer?.types?.includes("Files");
    const over = (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setDragging(true);
    };
    const leave = (e) => {
      if (!e.relatedTarget) setDragging(false);
    };
    const drop = (e) => {
      if (!hasFiles(e)) return;
      e.preventDefault();
      setDragging(false);
      addRef.current(e.dataTransfer.files);
    };
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
    };
  }, []);

  function onPaste(e) {
    // copying from Word / Excel puts text AND a picture on the clipboard: keep the text
    if (e.clipboardData.files.length > 0 && !e.clipboardData.types.includes("text/plain")) {
      e.preventDefault();
      attachments.add(e.clipboardData.files);
    }
  }
  const nearLimit = value.length > MAX_LEN * 0.9;

  // Grow with content up to a cap, then scroll inside.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 240)}px`;
  }, [value, inputRef]);

  function submit(e) {
    e?.preventDefault();
    if (canSend) onSubmit(trimmed);
  }

  return (
    <form
      className={`composer ${dragging ? "isDragging" : ""}`}
      onSubmit={submit}
      onClick={(e) => (e.target === e.currentTarget || e.target.classList.contains("composerBar")) && inputRef.current?.focus()}
    >
      {dragging && <div className="dropHint">Thả tệp vào đây để đính kèm</div>}
      {items.length > 0 && (
        <ul className="fileList" aria-label="Tệp sẽ gửi">
          {items.map((f) => (
            <FileChip key={f.id} file={f} onRemove={() => attachments.remove(f.id)} />
          ))}
        </ul>
      )}
      {notice && (
        <p className="fileNotice" role="alert">
          {notice}
        </p>
      )}
      <label htmlFor="composer-input" className="srOnly">
        Nhập câu hỏi
      </label>
      <textarea
        id="composer-input"
        ref={inputRef}
        value={value}
        rows={1}
        placeholder={offline ? "Đang mất kết nối với API — bạn vẫn có thể soạn câu hỏi" : "Hỏi bất cứ điều gì, hoặc kéo thả / dán tệp vào đây…"}
        onChange={(e) => onChange(e.target.value)}
        onPaste={onPaste}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
        }}
      />
      <div className="composerBar">
        <input
          ref={fileRef}
          type="file"
          multiple
          hidden
          accept={ATTACH_TYPES.join(",")}
          onChange={(e) => {
            attachments.add(e.target.files);
            e.target.value = ""; // the same file can be picked again
          }}
        />
        <button
          type="button"
          className="iconAction"
          onClick={() => fileRef.current?.click()}
          disabled={items.length >= ATTACH_MAX_FILES}
          aria-label="Đính kèm tệp"
          data-tooltip={`Đính kèm tệp (tối đa ${ATTACH_MAX_FILES} tệp, ${ATTACH_MAX_MB} MB/tệp)`}
        >
          <Paperclip size={18} />
        </button>
        <span className={`composerHint ${nearLimit ? "isWarn" : ""}`}>
          {busy ? (
            "Đang đọc tệp…"
          ) : nearLimit ? (
            `${value.length}/${MAX_LEN} ký tự`
          ) : (
            <>
              <kbd>Enter</kbd> gửi · <kbd>Shift</kbd>+<kbd>Enter</kbd> xuống dòng
            </>
          )}
        </span>
        {isSending ? (
          <button type="button" className="sendBtn isStop" onClick={onStop} aria-label="Dừng phản hồi" data-tooltip="Dừng">
            <Square size={13} fill="currentColor" />
          </button>
        ) : (
          <button type="submit" className="sendBtn" disabled={!canSend} aria-label="Gửi" data-tooltip="Gửi">
            <ArrowUp size={18} strokeWidth={2.25} />
          </button>
        )}
      </div>
    </form>
  );
}
