import { useEffect, useState } from "react";
import { Download, ImageOff, Loader2, Maximize2, X } from "lucide-react";
import { apiFetch } from "../lib/api.js";

/** A picture drawn by the agent. Fetched with the session token, so <img src> cannot point at the API. */
export default function GeneratedImage({ image }) {
  const [url, setUrl] = useState(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl;
    (async () => {
      try {
        const res = await apiFetch(`/api/images/${image.id}`, { signal: controller.signal });
        if (!res.ok) throw new Error(String(res.status));
        const blob = await res.blob();
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setUrl(objectUrl);
      } catch {
        if (!controller.signal.aborted) setFailed(true);
      }
    })();
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [image.id]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <figure className="genImage" style={{ aspectRatio: `${image.width} / ${image.height}` }}>
      {url ? (
        <>
          <button type="button" className="genImageOpen" onClick={() => setOpen(true)} aria-label="Phóng to ảnh">
            <img src={url} alt={image.alt} />
          </button>
          <div className="genImageTools">
            <button type="button" onClick={() => setOpen(true)} aria-label="Xem toàn màn hình">
              <Maximize2 size={15} />
            </button>
            <a href={url} download={`anh-${image.id.slice(0, 8)}.jpg`} aria-label="Tải ảnh về">
              <Download size={15} />
            </a>
          </div>
        </>
      ) : (
        <span className="genImageState" role="status">
          {failed ? <ImageOff size={20} /> : <Loader2 size={20} className="spin" />}
          {failed ? "Không tải được ảnh." : "Đang tải ảnh…"}
        </span>
      )}
      {open && (
        <div className="lightbox" role="dialog" aria-modal="true" aria-label={image.alt} onClick={() => setOpen(false)}>
          <img src={url} alt={image.alt} />
          <button type="button" aria-label="Đóng" autoFocus>
            <X size={20} />
          </button>
        </div>
      )}
    </figure>
  );
}
