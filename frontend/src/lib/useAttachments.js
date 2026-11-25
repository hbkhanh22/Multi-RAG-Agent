import { useCallback, useRef, useState } from "react";
import { apiFetch } from "./api.js";
import { ATTACH_MAX_FILES, ATTACH_MAX_MB, ATTACH_TYPES } from "./constants.js";
import { uid } from "./utils.js";

const extension = (name) => (name.match(/\.[^.]+$/)?.[0] || "").toLowerCase();

function discardOnServer(serverId) {
  apiFetch(`/api/attachments/${serverId}`, { method: "DELETE" }).catch(() => {});
}

/** Files in the composer. Each one is uploaded and read by the API as soon as it is added. */
export function useAttachments() {
  const [items, setItems] = useState([]);
  const [notice, setNotice] = useState("");
  const itemsRef = useRef(items); // latest list, also inside async callbacks

  const update = useCallback((fn) => {
    itemsRef.current = fn(itemsRef.current);
    setItems(itemsRef.current);
  }, []);

  const upload = useCallback(
    async (item, file) => {
      const patch = (values) => update((prev) => prev.map((f) => (f.id === item.id ? { ...f, ...values } : f)));
      try {
        const body = new FormData();
        body.append("file", file, item.filename);
        const res = await apiFetch("/api/attachments", {
          method: "POST",
          body,
          signal: item.controller.signal,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          const text =
            res.status === 429
              ? "Gửi quá nhanh, hãy thử lại sau ít phút."
              : typeof data.detail === "string"
                ? data.detail
                : res.status === 413
                  ? "Tệp quá lớn."
                  : `Không đọc được tệp (lỗi ${res.status}).`;
          throw new Error(text);
        }
        // removed while the last bytes arrived: do not leave it on the server
        if (!itemsRef.current.some((f) => f.id === item.id)) return discardOnServer(data.id);
        const { id, ...info } = data;
        patch({ ...info, serverId: id, status: "ready" });
      } catch (err) {
        if (err?.name === "AbortError") return;
        patch({ status: "error", error: err instanceof TypeError ? "Không kết nối được tới API." : err.message });
      }
    },
    [update]
  );

  const add = useCallback(
    (fileList) => {
      const problems = [];
      for (const file of Array.from(fileList || [])) {
        const filename = file.name || "anh.png";
        if (itemsRef.current.length >= ATTACH_MAX_FILES) {
          problems.push(`Tối đa ${ATTACH_MAX_FILES} tệp mỗi tin nhắn.`);
          break;
        }
        if (!ATTACH_TYPES.includes(extension(filename))) {
          problems.push(`${filename}: định dạng chưa được hỗ trợ.`);
          continue;
        }
        if (file.size > ATTACH_MAX_MB * 1024 * 1024) {
          problems.push(`${filename}: lớn hơn ${ATTACH_MAX_MB} MB.`);
          continue;
        }
        const item = {
          id: uid(),
          filename,
          size: file.size,
          status: "reading",
          preview: file.type.startsWith("image/") && !/tiff/.test(file.type) ? URL.createObjectURL(file) : null,
          controller: new AbortController(),
        };
        update((prev) => [...prev, item]);
        upload(item, file);
      }
      setNotice(problems.join(" "));
    },
    [update, upload]
  );

  const drop = useCallback((item) => {
    item.controller.abort();
    if (item.preview) URL.revokeObjectURL(item.preview);
    if (item.serverId) discardOnServer(item.serverId);
  }, []);

  const remove = useCallback(
    (id) => {
      const item = itemsRef.current.find((f) => f.id === id);
      update((prev) => prev.filter((f) => f.id !== id));
      if (item) drop(item);
      setNotice("");
    },
    [update, drop]
  );

  /** Empty the composer; `discard` also deletes the files (they were not sent). */
  const clear = useCallback(
    ({ discard = false } = {}) => {
      if (discard) itemsRef.current.forEach(drop);
      update(() => []);
      setNotice("");
    },
    [update, drop]
  );

  return {
    items,
    ready: items.filter((f) => f.status === "ready"),
    busy: items.some((f) => f.status === "reading"),
    notice,
    add,
    remove,
    clear,
  };
}
