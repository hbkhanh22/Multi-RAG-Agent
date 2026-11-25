// crypto.randomUUID only exists in secure contexts (https / localhost); fall back for LAN http access.
export const uid = () =>
  globalThis.crypto?.randomUUID?.() ?? `id-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

export function formatSeconds(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return value < 10 ? `${value.toFixed(1)}s` : `${Math.round(value)}s`;
}

export function hostname(url) {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
}

export const storage = {
  get(key, fallback = null) {
    try {
      return localStorage.getItem(key) ?? fallback;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      if (value === null || value === undefined) localStorage.removeItem(key);
      else localStorage.setItem(key, value);
    } catch {
      // private mode / blocked storage
    }
  },
};

/** Sidebar history buckets, newest first (same grouping ChatGPT / Claude use). */
export function groupConversations(conversations) {
  const startOfToday = new Date();
  startOfToday.setHours(0, 0, 0, 0);
  const day = 86400000;
  const buckets = [
    { label: "Hôm nay", from: startOfToday.getTime() },
    { label: "Hôm qua", from: startOfToday.getTime() - day },
    { label: "7 ngày qua", from: startOfToday.getTime() - 7 * day },
    { label: "30 ngày qua", from: startOfToday.getTime() - 30 * day },
    { label: "Cũ hơn", from: -Infinity },
  ].map((b) => ({ ...b, items: [] }));

  for (const c of conversations) {
    const t = new Date(c.updated_at || c.created_at || 0).getTime() || 0;
    buckets.find((b) => t >= b.from).items.push(c);
  }
  return buckets.filter((b) => b.items.length > 0);
}

/** Only real documents / web pages — the API also mixes in synthesizer stats entries. */
export function displayableSources(sources) {
  if (!Array.isArray(sources)) return [];
  const seen = new Set();
  return sources.filter((s) => {
    if (!s || (!s.url && !s.source && !s.title)) return false;
    if (s.retrieved_count != null || s.reranked_count != null) return false;
    const key = `${s.url || s.source}|${s.page ?? ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

export function sourceLink(source) {
  if (source.url) return source.url;
  if (source.source?.startsWith("http")) return source.source;
  return null;
}

export function fileName(path) {
  if (!path) return "Tài liệu nội bộ";
  return path.split(/[\\/]/).pop();
}
