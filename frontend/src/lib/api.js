import { API_BASE_URL } from "./constants.js";

/* Session: the access token (15 min) lives in memory only, never in localStorage. The refresh
   token is an httpOnly cookie that only /api/auth receives and JavaScript cannot read. */

let accessToken = null;
let expiresAt = 0;
let user = null;
let refreshing = null;
let epoch = 0; // bumped on sign-in / sign-out: a refresh started before it must not undo it
let listener = () => {};

/** `fn(user, reason)`; user is null when signed out, reason is "expired" or "signed_out". */
export function onSession(fn) {
  listener = fn;
  return () => {
    listener = () => {};
  };
}

function setSession(data) {
  accessToken = data.access_token;
  expiresAt = Date.now() + data.expires_in * 1000;
  user = data.user;
  listener(user);
}

function clearSession(reason) {
  accessToken = null;
  user = null;
  listener(null, reason);
}

const raw = (path, options) => fetch(`${API_BASE_URL}${path}`, { ...options, credentials: "include" });

async function errorOf(res) {
  const data = await res.json().catch(() => ({}));
  if (res.status === 429) {
    const wait = Number(res.headers.get("Retry-After"));
    return new Error(`Bạn đã thử quá nhiều lần. Hãy đợi ${wait > 0 ? `${wait} giây` : "ít phút"} rồi thử lại.`);
  }
  if (res.status === 422) return new Error("Thông tin chưa hợp lệ. Hãy kiểm tra lại các ô đã nhập.");
  if (res.status === 403) return new Error("Hệ thống hiện không cho tự đăng ký tài khoản.");
  return new Error(typeof data.detail === "string" ? data.detail : `Máy chủ gặp lỗi (${res.status}). Hãy thử lại.`);
}

export async function authConfig() {
  const res = await raw("/api/auth/config");
  if (!res.ok) throw new Error(`API trả về lỗi ${res.status}.`);
  return res.json();
}

async function doRefresh() {
  const started = epoch;
  // Tabs share the cookie: one refresh at a time (locks only exist on https / localhost)
  const call = () => raw("/api/auth/refresh", { method: "POST" });
  const res = await (navigator.locks ? navigator.locks.request("rag_refresh", call) : call());
  if (res.status === 401) {
    if (epoch === started) clearSession("expired");
    return false;
  }
  if (!res.ok) throw new Error(`API trả về lỗi ${res.status}.`); // server trouble is not a sign-out
  const data = await res.json();
  if (epoch !== started) return Boolean(accessToken);
  setSession(data);
  return true;
}

/** Single flight. True when signed in, false when there is no session; throws if unreachable. */
export function refreshSession() {
  refreshing ??= doRefresh().finally(() => {
    refreshing = null;
  });
  return refreshing;
}

async function signIn(path, body) {
  const res = await raw(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await errorOf(res);
  epoch += 1;
  setSession(await res.json());
}

export const login = (email, password) => signIn("/api/auth/login", { email, password });
export const register = (name, email, password) => signIn("/api/auth/register", { name, email, password });
export const googleLoginUrl = `${API_BASE_URL}/api/auth/google/login`;

/** Throws when the server was not reached: the cookie would still be valid, so stay signed in. */
export async function logout({ everywhere = false } = {}) {
  const res = everywhere
    ? await apiFetch("/api/auth/logout-all", { method: "POST" })
    : await raw("/api/auth/logout", { method: "POST" });
  if (!res.ok) throw await errorOf(res);
  endSession();
}

/** Local part of a sign-out (also when another tab signed out). */
export function endSession() {
  epoch += 1;
  clearSession("signed_out");
}

/** fetch() to the API with the access token; renews it when it is (about to be) expired. */
export async function apiFetch(path, options = {}) {
  const owner = user?.id;
  const send = () =>
    fetch(`${API_BASE_URL}${path}`, {
      ...options,
      credentials: "include",
      headers: { ...options.headers, ...(accessToken ? { Authorization: `Bearer ${accessToken}` } : {}) },
    });
  const renew = async () => {
    if (!(await refreshSession()) || user?.id !== owner) {
      // signed out, or another account signed in from another tab: never send this as someone else
      throw new Error("Phiên đăng nhập đã thay đổi. Hãy đăng nhập và thử lại.");
    }
    options.signal?.throwIfAborted();
  };

  if (!owner) return send(); // authentication is off on the server
  let renewed = false;
  if (Date.now() > expiresAt - 30000) {
    await renew();
    renewed = true;
  }
  const res = await send();
  if (res.status !== 401 || renewed) return res;
  await renew();
  return send();
}
