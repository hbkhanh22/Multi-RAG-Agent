import { useCallback, useEffect, useRef, useState } from "react";
import { LoaderCircle, WifiOff } from "lucide-react";
import App from "../App.jsx";
import LoginPage from "./LoginPage.jsx";
import { authConfig, endSession, onSession, refreshSession } from "../lib/api.js";
import { useTheme } from "../lib/useTheme.js";
import { storage } from "../lib/utils.js";

const SIGNOUT_KEY = "rag_signout"; // tells the other tabs

const HASH_ERRORS = {
  google_cancelled: "Bạn đã hủy đăng nhập bằng Google.",
  account: "Không thể dùng tài khoản Google này.",
  google: "Đăng nhập bằng Google không thành công. Hãy thử lại.",
};

function takeHashError() {
  const { hash } = window.location;
  if (!/^#(auth_error=|signed_in)/.test(hash)) return "";
  window.history.replaceState(null, "", window.location.pathname + window.location.search);
  const code = hash.split("auth_error=")[1];
  return code ? HASH_ERRORS[code] || HASH_ERRORS.google : "";
}

/** Shows the chat when the server has no authentication, or to a signed-in user. */
export default function AuthGate() {
  const theme = useTheme();
  const [config, setConfig] = useState(null);
  const [status, setStatus] = useState("loading"); // loading | unreachable | ready
  const [user, setUser] = useState(null);
  const [expiredUser, setExpiredUser] = useState(null); // keeps the chat (and its draft) behind the login form
  const [notice, setNotice] = useState("");
  const userRef = useRef(null);

  const start = useCallback(async () => {
    setStatus("loading");
    try {
      const cfg = await authConfig();
      setConfig(cfg);
      if (cfg.enabled) await refreshSession();
      setStatus("ready");
    } catch {
      setStatus("unreachable");
    }
  }, []);

  useEffect(() => {
    const off = onSession((next, reason) => {
      const previous = userRef.current;
      userRef.current = next;
      if (next) {
        // another account: forget the open conversation of the previous one
        const owner = storage.get("rag_user_id");
        if (owner && owner !== next.id) storage.set("rag_conversation_id", null);
        storage.set("rag_user_id", next.id);
        setExpiredUser(null);
        setNotice("");
      } else if (reason === "signed_out") {
        storage.set("rag_conversation_id", null);
        setExpiredUser(null);
        setNotice("");
        if (previous) storage.set(SIGNOUT_KEY, String(Date.now()));
      } else if (previous) {
        setExpiredUser(previous);
        setNotice("Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại.");
      }
      setUser(next);
    });
    const onStorage = (e) => {
      if (e.key === SIGNOUT_KEY && userRef.current) {
        userRef.current = null; // do not echo the message back
        endSession();
      }
    };
    window.addEventListener("storage", onStorage);
    const hashError = takeHashError();
    if (hashError) setNotice(hashError);
    start();
    return () => {
      off();
      window.removeEventListener("storage", onStorage);
    };
  }, [start]);

  if (status === "loading") {
    return (
      <div className="authScreen" role="status" aria-label="Đang tải">
        <LoaderCircle size={22} className="spin" />
      </div>
    );
  }
  if (status === "unreachable") {
    return (
      <div className="authScreen">
        <div className="authCard authCenter">
          <WifiOff size={22} />
          <h1>Không kết nối được máy chủ</h1>
          <p className="authHint">Hãy kiểm tra API đã chạy chưa rồi thử lại.</p>
          <button type="button" className="authSubmit" onClick={start}>
            Thử lại
          </button>
        </div>
      </div>
    );
  }
  if (!config.enabled) return <App theme={theme} />;

  const shown = user || expiredUser;
  return (
    <>
      {shown && (
        <div hidden={!user}>
          <App key={shown.id} theme={theme} user={shown} />
        </div>
      )}
      {!user && <LoginPage config={config} notice={notice} initialEmail={expiredUser?.email} />}
    </>
  );
}
