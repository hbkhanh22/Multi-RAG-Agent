import { useState } from "react";
import { Eye, EyeOff, LoaderCircle } from "lucide-react";
import LogoMark from "./LogoMark.jsx";
import { googleLoginUrl, login, register } from "../lib/api.js";

export default function LoginPage({ config, notice, initialEmail = "" }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState(initialEmail);
  const [password, setPassword] = useState("");
  const [visible, setVisible] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const registering = mode === "register";

  function switchMode() {
    setMode(registering ? "login" : "register");
    setError("");
    setPassword("");
    setVisible(false);
  }

  async function submit(e) {
    e.preventDefault();
    if (busy) return;
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(email.trim())) return setError("Email không hợp lệ.");
    if (registering && password.length < config.min_password_length) {
      return setError(`Mật khẩu cần ít nhất ${config.min_password_length} ký tự.`);
    }
    if (!password) return setError("Hãy nhập mật khẩu.");
    setError("");
    setBusy(true);
    try {
      if (registering) await register(name.trim(), email.trim(), password);
      else await login(email.trim(), password); // success unmounts this page
    } catch (err) {
      setError(err instanceof TypeError ? "Không kết nối được máy chủ. Hãy thử lại." : err.message);
      setBusy(false);
    }
  }

  return (
    <main className="authScreen">
      <div className="authCard">
        <span className="authLogo">
          <LogoMark size={22} />
        </span>
        <h1>{registering ? "Tạo tài khoản" : "Đăng nhập"}</h1>
        {notice && !error && <p className="authNotice" role="status">{notice}</p>}

        <form onSubmit={submit} noValidate>
          {registering && (
            <label className="authField">
              <span>Tên hiển thị (không bắt buộc)</span>
              <input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" maxLength={200} />
            </label>
          )}
          <label className="authField">
            <span>Email</span>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="username"
              maxLength={320}
              autoFocus={!initialEmail}
            />
          </label>
          <div className="authField">
            <label htmlFor="auth-password">Mật khẩu</label>
            <div className="authPassword">
              <input
                id="auth-password"
                type={visible ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={registering ? "new-password" : "current-password"}
                maxLength={256}
                autoFocus={Boolean(initialEmail)}
              />
              <button
                type="button"
                className="iconAction"
                onClick={() => setVisible((v) => !v)}
                aria-label={visible ? "Ẩn mật khẩu" : "Hiện mật khẩu"}
              >
                {visible ? <EyeOff size={17} /> : <Eye size={17} />}
              </button>
            </div>
            {registering && <span className="authHint">Ít nhất {config.min_password_length} ký tự</span>}
          </div>

          {error && <p className="authError" role="alert">{error}</p>}
          <button type="submit" className="authSubmit" disabled={busy}>
            {busy && <LoaderCircle size={16} className="spin" />}
            {registering ? "Tạo tài khoản" : "Đăng nhập"}
          </button>
        </form>

        {config.google && (
          <>
            <div className="authOr">hoặc</div>
            <a className="authGoogle" href={googleLoginUrl}>
              Tiếp tục với Google
            </a>
          </>
        )}

        {config.registration && (
          <p className="authSwitch">
            {registering ? "Đã có tài khoản?" : "Chưa có tài khoản?"}{" "}
            <button type="button" onClick={switchMode}>
              {registering ? "Đăng nhập" : "Đăng ký"}
            </button>
          </p>
        )}
        {!registering && <p className="authHint authCenter">Quên mật khẩu? Hãy liên hệ quản trị viên để đặt lại.</p>}
      </div>
    </main>
  );
}
