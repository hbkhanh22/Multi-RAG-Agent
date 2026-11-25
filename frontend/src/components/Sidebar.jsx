import { useEffect, useMemo, useRef, useState } from "react";
import { LogOut, Monitor, Moon, PanelLeftClose, Settings, SquarePen, Sun, Trash2 } from "lucide-react";
import LogoMark from "./LogoMark.jsx";
import { logout } from "../lib/api.js";
import { HEALTH_LABELS } from "../lib/constants.js";
import { groupConversations } from "../lib/utils.js";

export default function Sidebar({
  open,
  conversations,
  activeId,
  onSelect,
  onDelete,
  onNewChat,
  onClose,
  settings,
  health,
  user,
}) {
  const groups = useMemo(() => groupConversations(conversations), [conversations]);
  const [confirmId, setConfirmId] = useState(null);

  return (
    <aside className={`sidebar ${open ? "isOpen" : ""}`} aria-label="Thanh bên" inert={!open}>
      <div className="sidebarInner">
        <div className="sidebarHead">
          <div className="brand">
            <span className="brandMark">
              <LogoMark size={15} />
            </span>
            <span className="brandName">Multi-RAG</span>
          </div>
          <button type="button" className="iconAction" onClick={onClose} aria-label="Đóng thanh bên" data-tooltip="Đóng thanh bên" data-tooltip-pos="right">
            <PanelLeftClose size={18} />
          </button>
        </div>

        <nav className="sidebarNav">
          <button type="button" className="navItem" onClick={onNewChat}>
            <SquarePen size={17} />
            <span>Đoạn chat mới</span>
            <span className="navShortcut">Ctrl ⇧ O</span>
          </button>
        </nav>

        <div className="history" onMouseLeave={() => setConfirmId(null)}>
          {groups.length === 0 ? (
            <p className="historyEmpty">Các cuộc trò chuyện của bạn sẽ xuất hiện ở đây.</p>
          ) : (
            groups.map((group) => (
              <section key={group.label} className="historyGroup">
                <h2 className="historyLabel">{group.label}</h2>
                <ul>
                  {group.items.map((c) => {
                    const confirming = confirmId === c.id;
                    return (
                      <li key={c.id} className={`historyItem ${c.id === activeId ? "isActive" : ""} ${confirming ? "isConfirming" : ""}`}>
                        <button type="button" className="historyLink" onClick={() => onSelect(c.id)} title={c.title}>
                          {c.title || "Cuộc trò chuyện"}
                        </button>
                        {confirming ? (
                          <button type="button" className="historyConfirm" onClick={() => onDelete(c.id)} onBlur={() => setConfirmId(null)} autoFocus>
                            Xóa
                          </button>
                        ) : (
                          <button
                            type="button"
                            className="historyDelete"
                            onClick={() => setConfirmId(c.id)}
                            aria-label={`Xóa "${c.title || "cuộc trò chuyện"}"`}
                          >
                            <Trash2 size={14} />
                          </button>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>
            ))
          )}
        </div>

        <SettingsMenu settings={settings} health={health} user={user} />
      </div>
    </aside>
  );
}

const COMPONENT_LABELS = {
  llm_api_key: "LLM API key",
  mcp: "MCP servers",
};

function componentLabel(key) {
  if (COMPONENT_LABELS[key]) return COMPONENT_LABELS[key];
  const m = /^vector_store_(.+)$/.exec(key);
  return m ? `Vector store · ${m[1]}` : key;
}

function componentOk(value) {
  return !["unavailable", "missing", "no tools"].includes(value);
}

function Account({ user }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function leave(everywhere) {
    setBusy(true);
    setError("");
    try {
      await logout({ everywhere });
    } catch {
      setError("Chưa đăng xuất được. Hãy kiểm tra kết nối rồi thử lại.");
      setBusy(false);
    }
  }

  return (
    <div className="menuSection">
      <h3 className="menuLabel">Tài khoản</h3>
      <p className="accountName truncate">
        {user.name || user.email}
        {user.role === "admin" && <span className="badge tone-violet">Quản trị</span>}
      </p>
      <span className="menuMuted truncate" title={user.email}>{user.email}</span>
      <button type="button" className="navItem accountAction" disabled={busy} onClick={() => leave(false)}>
        <LogOut size={15} />
        Đăng xuất
      </button>
      <button type="button" className="navItem accountAction" disabled={busy} onClick={() => leave(true)}>
        <LogOut size={15} />
        Đăng xuất mọi thiết bị
      </button>
      {error && <p className="authError" role="alert">{error}</p>}
    </div>
  );
}

function SettingsMenu({ settings, health, user }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e) => !rootRef.current?.contains(e.target) && setOpen(false);
    const onKey = (e) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const components = Object.entries(health.components || {});

  return (
    <div className="settings" ref={rootRef}>
      {open && (
        <div className="menu" role="dialog" aria-label="Cài đặt">
          {user && <Account user={user} />}
          <div className="menuSection">
            <h3 className="menuLabel">Giao diện</h3>
            <div className="segmented" role="radiogroup" aria-label="Giao diện">
              {[
                { value: "light", label: "Sáng", icon: Sun },
                { value: "dark", label: "Tối", icon: Moon },
                { value: "system", label: "Hệ thống", icon: Monitor },
              ].map(({ value, label, icon: Icon }) => (
                <button
                  key={value}
                  type="button"
                  role="radio"
                  aria-checked={settings.theme === value}
                  className={settings.theme === value ? "isActive" : ""}
                  onClick={() => settings.setTheme(value)}
                >
                  <Icon size={13} />
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="menuSection">
            <h3 className="menuLabel">Phản hồi</h3>
            <Switch
              checked={settings.includeMetadata}
              onChange={settings.setIncludeMetadata}
              label="Thống kê xử lý"
              hint="Trả về thời gian và số liệu truy xuất"
            />
            <Switch checked={settings.debug} onChange={settings.setDebug} label="Chế độ debug" hint="Ghi log chi tiết ở backend" />
          </div>

          <div className="menuSection">
            <h3 className="menuLabel">Trạng thái hệ thống</h3>
            {components.length === 0 ? (
              <p className="menuMuted">{HEALTH_LABELS[health.status]}</p>
            ) : (
              <ul className="healthList">
                {components.map(([key, value]) => (
                  <li key={key}>
                    <span className={`dot ${componentOk(value) ? "isOk" : "isBad"}`} />
                    <span>{componentLabel(key)}</span>
                    <span className="menuMuted">{value}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      <button type="button" className="navItem settingsTrigger" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <Settings size={17} />
        <span>Cài đặt</span>
        <span className="healthPill" title={HEALTH_LABELS[health.status]}>
          <span className={`dot is-${health.status}`} />
          {health.status === "online" ? "Online" : HEALTH_LABELS[health.status]}
        </span>
      </button>
    </div>
  );
}

function Switch({ checked, onChange, label, hint }) {
  return (
    <label className="switchRow">
      <span>
        <span className="switchLabel">{label}</span>
        <span className="menuMuted">{hint}</span>
      </span>
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span className="switch" aria-hidden="true" />
    </label>
  );
}
