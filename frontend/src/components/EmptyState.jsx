import { FileSearch, Globe, Paperclip, Wrench } from "lucide-react";
import { SUGGESTIONS } from "../lib/constants.js";

const CAPABILITIES = [
  { icon: FileSearch, label: "Tài liệu Tech & Sport" },
  { icon: Globe, label: "Tìm kiếm web" },
  { icon: Wrench, label: "Công cụ MCP" },
  { icon: Paperclip, label: "Đọc PDF, ảnh, Word, Excel…" },
];

export function Greeting() {
  return (
    <div className="greeting">
      <h1>Hôm nay bạn muốn tìm hiểu gì?</h1>
      <ul className="capabilities" aria-label="Khả năng của agent">
        {CAPABILITIES.map(({ icon: Icon, label }) => (
          <li key={label}>
            <Icon size={14} />
            {label}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function Suggestions({ onPick }) {
  return (
    <ul className="suggestions" aria-label="Gợi ý câu hỏi">
      {SUGGESTIONS.map(({ icon: Icon, tone, label, prompt }) => (
        <li key={label}>
          <button type="button" className="suggestion" onClick={() => onPick(prompt)} title={prompt}>
            <Icon size={16} className={`text-${tone}`} />
            <span>{label}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}
