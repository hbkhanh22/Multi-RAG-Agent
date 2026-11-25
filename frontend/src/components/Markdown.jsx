import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Check, Copy } from "lucide-react";
import { useCopy } from "../lib/useCopy.js";

function CodeBlock({ children }) {
  const child = Array.isArray(children) ? children[0] : children;
  const className = child?.props?.className || "";
  const lang = /language-([\w+#-]+)/.exec(className)?.[1];
  const raw = child?.props?.children;
  const text = (Array.isArray(raw) ? raw.join("") : String(raw ?? "")).replace(/\n$/, "");
  const [copied, copy] = useCopy();

  return (
    <div className="codeBlock">
      <div className="codeHeader">
        <span>{lang || "code"}</span>
        <button type="button" className="codeCopy" onClick={() => copy(text)}>
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? "Đã sao chép" : "Sao chép"}
        </button>
      </div>
      <pre>
        <code className={className}>{text}</code>
      </pre>
    </div>
  );
}

const components = {
  pre: CodeBlock,
  table: ({ node, ...props }) => (
    <div className="tableScroll">
      <table {...props} />
    </div>
  ),
  a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
};

function Markdown({ children }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {children}
    </ReactMarkdown>
  );
}

export default memo(Markdown);
