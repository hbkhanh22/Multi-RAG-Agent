import { FileSpreadsheet, FileText, Image as ImageIcon, Loader2, X } from "lucide-react";

const isImage = (f) => f.kind === "image" || /\.(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(f.filename);

function size(bytes) {
  if (!bytes) return "";
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** One attached file, in the composer (removable) or in a sent message. */
export default function FileChip({ file, onRemove }) {
  const Icon = isImage(file) ? ImageIcon : /\.(xlsx|csv|tsv)$/i.test(file.filename) ? FileSpreadsheet : FileText;
  let sub = [size(file.size), file.pages && `${file.pages} trang`].filter(Boolean).join(" · ");
  if (file.status === "reading") sub = "Đang đọc nội dung…";
  else if (file.status === "error") sub = file.error;
  else if (file.warning && onRemove) sub = file.warning;

  return (
    <li className={`fileChip ${file.status === "error" ? "isError" : ""}`}>
      <span className="fileThumb">
        {file.status === "reading" ? (
          <Loader2 size={16} className="spin" />
        ) : file.preview ? (
          <img src={file.preview} alt="" />
        ) : (
          <Icon size={18} />
        )}
      </span>
      <span className="fileText">
        <span className="truncate" title={file.filename}>
          {file.filename}
        </span>
        <span className={`fileSub ${file.warning && onRemove ? "isWarn" : ""}`}>{sub}</span>
      </span>
      {onRemove && (
        <button type="button" className="fileRemove" onClick={onRemove} aria-label={`Bỏ tệp ${file.filename}`}>
          <X size={14} />
        </button>
      )}
    </li>
  );
}
