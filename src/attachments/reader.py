"""
Turn an uploaded file into text segments: [{"page": 0-based int | None, "label": str | None, "text": str}].

  text / code / csv / json  decoded                 html        visible text
  pdf                       text layer; pages without one are rendered and read by the vision model
  docx, xlsx, pptx          paragraphs, sheets, slides
  images                    vision model: transcription + description

Files are untrusted: sizes, page counts and uncompressed sizes are bounded.
"""

import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from config.settings import settings
from src.attachments import vision

logger = logging.getLogger(__name__)

Segment = Dict[str, Any]

IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
OFFICE = {".docx", ".xlsx", ".pptx"}
HTML = {".html", ".htm"}
TEXT = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".log", ".xml", ".yaml", ".yml", ".toml", ".ini",
    ".sql", ".py", ".js", ".jsx", ".ts", ".tsx", ".css", ".java", ".c", ".cpp", ".h", ".cs", ".go", ".rs",
    ".php", ".rb", ".sh", ".ps1", ".ipynb", ".tex", ".srt",
}
MAX_UNZIPPED = 300 * 1024 * 1024
MIN_PAGE_CHARS = 30          # a PDF page with less text than this is treated as a scan
EXTRACT_SECONDS = 60         # stop reading a huge / pathological document


class UnsupportedFile(Exception):
    """File type we do not read."""


class UnreadableFile(Exception):
    """Right type, but broken, encrypted or without any content. The message is shown to the user."""


@dataclass
class ReadResult:
    kind: str                                # image | pdf | document | text
    segments: List[Segment]
    pages: Optional[int] = None
    truncated: bool = False
    warning: Optional[str] = None
    image: Optional[bytes] = field(default=None, repr=False)   # normalised JPEG, images only

    @property
    def chars(self) -> int:
        return sum(len(s["text"]) for s in self.segments)


def accepted_types() -> List[str]:
    return sorted(IMAGES | OFFICE | HTML | TEXT | {".pdf"})


def read_file(filename: str, data: bytes) -> ReadResult:
    ext = Path(filename).suffix.lower()
    if ext in IMAGES:
        result = _read_image(data)
    elif ext == ".pdf":
        result = _read_pdf(data)
    elif ext in OFFICE:
        result = _read_office(ext, data)
    elif ext in HTML:
        result = ReadResult("text", _whole(_html_text(_decode(data))))
    elif ext in TEXT:
        result = ReadResult("text", _whole(_decode(data)))
    else:
        raise UnsupportedFile(ext or filename)

    # NUL (and lone surrogates, from odd PDF fonts) cannot be stored in Postgres
    for s in result.segments:
        s["text"] = s["text"].encode("utf-8", "replace").decode("utf-8").replace("\x00", "").strip()
    _limit(result)
    result.segments = [s for s in result.segments if s["text"]]
    if not result.segments:
        raise UnreadableFile("Tệp không có nội dung đọc được.")
    if result.truncated and not result.warning:
        result.warning = "Tệp quá dài, chỉ phần đầu được đọc."
    return result


def _limit(result: ReadResult) -> None:
    left = settings.attachment_max_chars
    for s in result.segments:
        if len(s["text"]) > left:
            s["text"] = s["text"][:left]
            result.truncated = True
        left -= len(s["text"])


def _seg(text: str, page: Optional[int] = None, label: Optional[str] = None) -> Segment:
    return {"page": page, "label": label, "text": text}


def _whole(text: str) -> List[Segment]:
    return [_seg(text)]


# --------------------------------------------------------------------- text

def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:8192]:
        raise UnreadableFile("Tệp là dữ liệu nhị phân, không phải văn bản.")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1258", errors="replace")   # Windows Vietnamese


def _html_text(html: str) -> str:
    from lxml import html as lh

    try:
        root = lh.document_fromstring(html.encode("utf-8"), parser=lh.HTMLParser(encoding="utf-8"))
    except Exception as e:
        raise UnreadableFile("Không đọc được tệp HTML này.") from e
    for el in root.xpath("//script|//style|//noscript"):
        el.drop_tree()
    lines = (" ".join(t.split()) for t in root.itertext())
    return "\n".join(line for line in lines if line)


# -------------------------------------------------------------------- image

def _read_image(data: bytes) -> ReadResult:
    try:
        jpeg = vision.to_jpeg(data)
    except ValueError as e:
        raise UnreadableFile("Ảnh bị hỏng, quá lớn hoặc không đúng định dạng.") from e
    return ReadResult("image", _whole(vision.transcribe(jpeg)), image=jpeg)


# ---------------------------------------------------------------------- pdf

def _read_pdf(data: bytes) -> ReadResult:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise UnreadableFile("PDF được bảo vệ bằng mật khẩu.")
        total = len(reader.pages)
    except UnreadableFile:
        raise
    except Exception as e:
        raise UnreadableFile("PDF bị hỏng hoặc không đúng định dạng.") from e

    result = ReadResult("pdf", [], pages=total)
    started, used = time.time(), 0
    for i in range(min(total, settings.attachment_max_pages)):
        if time.time() - started > EXTRACT_SECONDS or used >= settings.attachment_max_chars:
            break
        try:
            text = (reader.pages[i].extract_text() or "").strip()
        except Exception as e:
            logger.warning(f"PDF page {i + 1}: {type(e).__name__}: {e}")
            text = ""
        used += len(text)
        result.segments.append(_seg(text, page=i))
    warnings = []
    if len(result.segments) < total:
        result.truncated = True
        warnings.append(f"Chỉ đọc {len(result.segments)}/{total} trang đầu.")

    scans = [s for s in result.segments if len(s["text"]) < MIN_PAGE_CHARS]
    todo = scans[: settings.attachment_max_scan_pages]
    if todo:
        unread = _read_scanned_pages(data, todo)
        if unread == len(todo) and not any(s["text"] for s in result.segments):
            raise vision.VisionUnavailable("no scanned page could be read")
        if unread:
            warnings.append(f"{unread} trang scan không đọc được.")
    if len(scans) > len(todo):
        result.truncated = True
        warnings.append(f"PDF scan: chỉ nhận dạng {len(todo)}/{len(scans)} trang.")
    result.warning = " ".join(warnings) or None
    return result


def _read_scanned_pages(data: bytes, segments: List[Segment]) -> int:
    """Fill the segments with the vision model's transcription; returns how many failed."""
    images = _render_pages(data, [s["page"] for s in segments])

    def read(image: Optional[bytes]) -> Optional[str]:
        if image is None:
            return None
        try:
            return vision.transcribe(image)
        except vision.VisionUnavailable:
            return None

    failed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for s, text in zip(segments, pool.map(read, images)):
            if text is None:
                failed += 1
            else:
                s["text"] = text
    return failed


def _render_pages(data: bytes, pages: List[int]) -> List[Optional[bytes]]:
    import pypdfium2 as pdfium

    images: List[Optional[bytes]] = []
    try:
        with vision.PDFIUM_LOCK:          # pdfium is not thread-safe (uploads run in a thread pool)
            pdf = pdfium.PdfDocument(data)
            try:
                for number in pages:
                    page = pdf[number]
                    width, height = page.get_size()
                    scale = min(2.0, vision.MAX_SIDE / max(width, height, 1))
                    images.append(vision.pil_to_jpeg(page.render(scale=scale).to_pil()))
            finally:
                pdf.close()
    except Exception as e:
        logger.warning(f"Could not render scanned PDF: {type(e).__name__}: {e}")
    return images + [None] * (len(pages) - len(images))


# ------------------------------------------------------------------- office

def _read_office(ext: str, data: bytes) -> ReadResult:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNZIPPED:
                raise UnreadableFile("Tệp quá lớn sau khi giải nén.")
        if ext == ".docx":
            import docx2txt
            return ReadResult("document", _whole(docx2txt.process(io.BytesIO(data)) or ""))
        if ext == ".xlsx":
            return ReadResult("document", list(_sheets(data)))
        slides = list(_slides(data))
        return ReadResult("document", slides, pages=len(slides))
    except UnreadableFile:
        raise
    except Exception as e:
        logger.warning(f"Unreadable {ext}: {type(e).__name__}: {e}")
        raise UnreadableFile(f"Tệp {ext} bị hỏng, có mật khẩu hoặc không đúng định dạng.") from e


def _sheets(data: bytes) -> Iterator[Segment]:
    from openpyxl import load_workbook

    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    budget = settings.attachment_max_chars * 2      # over the limit, so _limit() marks it truncated
    started = time.time()
    try:
        for ws in wb.worksheets:
            rows: List[str] = []
            for row in ws.iter_rows(values_only=True, max_col=100):
                cells = ["" if v is None else " ".join(str(v).split()) for v in row]
                if any(cells):
                    line = " | ".join(cells).rstrip(" |")
                    rows.append(line)
                    budget -= len(line) + 1
                if budget <= 0 or time.time() - started > EXTRACT_SECONDS:
                    break
            yield _seg("\n".join(rows), label=ws.title)
            if budget <= 0:
                break
    finally:
        wb.close()


def _slides(data: bytes) -> Iterator[Segment]:
    from pptx import Presentation

    def texts(shapes) -> Iterator[str]:
        for sh in shapes:
            if hasattr(sh, "shapes"):                 # group
                yield from texts(sh.shapes)
            elif getattr(sh, "has_table", False) and sh.has_table:
                for row in sh.table.rows:
                    yield " | ".join(c.text for c in row.cells)
            elif sh.has_text_frame:
                yield sh.text_frame.text

    for i, slide in enumerate(Presentation(io.BytesIO(data)).slides):
        parts = [t for t in texts(slide.shapes) if t.strip()]
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text
            if notes.strip():
                parts.append(f"Ghi chú: {notes}")
        yield _seg("\n".join(parts), page=i)
