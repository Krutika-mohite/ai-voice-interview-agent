import io
import os

from pypdf import PdfReader


def _read_pdf(source) -> str:
    reader = PdfReader(source)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_resume(file_path):
    """Load resume text.

    Accepts (as before) a file path, and now also raw bytes or a file-like object
    (e.g. a Streamlit upload). Supports .pdf (as before) and now .txt.
    Returns "" on failure, like the original.
    """
    try:
        # Raw bytes / file-like (Streamlit UploadedFile) - processed in memory.
        if isinstance(file_path, (bytes, bytearray)):
            data = bytes(file_path)
            return _read_pdf(io.BytesIO(data)) if data[:5] == b"%PDF-" else data.decode("utf-8", "ignore")
        if hasattr(file_path, "read"):
            name = getattr(file_path, "name", "") or ""
            data = file_path.read()
            if name.lower().endswith(".pdf") or data[:5] == b"%PDF-":
                return _read_pdf(io.BytesIO(data))
            return data.decode("utf-8", "ignore") if isinstance(data, bytes) else str(data)

        # Path on disk.
        if os.path.splitext(str(file_path))[1].lower() == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        return _read_pdf(file_path)

    except Exception as e:
        print(f"Error reading resume: {e}")
        return ""
