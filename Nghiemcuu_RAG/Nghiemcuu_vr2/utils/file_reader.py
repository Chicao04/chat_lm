from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader

def load_file_text(path: str | Path) -> str:
    """
    Đọc file PDF, DOCX, TXT… và trả về chuỗi <~12k token.
    Chấp nhận cả str và pathlib.Path.
    """
    path = Path(path)                 # đảm bảo là Path object
    suffix = path.suffix.lower()

    # --- chọn loader ---
    if suffix == ".pdf":
        pages = PyPDFLoader(str(path)).load()
    else:
        pages = UnstructuredFileLoader(str(path)).load()

    # Ghép nội dung & cắt bớt nếu quá dài
    full_text = "\n".join(p.page_content for p in pages)
    return full_text[:12000]






