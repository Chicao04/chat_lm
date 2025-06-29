import os
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain.docstore.document import Document


def load_file(full_path):
    try:
        if full_path.endswith(".pdf"):
            loader = PyMuPDFLoader(full_path)
        elif full_path.endswith(".txt"):
            loader = TextLoader(full_path, encoding="utf-8")
        else:
            return []
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(full_path)
        return docs
    except Exception as e:
        print(f"❌ Lỗi khi đọc {full_path}: {e}")
        return []


def load_multimodal_data(path):
    all_files = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith((".txt", ".pdf")):
                all_files.append(os.path.join(root, f))

    all_docs = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        for docs in executor.map(load_file, all_files):
            all_docs.extend(docs)

    print(f"✅ Đã load {len(all_docs)} tài liệu từ {len(all_files)} file.")
    return all_docs
