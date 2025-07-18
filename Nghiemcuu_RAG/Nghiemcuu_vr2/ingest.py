import sys, pathlib
from utils.file_reader import load_file_text
from utils.qa_generator import generate_qa_pairs
from utils.db_utils import save_qas
from utils.classify_domain import process_file_with_topic  # đã đổi vai trò

def ingest(file_path: pathlib.Path, user_topic: str):
    print(f"📄 Đang xử lý: {file_path}")
    text = load_file_text(file_path)

    # Dùng chủ đề do người dùng truyền vào
    print(f"📂 Sử dụng chủ đề người dùng chọn: {user_topic}")
    _, db_schema = process_file_with_topic(text, user_topic)

    qa_pairs = generate_qa_pairs(text, max_q=100)
    print(f"✅ Sinh {len(qa_pairs)} cặp Q‑A")

    save_qas(db_schema, qa_pairs)
    print("💾 Đã ghi vào MySQL!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ingest.py <file_path> <chủ_đề>")
        print("Ví dụ: python ingest.py Hienmau.pdf \"Y tế\"")
        sys.exit(1)

    file_path = pathlib.Path(sys.argv[1]).resolve()
    user_topic = sys.argv[2]  # chủ đề do người dùng nhập
    ingest(file_path, user_topic)
