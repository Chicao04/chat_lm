# Bước	                Nội dung	                                                                     Mục đích
# data_extraction.py	Đọc file .txt, .pdf → lưu extracted.pkl	                                         Lấy dữ liệu thô
# post_processing.py	Làm sạch dữ liệu → lưu cleaned.pkl	                                             Loại bỏ phần thừa
# chunk_and_ingest.py	Cắt đoạn, tạo vector bằng HuggingFaceEmbeddings, lưu vào vector store (FAISS)	Tạo kho tri thức có thể tìm kiếm
# ✅ Sau bước này	Bạn có thể search bằng câu hỏi → tìm được đoạn phù hợp nhất


import os
os.system("python data_extraction.py")
os.system("python post_processing.py")
os.system("python chunk_and_ingest.py")
print("✅ Ingestion Pipeline Completed!")
