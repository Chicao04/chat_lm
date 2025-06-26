# thay vì lưu ở dạng chữ ta lưu ở dưới đoạn
import os
import requests
import json
from functools import partial
from typing import Optional, List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader

from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.llms.base import LLM
from functools import partial

# === Bước 1: Tạo LLM wrapper kết nối LM Studio ===

class LMStudioLLM(LLM):
    model: str = "llama2-7b-chat"
    base_url: str = "http://localhost:1234/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 100

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Lỗi gọi LM Studio: {response.status_code} - {response.text}")

    @property
    def _llm_type(self) -> str:
        return "lm_studio"

# === Bước 2: Load dữ liệu văn bản từ thư mục ===

print("📂 Đang tải dữ liệu từ thư mục ./data ...")

all_documents = []

# Load .txt
txt_loader = DirectoryLoader(
    './data',
    glob="**/*.txt",
    loader_cls=partial(TextLoader, encoding='utf-8')
)
all_documents.extend(txt_loader.load())

# Load .pdf
pdf_loader = DirectoryLoader(
    './data',
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader  # dùng pymupdf
)
all_documents.extend(pdf_loader.load())

# Load .docx


docx_loader = DirectoryLoader(
    './data',
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
)

all_documents.extend(docx_loader.load())

print(f"📄 Tổng số file văn bản đã load: {len(all_documents)}")

# === Bước 2.5: Chia văn bản bằng TokenTextSplitter ===

from langchain.text_splitter import TokenTextSplitter

# splitter = TokenTextSplitter(
#     chunk_size=256,       # Số token, tương đương khoảng 800-1000 ký tự
#     chunk_overlap=20      # Giữ ngữ cảnh giữa các đoạn
# )
#
# texts = splitter.split_documents(all_documents)

# cải tiến 26/06


splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)

texts = []

for doc in all_documents:
    chunks = splitter.split_text(doc.page_content)
    for chunk in chunks:
        texts.append(Document(page_content=chunk, metadata=doc.metadata))


###

print(f"📄 Số tài liệu gốc: {len(all_documents)}")
print(f"🔹 Sau khi chia nhỏ: {len(texts)} đoạn")

# === Bước 3: Tạo và lưu FAISS index ===

index_path = "C:/Users/FPTSHOP/PycharmProjects/Chat_lm/RAG/faiss_index2"
os.makedirs(index_path, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
    print("📦 Đang tải FAISS vectorstore từ ổ đĩa...")
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("📦 Đang tạo FAISS vectorstore và lưu vào ổ đĩa...")
    vectorstore = FAISS.from_documents(texts, embedding_model)
    vectorstore.save_local(index_path)

# === Bước 4: Tạo chuỗi hội thoại RAG ===

llm = LMStudioLLM()
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
chat_history = []

# === Bước 5: Giao diện CLI đơn giản ===

print("💬 Chatbot RAG đã sẵn sàng. Gõ 'exit' để thoát.\n")
while True:
    question = input("👤 Bạn: ")
    if question.lower() in ['exit', 'thoát']:
        print("👋 Tạm biệt!")
        break

    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    print("🤖 Bot:", result["answer"])
    chat_history.append((question, result["answer"]))
