import os
import requests
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain.llms.base import LLM
from typing import Optional, List
from functools import partial
from langchain_huggingface import HuggingFaceEmbeddings


### 🧩 Bước 1: Tạo wrapper kết nối LM Studio (OpenAI API format)

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

### ✅ Bước 2: Tải và xử lý dữ liệu từ thư mục chứa file .txt

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


splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = splitter.split_documents(all_documents)

### ✅ Bước 3: Dùng HuggingFace Embedding + FAISS : Chuyển tệp dữ liệu đầu vào thành các vector

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(texts, embedding_model)
# vectorstore.save_local("faiss_index")


#  cải tiến 25/06


index_path = "C:/Users/FPTSHOP/PycharmProjects/Chat_lm/RAG/faiss_index"

# Tạo thư mục nếu chưa có
os.makedirs(index_path, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Kiểm tra xem index đã tồn tại hay chưa bằng cách kiểm tra file FAISS
if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
    print("📦 Đang tải FAISS vectorstore từ ổ đĩa...")
    vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
else:
    print("📦 Đang tạo FAISS vectorstore và lưu vào ổ đĩa...")
    vectorstore = FAISS.from_documents(texts, embedding_model)
    vectorstore.save_local(index_path)


### ✅ Bước 4: Xây dựng conversational RAG chain

llm = LMStudioLLM()
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
chat_history = []

### ✅ Bước 5: Giao diện chat đơn giản

print("💬 Chatbot RAG đã sẵn sàng. Gõ 'exit' để thoát.\n")
while True:
    question = input("👤 Bạn: ")
    if question.lower() in ['exit', 'thoát']:
        print("👋 Tạm biệt!")
        break

    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    print("🤖 Bot:", result["answer"])
    chat_history.append((question, result["answer"]))