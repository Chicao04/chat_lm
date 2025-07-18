import os
import yaml
import json
import re
import unicodedata
import requests
from typing import Optional, List
from langchain.llms.base import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# === Hàm chuẩn hóa câu hỏi ===
def clean_query(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text

# === Load config.yaml ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embedding = HuggingFaceEmbeddings(model_name=config["embedding_model"])
db_path = config["vectorstore_path"]
if not os.path.exists(db_path):
    raise FileNotFoundError("❌ Vectorstore chưa được tạo. Vui lòng chạy chunk_and_ingest.py.")

db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# === Lớp kết nối LM Studio ===
class LMStudioLLM(LLM):
    model: str = "llama2-7b-chat"
    base_url: str = "http://localhost:1234/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 700

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

# === Khởi tạo mô hình ===
llm = LMStudioLLM()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# === CLI Loop: hỏi → refine trả lời ===
while True:
    query = input("\n🔍 Nhập câu hỏi (hoặc 'exit' để thoát): ").strip()
    if query.lower() == "exit":
        print("👋 Thoát.")
        break

    cleaned_query = clean_query(query)

    # --- Vòng 1: RAG trả lời từ context ---
    result1 = qa_chain.invoke(cleaned_query)
    first_answer = result1["result"].strip()

    # --- Lấy context gốc từ vectorstore ---
    retrieved_context = "\n\n".join([doc.page_content for doc in result1["source_documents"]])

    # --- Vòng 2: LLM refine lại câu trả lời ---
    refine_prompt = f"""
Bạn là một trợ lý AI. Dưới đây là một số tài liệu liên quan và câu hỏi của người dùng. Hãy đọc kỹ tài liệu và đưa ra câu trả lời chính xác, rõ ràng và đầy đủ nhất **bằng tiếng Việt tự nhiên**.

### Tài liệu:
{retrieved_context}

### Câu hỏi:
{query}

### Câu trả lời:
"""

    final_answer = llm._call(refine_prompt)

    # --- In kết quả ---
    print("\n🤖 Câu trả lời cuối cùng:")
    print(final_answer.strip())

    print("\n📂 Nguồn trích:")
    for doc in result1["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Không rõ')}")
