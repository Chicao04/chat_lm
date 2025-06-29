# 29/06 Kết nốt llm chả lời chính xác hơn

import os
import yaml
import json
import requests
from typing import Optional, List
from langchain.llms.base import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embedding = HuggingFaceEmbeddings(model_name=config['embedding_model'])

# Load vectorstore
db_path = config["vectorstore_path"]
if not os.path.exists(db_path):
    raise FileNotFoundError("❌ Vectorstore chưa được tạo. Vui lòng chạy chunk_and_ingest.py.")

db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# Lớp kết nối LM Studio
class LMStudioLLM(LLM):
    model: str = "llama2-7b-chat"
    base_url: str = "http://localhost:1234/v1/chat/completions"
    temperature: float = 0.7
    max_tokens: int = 512

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

# ✅ Khởi tạo mô hình
llm = LMStudioLLM()

# Tạo chain RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# CLI
while True:
    query = input("\n🔍 Nhập câu hỏi (hoặc 'exit' để thoát): ").strip()
    if query.lower() == "exit":
        print("👋 Thoát.")
        break

    result = qa_chain.invoke(query)

    print("\n🤖 Câu trả lời:")
    print(result["result"].strip())

    print("\n📂 Nguồn trích:")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Không rõ')}")
