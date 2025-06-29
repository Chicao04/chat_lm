# 29/06 bổ sung thêm promt
# Kết nốt llm chả lời chính xác hơn

import os
import yaml
import json
import requests
from typing import Optional, List
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === 1. Load cấu hình ===
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

embedding = HuggingFaceEmbeddings(model_name=config['embedding_model'])

db_path = config["vectorstore_path"]
if not os.path.exists(db_path):
    raise FileNotFoundError("❌ Vectorstore chưa được tạo. Vui lòng chạy chunk_and_ingest.py.")

# === 2. Load FAISS retriever ===
db = FAISS.load_local(db_path, embedding, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# === 3. Lớp kết nối LM Studio LLM ===
class LMStudioLLM(LLM):
    model: str = "llama2-7b-chat"  # hoặc thay bằng tên model bạn đang dùng
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

# === 4. Khởi tạo LLM ===
llm = LMStudioLLM()

# === 5. PromptTemplate trả lời tự nhiên ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Bạn là một trợ lý AI thân thiện, chuyên hỗ trợ người dùng trả lời các câu hỏi dựa trên tài liệu sau.

Hãy đọc kỹ phần tài liệu và trả lời bằng **tiếng Việt rõ ràng, tự nhiên, đúng trọng tâm**.

### Tài liệu:
{context}

### Câu hỏi:
{question}

### Trả lời:
"""
)

# === 6. Tạo RetrievalQA chain dùng prompt tùy chỉnh ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# === 7. Vòng lặp CLI hỏi đáp ===
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
