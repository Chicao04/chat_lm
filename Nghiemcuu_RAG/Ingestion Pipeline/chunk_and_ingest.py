import yaml, pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load tài liệu đã clean
with open("scripts/cleaned.pkl", "rb") as f:
    cleaned_docs = pickle.load(f)

# Splitter config
splitter = RecursiveCharacterTextSplitter(
    chunk_size=config['chunk_size'],
    chunk_overlap=config['chunk_overlap']
)

# Tách từng file riêng để tránh trộn
chunks = []
for doc in cleaned_docs:
    doc_chunks = splitter.split_documents([doc])
    for chunk in doc_chunks:
        chunk.metadata["source"] = doc.metadata.get("source", "unknown")
    chunks.extend(doc_chunks)

# Embedding và FAISS
embedding = HuggingFaceEmbeddings(model_name=config['embedding_model'])
db = FAISS.from_documents(chunks, embedding)
db.save_local(config['vectorstore_path'])

print(f"✅ Chunk & FAISS hoàn tất: {len(chunks)} đoạn được lưu.")
