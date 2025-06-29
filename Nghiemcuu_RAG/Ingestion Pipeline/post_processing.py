def clean_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if "Page" not in line and line.strip()]
    return "\n".join(lines)

def clean_documents(docs):
    cleaned = []
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        if len(doc.page_content.strip()) > 100:
            cleaned.append(doc)
    return cleaned

import pickle
with open("scripts/extracted.pkl", "rb") as f:
    docs = pickle.load(f)

cleaned_docs = clean_documents(docs)
with open("scripts/cleaned.pkl", "wb") as f:
    pickle.dump(cleaned_docs, f)