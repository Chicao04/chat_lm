from .gemini_utils import gemini_generate

LABELS = {
    "Kinh tế":  "econ_qa",
    "Chính trị": "pol_qa",
    "Y tế":      "health_qa",
    "Dịch vụ":   "service_qa",
}

def process_file_with_topic(text: str, user_topic: str) -> tuple[str, str]:
    """Xử lý văn bản theo chủ đề do người dùng chọn trước"""
    user_topic = user_topic.strip()

    if user_topic not in LABELS:
        raise ValueError(f"❌ Chủ đề '{user_topic}' không hợp lệ. Vui lòng chọn một trong: {list(LABELS.keys())}")

    db_name = LABELS[user_topic]

    # Prompt tùy theo chủ đề
    prompt = f"""
Bạn là trợ lý chuyên về lĩnh vực {user_topic}.
Dưới đây là nội dung cần phân tích:
{text[:3000]}

Hãy đưa ra câu trả lời phù hợp theo lĩnh vực {user_topic}.
"""
    response = gemini_generate(prompt).strip()
    return response, db_name
