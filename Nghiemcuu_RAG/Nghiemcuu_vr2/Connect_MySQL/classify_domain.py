from .gemini_utils import gemini_generate

LABELS = {
    "Kinh tế":  "econ_qa",
    "Chính trị": "pol_qa",
    "Y tế":      "health_qa",
    "Dịch vụ":   "service_qa",
}

def classify_domain(text: str) -> str:
    """Phân loại nội dung thành một miền: Kinh tế, Chính trị, Y tế, Dịch vụ."""

    # Tạo phần danh sách lựa chọn
    label_names = list(LABELS.keys())
    label_options = "\n".join([f"- {label}" for label in label_names])

    # Prompt gồm hướng dẫn và ví dụ rõ ràng
    prompt = f"""
Bạn là một hệ thống phân loại tài liệu tự động.

Dưới đây là một số ví dụ:
---
Văn bản: "Ngân hàng trung ương điều chỉnh lãi suất..."  
→ Kết quả: Kinh tế

Văn bản: "Chính phủ tổ chức cuộc họp về hiến pháp..."  
→ Kết quả: Chính trị

Văn bản: "Bộ Y tế công bố dịch cúm mới tại miền Bắc..."  
→ Kết quả: Y tế

Văn bản: "Dịch vụ giao hàng GrabExpress mở rộng thêm 10 tỉnh..."  
→ Kết quả: Dịch vụ
---

Giờ hãy phân loại đoạn sau và TRẢ LỜI CHÍNH XÁC MỘT TRONG CÁC TỪ SAU (KHÔNG THÊM GÌ KHÁC):
{label_options}

Nội dung:
{text[:3000]}

Trả lời 1 từ duy nhất.
"""

    # Gọi Gemini sinh kết quả
    raw = gemini_generate(prompt).strip().lower()

    # Xử lý đầu ra - kiểm tra khớp chính xác
    for vn, db in LABELS.items():
        if vn.lower() == raw:
            return db

    # Nếu không khớp chính xác, kiểm tra khớp gần (fuzzy match fallback)
    import difflib
    closest = difflib.get_close_matches(raw, [k.lower() for k in LABELS.keys()], n=1, cutoff=0.6)
    if closest:
        for vn, db in LABELS.items():
            if vn.lower() == closest[0]:
                return db

    print(f"[⚠️] Gemini trả về không khớp: {raw}")
    return "service_qa"  # fallback mặc định
