from .gemini_utils import gemini_generate   # dấu chấm = relative import
import re

def _ask_num_questions(text: str, max_q: int = 100) -> int:
    """Nhờ Gemini đề xuất số Q‑A cần sinh (1‒max_q)."""
    prompt = f"""
Bạn đóng vai trò biên soạn tài liệu Hỏi‑Đáp.  
Đọc nội dung dưới đây rồi cho biết:  
- Cần bao nhiêu cặp Hỏi‑Đáp để bao phủ đầy đủ nhưng không dư thừa?  
- Trả lời chỉ 1 số nguyên từ 1 đến {max_q}.  
---
{text[:8000]}
"""
    raw = gemini_generate(prompt)
    m = re.search(r"\d+", raw)
    if m:
        n = int(m.group())
        return max(1, min(n, max_q))
    # fallback nếu Gemini không trả số
    return 5

def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    pairs, q = [], ""
    for line in raw.splitlines():
        line = line.strip()
        if line.lower().startswith("q:"):
            q = line[2:].strip()
        elif line.lower().startswith("a:") and q:
            a = line[2:].strip()
            pairs.append((q, a))
            q = ""
    return pairs

def generate_qa_pairs(text: str, max_q: int = 100) -> list[tuple[str, str]]:
    """Tự hỏi Gemini số cặp, rồi sinh đúng số đó (≤ max_q)."""
    n_q = _ask_num_questions(text, max_q)
    prompt = f"""
Sinh {n_q} cặp Hỏi‑Đáp, đúng trọng tâm (định dạng Q:/A:), đầy đủ , tiếng việt .  
---
{text[:8000]}
"""
    raw = gemini_generate(prompt)
    pairs = _parse_pairs(raw)
    # Nếu Gemini trả thiếu, gọi bổ sung (tùy chọn)
    return pairs[:max_q]
