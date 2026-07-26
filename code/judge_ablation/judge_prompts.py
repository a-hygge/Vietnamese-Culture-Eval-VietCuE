"""
4 prompting strategies cho LLM-as-judge — chấm 0-5 ordinal scale.

Mỗi prompt nhận: question, gold (đáp án chuẩn), output (câu trả lời model cần chấm)
→ return integer score 0-5.

Theo template VIVID (LREC 2026):
  1. Zero-shot     — chỉ rubric + ask score
  2. CoT           — yêu cầu reasoning trước khi score
  3. Demonstration — kèm 4 ví dụ minh họa rubric
  4. Aspect-based  — chấm từng khía cạnh (content/keywords/accuracy) rồi tổng
"""

SYSTEM_PROMPT = (
    "Bạn là giám khảo chấm điểm câu trả lời tiếng Việt về văn hóa, lịch sử, "
    "pháp luật Việt Nam. Hãy chấm THẬT KHÁCH QUAN dựa trên rubric, "
    "KHÔNG bị ảnh hưởng bởi style hay độ dài câu trả lời. Focus vào NỘI DUNG."
)

RUBRIC = """RUBRIC CHẤM ĐIỂM (0-5):
- 5: Khớp hoàn toàn nội dung và keywords chính (kể cả phrase khác)
- 4: Khớp ý chính, thiếu 1-2 chi tiết KHÔNG quan trọng
- 3: Đúng MỘT PHẦN, có liên quan nhưng thiếu hoặc lệch
- 2: Off-direction (lệch hướng), có 1 yếu tố đúng
- 1: Sai hoàn toàn / hallucinate (bịa thông tin)
- 0: Refuse / không trả lời / để trống

NGUYÊN TẮC:
- KHÔNG bị ảnh hưởng bởi style. Focus NỘI DUNG.
- Cho 5 nếu model có ĐỦ keywords/concepts của gold (kể cả rephrase hoàn toàn).
- Cho 4 nếu match concept chính, chỉ thiếu chi tiết phụ.
- Cho 3 nếu cùng chủ đề nhưng aspect khác.
- Cho 2 cho off-direction.
- Cho 1 cho hallucinate / sai hoàn toàn."""

DEMO_EXAMPLES = """VÍ DỤ:

VÍ DỤ 1 (score 5 — match nội dung):
  Câu hỏi: "Hai Bà Trưng khởi nghĩa năm nào?"
  Gold: "Năm 40 SCN, từ Hát Môn"
  Model: "Năm 40, từ cửa sông Hát"
  → Score: 5

VÍ DỤ 2 (score 5 — model đầy đủ hơn):
  Câu hỏi: "Vai trò người phụ nữ trong gia đình VN?"
  Gold: "vai trò người phụ nữ"
  Model: "vai trò phụ nữ, lời nói/quyết định ảnh hưởng lớn"
  → Score: 5

VÍ DỤ 3 (score 2 — off-direction):
  Câu hỏi: "Quan niệm về các dân tộc theo người Việt xưa?"
  Gold: "Mọi dân tộc ưa lành ghét dữ"
  Model: "Các dân tộc đều bình đẳng"
  → Score: 2

VÍ DỤ 4 (score 1 — hallucinate):
  Câu hỏi: "Ai chịu trách nhiệm trong kỳ thi Đình?"
  Gold: "Quan Đề điệu chịu trách nhiệm"
  Model: "Chánh chủ khảo"
  → Score: 1
"""


# ============================================================
# 4 STRATEGIES
# ============================================================

def prompt_zero_shot(question: str, gold: str, output: str) -> str:
    return f"""{RUBRIC}

Hãy chấm câu trả lời sau:

Câu hỏi: {question}

Đáp án chuẩn (gold): {gold}

Câu trả lời cần chấm (model): {output}

Chỉ trả về MỘT số nguyên 0-5 (không kèm giải thích). Format: chỉ con số.
"""


def prompt_cot(question: str, gold: str, output: str) -> str:
    return f"""{RUBRIC}

Hãy chấm câu trả lời sau bằng phương pháp Chain-of-Thought (suy luận từng bước):

Câu hỏi: {question}

Đáp án chuẩn (gold): {gold}

Câu trả lời cần chấm (model): {output}

Hãy suy luận theo các bước:
1. Xác định keywords/concepts chính trong gold.
2. Kiểm tra model có cover các keywords/concepts đó không.
3. Đánh giá có sai sót hay hallucinate không.
4. Quyết định điểm 0-5 dựa trên rubric.

Cuối cùng, KẾT LUẬN bằng 1 dòng đúng format:
SCORE: <số 0-5>
"""


def prompt_demonstration(question: str, gold: str, output: str) -> str:
    return f"""{RUBRIC}

{DEMO_EXAMPLES}

Áp dụng tương tự, hãy chấm câu trả lời sau:

Câu hỏi: {question}

Đáp án chuẩn (gold): {gold}

Câu trả lời cần chấm (model): {output}

Chỉ trả về MỘT số nguyên 0-5 (không kèm giải thích). Format: chỉ con số.
"""


def prompt_aspect_based(question: str, gold: str, output: str) -> str:
    return f"""{RUBRIC}

Hãy chấm câu trả lời sau theo phương pháp ASPECT-BASED — đánh giá từng khía cạnh
trước khi tổng hợp điểm cuối:

Câu hỏi: {question}

Đáp án chuẩn (gold): {gold}

Câu trả lời cần chấm (model): {output}

Đánh giá từng khía cạnh sau (mỗi khía cạnh 0-5):

A. Content match (mức độ khớp nội dung chính của gold): __/5
B. Keyword coverage (model có nhắc keywords/concepts của gold không): __/5
C. Accuracy (có sai sự thật, hallucinate, hay lệch hướng không): __/5

Tổng hợp 3 khía cạnh → điểm cuối 0-5.

KẾT LUẬN format:
A: <0-5>
B: <0-5>
C: <0-5>
SCORE: <0-5>
"""


PROMPT_FN = {
    'zero_shot':     prompt_zero_shot,
    'cot':           prompt_cot,
    'demonstration': prompt_demonstration,
    'aspect_based':  prompt_aspect_based,
}


def parse_score(text: str, strategy: str) -> int | None:
    """Parse score 0-5 từ response. Return None nếu parse fail."""
    import re
    text = text.strip()
    if strategy in ('cot', 'aspect_based'):
        # Tìm "SCORE: N"
        m = re.search(r'SCORE\s*:\s*([0-5])\b', text, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # Fallback: tìm số 0-5 ở cuối
        m = re.search(r'\b([0-5])\b\s*\.?\s*$', text)
        if m:
            return int(m.group(1))
    # Zero-shot / Demo: response chỉ có số
    m = re.search(r'\b([0-5])\b', text)
    return int(m.group(1)) if m else None
