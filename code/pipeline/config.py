import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_TOKENS = 4000

SYSTEM_PROMPT = (
    "Bạn là một chuyên gia trả lời câu hỏi. "
    "Hãy trả lời câu hỏi sau một cách chính xác và ngắn gọn. "
    "Trả lời bằng văn bản thuần túy, KHÔNG sử dụng định dạng markdown "
    "(không dùng **, #, -, ```, v.v.)."
)

JUDGE_PROMPT = """Bạn là một giám khảo chấm thi chuyên nghiệp. Nhiệm vụ của bạn là so sánh câu trả lời của một model AI với đáp án đúng và chấm điểm theo thang CSAT từ 1 đến 5 dựa trên các tiêu chí dưới đây.

TIÊU CHÍ CHẤM ĐIỂM:

5 điểm — Trả lời hoàn toàn chính xác, đầy đủ, có thể mở rộng đúng trọng tâm.
- Câu trả lời cung cấp đúng và đầy đủ toàn bộ đáp án.
- Có thể bổ sung thêm thông tin nhưng không làm lệch trọng tâm câu hỏi, không có thông tin sai hoặc thừa gây nhiễu.
- Ví dụ: Câu hỏi "Chiến dịch Điện Biên Phủ diễn ra vào năm nào?", đáp án đúng "1954". Model trả lời "Chiến dịch Điện Biên Phủ diễn ra năm 1954" → 5 điểm. Trả lời "Chiến dịch Điện Biên Phủ diễn ra năm 1954, kết thúc thắng lợi ngày 7/5/1954" → 5 điểm.

4 điểm — Trả lời chính xác, có phần mở rộng nhưng phải nằm trong phần trọng tâm của đáp án.
- Câu trả lời đúng đáp án chính. Có thể mở rộng thêm thông tin liên quan, nhưng phần mở rộng vẫn nằm trong phạm vi nội dung câu hỏi, không thêm chi tiết sai hoặc không liên quan.
- Ví dụ: Câu hỏi "Di tích Núi Đọ thuộc tỉnh nào sau đây?", đáp án đúng "Thanh Hóa". Model trả lời "Di tích Núi Đọ thuộc địa phận hai xã Thiệu Tân và Thiệu Khánh, huyện Đông Sơn, Thanh Hóa" → 4 điểm.

3 điểm — Trả lời chính xác một phần, chưa đầy đủ hết các đáp án.
- Câu trả lời có chứa thông tin đúng, nhưng thiếu một hoặc nhiều ý quan trọng so với đáp án chuẩn. Không có thông tin sai, chỉ thiếu.
- Ví dụ: Câu hỏi "Vật dụng nào thường được đặt trong quan tài theo quan niệm truyền thống?", đáp án đúng "Gạo, muối, tiền giấy". Model trả lời "Gạo" → 3 điểm. Model trả lời "Muối và tiền giấy" → 3 điểm.

2 điểm — Trả lời đúng một phần và sai một phần.
- Câu trả lời vừa chứa thông tin đúng vừa chứa thông tin sai. Phần sai làm ảnh hưởng đến tính chính xác của toàn bộ câu trả lời, dù vẫn có yếu tố đúng.
- Ví dụ: Câu hỏi "Người Hòa Bình trộn thêm chất gì vào đất khi làm đồ gốm để tránh nứt vỡ?", đáp án đúng "Cát". Model trả lời "Cát và xác thực vật" → 2 điểm.

1 điểm — Câu trả lời sai hoàn toàn.
- Câu trả lời không đúng với đáp án chuẩn, hoặc thông tin đưa ra hoàn toàn sai, không liên quan đến câu hỏi.
- Ví dụ: Câu hỏi "Tốc độ tăng trưởng GDP bình quân của Việt Nam giai đoạn 1991–1995?", đáp án đúng "7–8%/năm". Model trả lời "4–5%/năm" → 1 điểm.

ĐỊNH DẠNG OUTPUT (BẮT BUỘC):
Chỉ trả lời DUY NHẤT một JSON object hợp lệ, không kèm markdown code fence, không kèm bất kỳ text giải thích nào ngoài JSON. Cấu trúc:
{"score": <số nguyên từ 1 đến 5>, "reasoning": "<lý do ngắn gọn bằng tiếng Việt>"}"""

DEFAULT_MODEL = "gpt-4o"

ALLOWED_MODELS = [
    "claude-haiku-4.5",
    "claude-opus-4.5",
    "claude-opus-4.6",
    "claude-opus-4.7",
    "claude-sonnet-4.5",
    "claude-sonnet-4.6",
    "deepseek-r1",
    "gemini-2.5-flash",
    "gemini-2.5-flash-image",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-pro-image-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "gemma-4-31B-it",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-chat",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5-pro",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.2-chat",
    "gpt-5.2-codex",
    "gpt-5.3-chat",
    "gpt-5.3-codex",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.4-pro",
    "gpt-5.5",
    "gpt-image-1",
    "gpt-image-1-mini",
    "gpt-image-1.5",
    "gpt-image-2",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3-1-8b",
    "llama-3-2-90b-vision",
    "llama-3-3-70b",
    "o4-mini",
    "qwen-3.5-397b-a17b-FP8",
    "qwen3-next-80b-a3b-instruct",
    "text-embedding-3-large",
    "text-embedding-3-small",
]

NUM_WORKERS = 48

MAX_RETRIES = 2
RETRY_BASE_DELAY = 2

SAVE_EVERY = 20

DATA_PATH = os.path.join(_THIS_DIR, "run_models/data/kept_items_full.json")
OUTPUT_DIR = os.path.join(_THIS_DIR, "output_test")
