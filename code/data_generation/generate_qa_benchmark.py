"""
Vietnamese Benchmark Q&A Generation Script
Generates Q&A pairs from Vietnamese culture and law text sources.

Output format:
{
    "id": "VH_001" or "PL_001",
    "source": "Bản sắc văn hóa Việt Nam" or "Pháp luật đại cương",
    "category": "culture" or "law",
    "question": "...",
    "answer": "...",
    
}
"""

import os
import json
import openai
from typing import List, Dict, Literal

client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-02-15-preview"
)

SOURCES = {
    "culture": {
        "name": "Bản sắc văn hóa Việt Nam",
        "id_prefix": "VH",
        "input_dir": r"../data_sources/ban_sac_van_hoa_viet_nam/structured_chunks_v2",
        "output_dir": r"../data_question_answer/ban_sac_van_hoa_viet_nam"
    },
    "law": {
        "name": "Pháp luật đại cương",
        "id_prefix": "PL",
        "input_dir": r"../data_sources/bai_giang_phap_luat_dai_cuong/structured_chunks_v2",
        "output_dir": r"../data_question_answer/bai_giang_phap_luat_dai_cuong"
    }
}

SYSTEM_PROMPT_CULTURE = """
Bạn là chuyên gia tạo câu hỏi đánh giá hiểu biết về văn hóa Việt Nam.

ĐỌC đoạn văn dưới đây và TẠO 5 câu hỏi-đáp án chất lượng cao bằng TIẾNG VIỆT.

## QUY TẮC BẮT BUỘC:

### 1. LOẠI CÂU HỎI PHÙ HỢP:
- Hỏi về khái niệm, định nghĩa
- Hỏi về ý nghĩa, vai trò
- Hỏi về đặc điểm, tính chất
- Hỏi về nguyên nhân, mục đích
- Hỏi về mối quan hệ

### 2. TUYỆT ĐỐI KHÔNG:
- KHÔNG hỏi "đoạn văn này nói gì về..."
- KHÔNG suy luận ngoài văn bản
- KHÔNG hỏi quá mở hoặc quá dài
- KHÔNG dùng các cụm từ: "trong đoạn văn", "theo văn bản", "theo tài liệu", "tác giả đề cập", "chương này", "phần này"

### 3. CÁCH VIẾT CÂU HỎI:
- Câu hỏi phải TỰ HIỂU ĐƯỢC (không cần đọc đoạn văn gốc)
- Dùng tên riêng, khái niệm cụ thể
- Ngắn gọn, rõ ràng, đi thẳng vào vấn đề

### 4. CÁCH VIẾT CÂU TRẢ LỜI:
- DỰA 100% vào nội dung đoạn văn
- KHÔNG suy luận ngoài văn bản
- Giữ nguyên ý chính, diễn đạt phù hợp
- Trả lời trực tiếp câu hỏi

### 5. CONTEXT (BẮT BUỘC):
Mỗi câu hỏi PHẢI có context ngắn gọn cho biết chủ đề/lĩnh vực của câu hỏi.
Context giúp người trả lời biết đang hỏi về lĩnh vực gì.

Các loại context phù hợp cho VĂN HÓA VIỆT NAM:
- "Sau đây là câu hỏi về phong tục tập quán Việt Nam."
- "Sau đây là câu hỏi về tín ngưỡng và tôn giáo Việt Nam."
- "Sau đây là câu hỏi về nghệ thuật truyền thống Việt Nam."
- "Sau đây là câu hỏi về ẩm thực Việt Nam."
- "Sau đây là câu hỏi về lễ hội truyền thống Việt Nam."
- "Sau đây là câu hỏi về văn hóa gia đình Việt Nam."
- "Sau đây là câu hỏi về văn hóa vùng miền Việt Nam."
- "Sau đây là câu hỏi về bản sắc văn hóa dân tộc Việt Nam."
- "Sau đây là câu hỏi về lịch sử văn hóa Việt Nam."
- Hoặc context cụ thể phù hợp với nội dung câu hỏi.

## VÍ DỤ:

{{
  "context": "Sau đây là câu hỏi về bản sắc văn hóa dân tộc Việt Nam.",
  "question": "Ẩn dụ văn hóa trong văn hóa Việt Nam là gì?",
  "answer": "Ẩn dụ văn hóa là cách người Việt dùng những hình ảnh quen thuộc trong đời sống để thể hiện ý nghĩa sâu xa hơn của tư tưởng, tình cảm hoặc quan niệm về cuộc sống."
}}

{{
  "context": "Sau đây là câu hỏi về tín ngưỡng thờ cúng tổ tiên của người Việt.",
  "question": "Ý nghĩa của việc thờ cúng tổ tiên trong văn hóa Việt Nam là gì?",
  "answer": "Thờ cúng tổ tiên thể hiện lòng hiếu thảo, biết ơn và sự kết nối giữa các thế hệ trong gia đình Việt Nam."
}}

## ĐỊNH DẠNG JSON YÊU CẦU:
[
  {{
    "context": "Sau đây là câu hỏi về [chủ đề cụ thể].",
    "question": "Câu hỏi ngắn gọn, rõ ràng",
    "answer": "Câu trả lời dựa 100% vào đoạn văn"
  }},
  ...
]

Tạo ĐÚNG 5 câu hỏi có chất lượng cao.
"""

SYSTEM_PROMPT_LAW = """
Bạn là chuyên gia tạo câu hỏi đánh giá hiểu biết về pháp luật Việt Nam.

ĐỌC đoạn văn dưới đây và TẠO 5 câu hỏi-đáp án chất lượng cao bằng TIẾNG VIỆT.

## QUY TẮC BẮT BUỘC:

### 1. LOẠI CÂU HỎI PHÙ HỢP:
- Hỏi về khái niệm, định nghĩa pháp lý
- Hỏi về vai trò, chức năng của pháp luật
- Hỏi về quyền và nghĩa vụ công dân
- Hỏi về nguyên tắc pháp luật
- Hỏi về cơ cấu tổ chức nhà nước

### 2. TUYỆT ĐỐI KHÔNG:
- KHÔNG hỏi "đoạn văn này nói gì về..."
- KHÔNG suy luận ngoài văn bản
- KHÔNG hỏi quá mở hoặc quá dài
- KHÔNG dùng các cụm từ: "trong đoạn văn", "theo văn bản", "theo tài liệu", "tác giả đề cập", "chương này", "phần này"

### 3. CÁCH VIẾT CÂU HỎI:
- Câu hỏi phải TỰ HIỂU ĐƯỢC (không cần đọc đoạn văn gốc)
- Dùng thuật ngữ pháp lý chính xác
- Ngắn gọn, rõ ràng, đi thẳng vào vấn đề

### 4. CÁCH VIẾT CÂU TRẢ LỜI:
- DỰA 100% vào nội dung đoạn văn
- KHÔNG suy luận ngoài văn bản
- Giữ nguyên ý chính, diễn đạt phù hợp
- Trả lời trực tiếp câu hỏi

### 5. CONTEXT (BẮT BUỘC):
Mỗi câu hỏi PHẢI có context ngắn gọn cho biết chủ đề/lĩnh vực của câu hỏi.
Context giúp người trả lời biết đang hỏi về lĩnh vực gì.

Các loại context phù hợp cho PHÁP LUẬT VIỆT NAM:
- "Sau đây là câu hỏi về khái niệm pháp luật."
- "Sau đây là câu hỏi về Hiến pháp Việt Nam."
- "Sau đây là câu hỏi về quyền công dân."
- "Sau đây là câu hỏi về nghĩa vụ công dân."
- "Sau đây là câu hỏi về tổ chức bộ máy nhà nước Việt Nam."
- "Sau đây là câu hỏi về luật dân sự."
- "Sau đây là câu hỏi về luật hình sự."
- "Sau đây là câu hỏi về luật hành chính."
- "Sau đây là câu hỏi về nguyên tắc pháp quyền."
- Hoặc context cụ thể phù hợp với nội dung câu hỏi.

## VÍ DỤ:

{{
  "context": "Sau đây là câu hỏi về nguyên tắc tổ chức nhà nước pháp quyền Việt Nam.",
  "question": "Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam được tổ chức theo nguyên tắc pháp quyền nào?",
  "answer": "Nhà nước hoạt động trên nền tảng Hiến pháp và pháp luật, đảm bảo quyền lực nhà nước thống nhất và bảo vệ quyền con người, quyền công dân."
}}

{{
  "context": "Sau đây là câu hỏi về vai trò của pháp luật trong quản lý xã hội.",
  "question": "Vai trò của pháp luật trong quản lý xã hội là gì?",
  "answer": "Pháp luật đóng vai trò điều chỉnh hành vi, duy trì trật tự và đảm bảo công bằng trong xã hội."
}}

## ĐỊNH DẠNG JSON YÊU CẦU:
[
  {{
    "context": "Sau đây là câu hỏi về [chủ đề cụ thể].",
    "question": "Câu hỏi ngắn gọn, rõ ràng",
    "answer": "Câu trả lời dựa 100% vào đoạn văn"
  }},
  ...
]

Tạo ĐÚNG 5 câu hỏi có chất lượng cao.
"""

VERIFICATION_PROMPT = """
Kiểm tra và sửa lại các câu hỏi theo CHECKLIST:

## Bước 1 - Kiểm tra từ ngữ cấm:
Tìm và XÓA các cụm từ sau:
- "trong đoạn văn", "theo đoạn văn", "văn bản", "tài liệu"
- "theo tác giả", "đoạn trích", "chương này", "phần này", "bài này"

## Bước 2 - Kiểm tra chất lượng câu hỏi:
- Câu hỏi có tự hiểu được không? (không cần đọc đoạn văn gốc)
- Câu hỏi có đi vào khái niệm/ý nghĩa/đặc điểm cụ thể không?
- Câu hỏi có quá mở hoặc quá dài không?

## Bước 3 - Kiểm tra câu trả lời:
- Câu trả lời có dựa 100% vào nội dung đoạn văn không?
- Câu trả lời có suy luận ngoài văn bản không?
- Câu trả lời có trả lời trực tiếp câu hỏi không?

## Bước 4 - Loại bỏ thông tin thay đổi theo thời gian:
- GDP năm X, dân số năm X, v.v.
- Các số liệu cụ thể có thể lỗi thời

## Bước 5 - Kiểm tra context:
- Mỗi câu hỏi PHẢI có context ngắn gọn
- Context phải phù hợp với nội dung câu hỏi
- Context bắt đầu bằng "Sau đây là câu hỏi về..."

VIẾT LẠI các câu hỏi vi phạm. Trả về mảng JSON đã sửa hoàn chỉnh với ĐÚNG 5 câu hỏi.
Mỗi object PHẢI có 3 trường: context, question, answer.
"""

FINAL_PROMPT = """
Trả về danh sách cuối cùng của 5 cặp câu hỏi-đáp án ở định dạng JSON hợp lệ.

YÊU CẦU:
1. Phải là mảng JSON hợp lệ
2. Mỗi object có ĐÚNG 3 trường: context, question, answer
3. Tất cả bằng TIẾNG VIỆT
4. KHÔNG có từ "văn bản", "đoạn văn", "theo tác giả" trong câu hỏi
5. Câu trả lời dựa 100% vào đoạn văn gốc
6. Context phải ngắn gọn, bắt đầu bằng "Sau đây là câu hỏi về..."

Chỉ trả về JSON, không giải thích thêm.
"""

def generate_questions(passage: str, source_type: Literal["culture", "law"]) -> List[Dict]:
    """
    Generate Q&A pairs from a text passage.

    Args:
        passage: Text passage to generate questions from
        source_type: Either "culture" or "law"

    Returns:
        List of Q&A dictionaries with context, question, answer
    """
                                                    
    system_prompt = SYSTEM_PROMPT_CULTURE if source_type == "culture" else SYSTEM_PROMPT_LAW

    try:
                                            
        response_1 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Đoạn văn:\n\n{passage}\n\nHãy tạo 5 câu hỏi chất lượng cao theo yêu cầu."}
            ],
            temperature=0.7,
            max_tokens=3000
        )

        initial_questions = response_1.choices[0].message.content
        print(f"  → Câu hỏi ban đầu được tạo")

        response_2 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": VERIFICATION_PROMPT},
                {"role": "user", "content": f"Kiểm tra và sửa lại:\n\n{initial_questions}"}
            ],
            temperature=0.3,
            max_tokens=3000
        )

        verified_questions = response_2.choices[0].message.content
        print(f"  → Đã verify và cải thiện")

        response_3 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": FINAL_PROMPT},
                {"role": "user", "content": verified_questions}
            ],
            temperature=0.1,
            max_tokens=3000
        )

        final_output = response_3.choices[0].message.content
        print(f"  → Đã làm sạch cuối cùng")

        if "```json" in final_output:
            final_output = final_output.split("```json")[1].split("```")[0].strip()
        elif "```" in final_output:
            final_output = final_output.split("```")[1].split("```")[0].strip()

        questions = json.loads(final_output)

        if not isinstance(questions, list):
            raise ValueError("Output không phải là list")

        for q in questions:
            if not all(key in q for key in ['context', 'question', 'answer']):
                raise ValueError(f"Thiếu trường trong câu hỏi: {q}")

        print(f"  ✓ Validated {len(questions)} câu hỏi")
        return questions

    except json.JSONDecodeError as e:
        print(f"  ✗ Lỗi JSON: {str(e)}")
        return []
    except Exception as e:
        print(f"  ✗ Lỗi: {str(e)}")
        return []

def format_qa_output(questions: List[Dict], source_type: Literal["culture", "law"], start_id: int, chunk_file: str = "") -> List[Dict]:
    """
    Format Q&A pairs with proper ID, source, category, and context.

    Args:
        questions: List of Q&A dictionaries (with context, question, answer)
        source_type: Either "culture" or "law"
        start_id: Starting ID number
        chunk_file: Source chunk file name (e.g., "chunk_001.txt")

    Returns:
        List of formatted Q&A dictionaries
    """
    source_config = SOURCES[source_type]
    formatted = []

    if source_type == "culture":
        source_path = f"data_sources\\ban_sac_van_hoa_viet_nam\\structured_chunks_v2\\{chunk_file}"
    else:
        source_path = f"data_sources\\bai_giang_phap_luat_dai_cuong\\structured_chunks_v2\\{chunk_file}"

    for i, q in enumerate(questions):
        formatted.append({
            "id": f"{source_config['id_prefix']}_{start_id + i:03d}",
            "source": source_path,                              
            "category": source_type,
            "context": q.get("context", ""),                  
            "question": q["question"],
            "answer": q["answer"]
        })

    return formatted

def load_progress(progress_file: str) -> dict:
    """Load progress from file to resume interrupted generation."""
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {"processed_files": [], "current_id": 1}

def save_progress(progress_file: str, processed_files: list, current_id: int):
    """Save progress to file."""
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump({
            "processed_files": processed_files,
            "current_id": current_id
        }, f, ensure_ascii=False, indent=2)

def save_chunk_questions(output_dir: str, chunk_name: str, questions: list):
    """Save questions for a single chunk immediately."""
    chunk_dir = os.path.join(output_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_file = os.path.join(chunk_dir, f"{chunk_name}.json")
    with open(chunk_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

def merge_all_chunks(output_dir: str, source_type: str) -> list:
    """Merge all chunk JSON files into one benchmark file."""
    chunk_dir = os.path.join(output_dir, "chunks")
    if not os.path.exists(chunk_dir):
        return []

    all_questions = []
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.json')])

    for chunk_file in chunk_files:
        file_path = os.path.join(chunk_dir, chunk_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
                all_questions.extend(questions)
        except Exception as e:
            print(f"  Warning: Could not load {chunk_file}: {e}")

    return all_questions

def process_source(source_type: Literal["culture", "law"]):
    """
    Process all text chunks from a source and generate Q&A pairs.
    SAVES IMMEDIATELY after each chunk to prevent data loss.

    Args:
        source_type: Either "culture" or "law"
    """
    source_config = SOURCES[source_type]
    input_dir = source_config["input_dir"]
    output_dir = source_config["output_dir"]

    os.makedirs(output_dir, exist_ok=True)

    progress_file = os.path.join(output_dir, f"{source_type}_progress.json")
    progress = load_progress(progress_file)
    processed_files = set(progress["processed_files"])
    current_id = progress["current_id"]

    if not os.path.exists(input_dir):
        print(f"Input directory does not exist: {input_dir}")
        return

    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    if not txt_files:
        print(f"No text files found in {input_dir}")
        return

    remaining_files = [f for f in txt_files if f not in processed_files]

    print(f"\n{'='*60}")
    print(f"Processing: {source_config['name']}")
    print(f"Total files: {len(txt_files)}")
    print(f"Already processed: {len(processed_files)}")
    print(f"Remaining: {len(remaining_files)}")
    print(f"Starting ID: {current_id}")
    print(f"{'='*60}")

    if not remaining_files:
        print("All files already processed!")
                                        
        all_questions = merge_all_chunks(output_dir, source_type)
        if all_questions:
            output_file = os.path.join(output_dir, f"{source_type}_benchmark.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_questions, f, ensure_ascii=False, indent=2)
            print(f"✓ Final benchmark: {len(all_questions)} Q&A pairs")
        return all_questions

    for idx, txt_file in enumerate(remaining_files, 1):
        file_path = os.path.join(input_dir, txt_file)
        total_idx = len(processed_files) + idx
        print(f"\n[{total_idx}/{len(txt_files)}] Processing: {txt_file}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                passage = f.read().strip()

            if not passage:
                print(f"  Empty file, skipping")
                processed_files.add(txt_file)
                save_progress(progress_file, list(processed_files), current_id)
                continue

            print(f"  Read passage with {len(passage)} characters")

            questions = generate_questions(passage, source_type)

            if questions:
                                                     
                formatted_questions = format_qa_output(questions, source_type, current_id, txt_file)

                chunk_name = txt_file.replace('.txt', '')
                save_chunk_questions(output_dir, chunk_name, formatted_questions)

                current_id += len(questions)
                print(f"  ✓ Generated and SAVED {len(questions)} Q&A pairs")
            else:
                print(f"  Could not generate questions for this file")

            processed_files.add(txt_file)
            save_progress(progress_file, list(processed_files), current_id)

        except Exception as e:
            print(f"  Error processing {txt_file}: {str(e)}")
                                                               
            print(f"  ⚠ File NOT marked as processed - will retry on next run")
            continue

    print(f"\n{'='*60}")
    print("Merging all chunks into final benchmark...")
    all_questions = merge_all_chunks(output_dir, source_type)

    output_file = os.path.join(output_dir, f"{source_type}_benchmark.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(all_questions)} Q&A pairs to: {output_file}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error saving final output: {str(e)}")

    return all_questions

def main():
    """Main function to process all sources."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Vietnamese Benchmark Q&A')
    parser.add_argument('--source', choices=['culture', 'law', 'all'], default='all',
                        help='Source to process (culture, law, or all)')
    args = parser.parse_args()

    if args.source == 'all':
        for source_type in SOURCES.keys():
            process_source(source_type)
    else:
        process_source(args.source)

if __name__ == "__main__":
    main()
