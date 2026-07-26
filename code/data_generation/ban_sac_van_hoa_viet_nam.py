import os
import json
import openai
import random

client = openai.AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-02-15-preview"
)

input_root = r"../data_sources/ban_sac_van_hoa_viet_nam/structured_chunks_v2"
output_root = r"../data_question_answer/ban_sac_van_hoa_viet_nam/structured_chunks_v2"
if not os.path.exists(output_root):
    os.makedirs(output_root)
system_prompt_1 = """
Bạn là chuyên gia tạo câu hỏi đánh giá hiểu biết. Đọc đoạn văn dưới đây và tạo 10 câu hỏi chất lượng cao bằng TIẾNG VIỆT.

QUY TẮC BỮT BUỘC - VI PHẠM SẼ BỊ TỪ CHỐI:

1. **TUYỆT ĐỐI KHÔNG dùng các cụm từ:**
    "trong đoạn văn", "theo đoạn văn", "văn bản", "tài liệu", "theo tác giả", "đoạn trích"
    "chương này", "phần này", "bài này", "trong bài"
 Thay bằng tên cụ thể: "UNESCO", "Trần Ngọc Thêm", "văn hóa Việt Nam", v.v.

2. **Câu hỏi phải TỰ HIỂU ĐƯỢC (self-contained):**
   - Người đọc KHÔNG CẦN đọc đoạn văn gốc vẫn hiểu câu hỏi
   - Cung cấp đủ ngữ cảnh cần thiết TRONG câu hỏi
   - KHÔNG tiết lộ đáp án trong phần context

3. **Cụ thể và rõ ràng:**
   - Dùng tên riêng, khái niệm cụ thể
   - Tránh đại từ mơ hồ ("nó", "điều này", "cái đó")

VÍ DỤ:
 SAI: "Văn hóa được định nghĩa như thế nào trong văn bản?"
 ĐÚNG: "Theo UNESCO, văn hóa đóng vai trò gì trong sự phát triển của xã hội?"

 SAI: "Theo đoạn văn, tác giả cho rằng điều gì quan trọng?"
 ĐÚNG: "Trần Ngọc Thêm cho rằng yếu tố nào quan trọng nhất trong việc bảo tồn văn hóa dân tộc?"

 SAI: "Chương này đề cập đến vấn đề gì?"
 ĐÚNG: "Những thách thức chính mà văn hóa Việt Nam gặp phải trong thời kỳ hội nhập là gì?"

ĐỊNH DẠNG JSON YÊU CẦU:
[
  {{
    "cite": "Trích dẫn CHÍNH XÁC câu gốc từ đoạn văn làm cơ sở cho câu hỏi",
    "question_id": {current_cnt},
    "context": "Thông tin nền (nếu cần) - KHÔNG bao gồm đáp án",
    "question": "Câu hỏi (phải tuân thủ tất cả quy tắc trên)",
    "answer": "Đáp án chính xác"
  }},
  ...
]

Tạo 10 câu hỏi, đánh số từ {current_cnt} đến {current_cnt_end}.
"""

self_verification_prompt_1 = """
Kiểm tra và sửa lại các câu hỏi theo CHECKLIST bắt buộc:

 **Bước 1 - Loại bỏ từ ngữ cấm:**
Tìm và XÓA BỎ các cụm từ:
- "văn bản", "đoạn văn", "theo văn bản", "trong đoạn văn", "đoạn trích"
- "chương này", "phần này", "bài này", "tài liệu này"
- "tác giả đề cập", "trong bài"

Thay thế bằng tên cụ thể hoặc diễn đạt trực tiếp.

 **Bước 2 - Đảm bảo câu hỏi tự hiểu được:**
Mỗi câu hỏi phải:
- Người đọc hiểu ngay mà KHÔNG cần đọc đoạn văn gốc
- Có đủ ngữ cảnh cần thiết
- Không có đại từ mơ hồ

 **Bước 3 - Kiểm tra format:**
- Mỗi object có đủ 5 trường: cite, question_id, context, question, answer
- question_id là SỐ NGUYÊN (không phải string)
- Là mảng JSON hợp lệ

 **Bước 4 - Kiểm tra nội dung:**
- 10 câu hỏi khác nhau, không trùng lặp
- Đáp án rõ ràng, chính xác
- Context KHÔNG chứa đáp án

VIẾT LẠI các câu hỏi vi phạm. Trả về mảng JSON đã sửa hoàn chỉnh.
"""

final_prompt = """
Trả về danh sách cuối cùng của các cặp câu hỏi-đáp án ở định dạng JSON hợp lệ.

YÊU CẦU CUỐI CÙNG:
1. Phải là mảng JSON hợp lệ (bắt đầu bằng [ và kết thúc bằng ])
2. Mỗi object có ĐÚNG 5 trường: cite, question_id, context, question, answer
3. question_id phải là số nguyên
4. Tất cả câu hỏi đều bằng TIẾNG VIỆT
5. KHÔNG có từ "văn bản", "đoạn văn", "chương này", "theo tác giả" trong câu hỏi

Chỉ trả về JSON, không giải thích thêm.
"""

current_cnt = 1

def generate_questions(passage, current_count):
    try:
        current_cnt_end = current_count + 9              
        
        response_1 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": system_prompt_1.format(
                    current_cnt=current_count,
                    current_cnt_end=current_cnt_end
                )},
                {"role": "user", "content": f"Đoạn văn:\n\n{passage}\n\nHãy tạo 10 câu hỏi chất lượng cao theo yêu cầu."}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        
        initial_questions = response_1.choices[0].message.content
        print(f"  → Câu hỏi ban đầu được tạo")
        
        response_2 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": self_verification_prompt_1},
                {"role": "user", "content": f"Kiểm tra và sửa lại:\n\n{initial_questions}"}
            ],
            temperature=0.3,
            max_tokens=4000
        )
        
        verified_questions = response_2.choices[0].message.content
        print(f"  → Đã verify và cải thiện")
        
        response_3 = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": verified_questions}
            ],
            temperature=0.1,
            max_tokens=4000
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
            if not all(key in q for key in ['cite', 'question_id', 'context', 'question', 'answer']):
                raise ValueError(f"Thiếu trường trong câu hỏi: {q}")
        
        print(f"  ✓ Validated {len(questions)} câu hỏi")
        return questions
        
    except json.JSONDecodeError as e:
        print(f"  ✗ Lỗi JSON: {str(e)}")
        print(f"  Output: {final_output[:200]}...")
        return None
    except Exception as e:
        print(f"  ✗ Lỗi: {str(e)}")
        return None

def main():
    global current_cnt
    total_questions = 0
    
    txt_files = sorted([f for f in os.listdir(input_root) if f.endswith('.txt')])
    
    print(f"Tìm thấy {len(txt_files)} file trong {input_root}")
    
    for idx, txt_file in enumerate(txt_files, 1):
        file_path = os.path.join(input_root, txt_file)
        print(f"\n[{idx}/{len(txt_files)}] Đang xử lý file: {txt_file}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                passage = f.read().strip()
            
            if not passage:
                print(f"  File rỗng, bỏ qua")
                continue
                
            print(f"  Đã đọc passage với {len(passage)} ký tự")
            
            questions = generate_questions(passage, current_cnt)
            
            if questions:
                                                                       
                output_filename = txt_file.replace('.txt', '.json')
                output_file = os.path.join(output_root, output_filename)
                
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(questions, f, ensure_ascii=False, indent=2)
                    print(f"  ✓ Đã lưu {len(questions)} câu hỏi vào {output_filename}")
                    
                    current_cnt += len(questions)
                    total_questions += len(questions)
                except Exception as e:
                    print(f"  ✗ Lỗi khi lưu file {output_filename}: {str(e)}")
            else:
                print(f"  Không thể tạo câu hỏi cho file này")
                
        except Exception as e:
            print(f"  Lỗi khi đọc file {txt_file}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Hoàn thành! Tổng cộng đã tạo {total_questions} câu hỏi từ {len(txt_files)} file")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
