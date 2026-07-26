"""
Data Quality Checker - "Nguoi khong biet gi ve sach"

Kiem tra du lieu benchmark theo goc nhin cua nguoi CHUA DOC SACH.
Phat hien 5 loai loi chinh:
1. Loi Mat Ngu Canh (Context Decontextualization)
2. Loi Tri thuc gia dinh (Assumed Knowledge)
3. Loi Nhieu Du lieu (Data Noise)
4. Loi Mo ho (Ambiguity)
5. Loi Do dai/Cau truc (Chunking Strategy)

Output: Them cot "data_quality_issues" vao ket qua danh gia
"""

import re
import json
from typing import List, Dict, Tuple

CRITICAL_ISSUE_TYPES = [
    "TRUNG_LAP",                       
    "NHIEU",                                                  
    "MO_HO",                                        
    "KHO_HIEU",                                                    
    "NHAY_CAM",                                                          
]

WARNING_ISSUE_TYPES = [
    "NGU_CANH",                                                  
    "TRI_THUC",                                                         
    "CAU_TRUC",                                                 
]

VAGUE_PRONOUNS = [
    r'\b(anh ấy|cô ấy|họ|hắn|nó|ông ấy|bà ấy|chị ấy|em ấy)\b',
    r'\b(việc đó|điều đó|việc này|điều này|cái đó|cái này)\b',
    r'\b(ở đây|ở đó|nơi đây|nơi đó|chỗ đó|chỗ này)\b',
    r'\b(lúc đó|khi đó|lúc ấy|khi ấy|hồi đó)\b',
    r'\b(như vậy|như thế|thế này|thế kia)\b',
]

UNEXPLAINED_TERMS = [
    r'\b([A-Z]{2,})\b',                   
]

NOISE_PATTERNS = [
    r'Trang\s*\d+',            
    r'Chuong\s*\d+',              
    r'\[\d+\]',              
    r'\(\d+\)',                  
    r'^\d+\.\s',                   
]

WEAK_CONTEXT_PATTERNS = [
    r'^Sau đây là câu hỏi về',
    r'^Đây là câu hỏi',
    r'^Câu hỏi về',
    r'^Hãy trả lời',
]

CONFUSING_PATTERNS = [
    r'\?\s*\?',                 
    r'\.\.\.',                
    r'^\s*$',              
    r'^.{0,10}\?$',                                    
]

SENSITIVE_POLITICS = [
    r'\b(chống phá|lật đổ|phản động|thù địch)\b',
    r'\b(biểu tình|bạo loạn|nổi dậy|đảo chính)\b',
    r'\b(tuyên truyền chống|xuyên tạc|bôi nhọ)\b',
    r'\b(chế độ (cũ|ngụy|bù nhìn))\b',
    r'\b(cờ vàng|việt nam cộng hòa|vnch)\b',
    r'\b(phản cách mạng|chống cộng)\b',
]

SENSITIVE_RELIGION = [
    r'\b(tà đạo|dị giáo|mê tín dị đoan)\b',
    r'\b(chống (phật|công giáo|tin lành|hồi giáo))\b',
    r'\b(xúc phạm (thánh|thần|phật|chúa))\b',
]

SENSITIVE_DISCRIMINATION = [
    r'\b(da (đen|vàng|trắng) (là|thì|đều))\b',
    r'\b(dân tộc.*(thấp kém|lạc hậu|ngu))\b',
    r'\b(phụ nữ.*(yếu đuối|kém|không nên))\b',
    r'\b(đồng tính.*(bệnh|tội|lệch lạc))\b',
    r'\b(người (khuyết tật|tàn tật).*(gánh nặng|vô dụng))\b',
]

SENSITIVE_TERRITORY = [
    r'\b(hoàng sa|trường sa).*(của (trung quốc|trung hoa))\b',
    r'\b(biển (đông|nam trung hoa)).*(thuộc|của) trung quốc\b',
    r'\b(đường lưỡi bò|đường chín đoạn)\b',
]

SENSITIVE_VIOLENCE = [
    r'\b(cách (chế tạo|làm) (bom|vũ khí|thuốc nổ))\b',
    r'\b(giết người|tra tấn|hành hình)\b',
    r'\b(khủng bố|tấn công|đánh bom)\b',
]

SENSITIVE_ADULT = [
    r'\b(quan hệ tình dục|giao cấu|hiếp dâm)\b',
    r'\b(mại dâm|đĩ điếm|gái gọi)\b',
    r'\b(khiêu dâm|đồi trụy)\b',
]

def check_context_loss(item: Dict) -> List[str]:
    """
    Kiem tra Loi Mat Ngu Canh (Context Decontextualization)
    - Dai tu khong ro rang
    - Thieu moc thoi gian/dia diem
    - Context qua yeu
    """
    issues = []
    context = item.get("context", "")
    question = item.get("question", "")
    answer = item.get("answer", "")

    full_text = f"{context} {question} {answer}"

    for pattern in VAGUE_PRONOUNS:
        matches = re.findall(pattern, full_text, re.IGNORECASE)
        if matches:
            issues.append(f"[NGU_CANH] Dai tu mo ho: {', '.join(set(matches))}")
            break                     

    for pattern in WEAK_CONTEXT_PATTERNS:
        if re.match(pattern, context, re.IGNORECASE):
            issues.append("[NGU_CANH] Context khong cung cap du thong tin de tra loi")
            break

    if len(context.strip()) < 50:
        issues.append("[NGU_CANH] Context qua ngan")

    return issues

def check_assumed_knowledge(item: Dict) -> List[str]:
    """
    Kiem tra Loi Tri thuc gia dinh (Assumed Knowledge)
    - Cau hoi yeu cau thong tin khong co trong context
    - Cau hoi qua rong
    """
    issues = []
    context = item.get("context", "")
    question = item.get("question", "")
    answer = item.get("answer", "")

    if re.search(r'\b(tại sao|vì sao|nguyên nhân)\b', question, re.IGNORECASE):
        if len(context) < 100:
            issues.append("[TRI_THUC] Cau hoi 'tai sao' nhung context khong giai thich nguyen nhan")

    if re.search(r'\b(những gì|bao gồm|liệt kê|các loại)\b', question, re.IGNORECASE):
        if len(context) < 100:
            issues.append("[TRI_THUC] Cau hoi liet ke nhung context khong du thong tin")

    if re.search(r'\b(ai|người nào|đối tượng nào)\b', question, re.IGNORECASE):
        if not re.search(r'\b(cá nhân|tổ chức|công dân|người)\b', context, re.IGNORECASE):
            issues.append("[TRI_THUC] Cau hoi ve 'ai' nhung context khong nhac den chu the")

    return issues

def check_data_noise(item: Dict) -> List[str]:
    """
    Kiem tra Loi Nhieu Du lieu (Data Noise)
    - Header/Footer dinh vao noi dung
    - So trang, ten chuong
    - Loi xuong dong sai
    """
    issues = []
    context = item.get("context", "")
    question = item.get("question", "")
    answer = item.get("answer", "")

    full_text = f"{context} {question} {answer}"

    for pattern in NOISE_PATTERNS:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append(f"[NHIEU] Co the chua nhieu: {pattern}")
            break

    if re.search(r'\w-\s*\n\s*\w', full_text):
        issues.append("[NHIEU] Cau bi ngat giua chung do xuong dong")

    return issues

def check_ambiguity(item: Dict) -> List[str]:
    """
    Kiem tra Loi Mo ho (Ambiguity)
    - Cau hoi thieu dinh danh
    - Cau tra loi chu quan
    - Nhieu dap an co the dung
    """
    issues = []
    question = item.get("question", "")
    answer = item.get("answer", "")

    vague_question_patterns = [
        r'^(gì|như thế nào|ra sao)\?$',
        r'\b(ông ấy|bà ấy|người đó)\b',
    ]

    for pattern in vague_question_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            issues.append("[MO_HO] Cau hoi thieu dinh danh cu the")
            break

    subjective_patterns = [
        r'\b(bạn nghĩ|theo bạn|ý kiến của bạn)\b',
        r'\b(cảm thấy|cảm nghĩ|cảm xúc)\b',
        r'\b(tốt hay xấu|đúng hay sai)\b',
    ]

    for pattern in subjective_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            issues.append("[MO_HO] Cau hoi mang tinh chu quan")
            break

    if len(answer) < 30:
        issues.append("[MO_HO] Cau tra loi qua ngan, co the mo ho")

    return issues

def check_chunking_issues(item: Dict) -> List[str]:
    """
    Kiem tra Loi Do dai/Cau truc (Chunking Strategy)
    - Context qua ngan/dai
    - Cat giua cau quan trong
    """
    issues = []
    context = item.get("context", "")
    answer = item.get("answer", "")

    if len(context.strip()) < 50:
        issues.append("[CAU_TRUC] Context qua ngan de cung cap du thong tin")

    if len(context.strip()) > 2000:
        issues.append("[CAU_TRUC] Context qua dai, co the gay kho khan tim thong tin")

    if context.strip() and not context.strip().endswith(('.', '?', '!', '"', ')', ']')):
        issues.append("[CAU_TRUC] Context co the bi cat giua cau")

    return issues

def check_confusing_question(item: Dict) -> List[str]:
    """
    Kiem tra Cau hoi kho hieu (Confusing/Unclear)
    - Ngu phap sai
    - Cau hoi khong ro rang
    - Cau hoi qua ngan/rong
    """
    issues = []
    question = item.get("question", "")
    answer = item.get("answer", "")

    if not question.strip():
        issues.append("[KHO_HIEU] Cau hoi trong")
        return issues

    if len(question.strip()) < 15:
        issues.append("[KHO_HIEU] Cau hoi qua ngan, co the khong ro rang")

    if '?' not in question:
        issues.append("[KHO_HIEU] Cau hoi khong co dau cham hoi")

    for pattern in CONFUSING_PATTERNS:
        if re.search(pattern, question):
            issues.append("[KHO_HIEU] Cau hoi co dinh dang bat thuong")
            break

    if len(answer.strip()) < 10:
        issues.append("[KHO_HIEU] Cau tra loi qua ngan, cau hoi co the khong cu the")

    if question.count(',') > 5 or question.count('và') > 4:
        issues.append("[KHO_HIEU] Cau hoi qua phuc tap, nhieu menh de")

    return issues

def check_sensitive_content(item: Dict) -> List[str]:
    """
    Kiem tra Noi dung nhay cam (Sensitive Content)
    - Chinh tri
    - Ton giao
    - Phan biet doi xu
    - Lanh tho/Chu quyen
    - Bao luc
    - Noi dung nguoi lon
    """
    issues = []
    question = item.get("question", "").lower()
    answer = item.get("answer", "").lower()
    context = item.get("context", "").lower()

    full_text = f"{context} {question} {answer}"

    for pattern in SENSITIVE_POLITICS:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append("[NHAY_CAM] Noi dung chinh tri nhay cam")
            break

    for pattern in SENSITIVE_RELIGION:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append("[NHAY_CAM] Noi dung ton giao nhay cam")
            break

    for pattern in SENSITIVE_DISCRIMINATION:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append("[NHAY_CAM] Noi dung phan biet doi xu")
            break

    for pattern in SENSITIVE_TERRITORY:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append("[NHAY_CAM] Noi dung ve lanh tho/chu quyen nhay cam")
            break

    for pattern in SENSITIVE_VIOLENCE:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append("[NHAY_CAM] Noi dung bao luc/khung bo")
            break

    for pattern in SENSITIVE_ADULT:
        if re.search(pattern, full_text, re.IGNORECASE):
            issues.append("[NHAY_CAM] Noi dung nguoi lon")
            break

    return issues

def check_duplicate_questions(items: List[Dict]) -> Dict[str, List[str]]:
    """
    Kiem tra cau hoi trung lap hoac qua giong nhau
    """
    duplicates = {}
    questions_seen = {}

    for item in items:
        item_id = item.get("id", "")
        question = item.get("question", "").lower().strip()

        normalized = re.sub(r'[^\w\s]', '', question)
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        if normalized in questions_seen:
            if item_id not in duplicates:
                duplicates[item_id] = []
            duplicates[item_id].append(f"[TRUNG_LAP] Trung voi cau hoi {questions_seen[normalized]}")
        else:
            questions_seen[normalized] = item_id

    return duplicates

def analyze_item(item: Dict) -> List[str]:
    """
    Phan tich mot item va tra ve danh sach loi
    """
    all_issues = []

    all_issues.extend(check_context_loss(item))
    all_issues.extend(check_assumed_knowledge(item))
    all_issues.extend(check_data_noise(item))
    all_issues.extend(check_ambiguity(item))
    all_issues.extend(check_chunking_issues(item))
    all_issues.extend(check_confusing_question(item))       
    all_issues.extend(check_sensitive_content(item))        

    return all_issues

def classify_issues(issues: List[str]) -> Tuple[List[str], List[str]]:
    """
    Phan loai issues thanh CRITICAL va WARNING.

    Returns:
        Tuple of (critical_issues, warning_issues)
    """
    critical = []
    warning = []

    for issue in issues:
        is_critical = False
        for crit_type in CRITICAL_ISSUE_TYPES:
            if f"[{crit_type}]" in issue:
                critical.append(issue)
                is_critical = True
                break
        if not is_critical:
            warning.append(issue)

    return critical, warning

def get_quality_status(issues: List[str]) -> str:
    """
    Tra ve trang thai chat luong:
    - "OK": Khong co loi
    - "OK (warning)": Chi co loi cau truc, cau hoi van on
    - "CRITICAL": Co loi noi dung can sua
    """
    if not issues:
        return "OK"

    critical, warning = classify_issues(issues)

    if critical:
        return "CRITICAL"
    else:
        return "OK (warning)"

def analyze_benchmark(data: List[Dict]) -> Dict[str, List[str]]:
    """
    Phan tich toan bo benchmark va tra ve dict {id: [issues]}
    """
    results = {}

    for item in data:
        item_id = item.get("id", "")
        issues = analyze_item(item)
        if issues:
            results[item_id] = issues

    duplicates = check_duplicate_questions(data)
    for item_id, dup_issues in duplicates.items():
        if item_id in results:
            results[item_id].extend(dup_issues)
        else:
            results[item_id] = dup_issues

    return results

def analyze_benchmark_with_status(data: List[Dict]) -> Dict[str, Dict]:
    """
    Phan tich benchmark va tra ve dict voi status.

    Returns:
        Dict {id: {"issues": [...], "status": "OK" | "OK (warning)" | "CRITICAL"}}
    """
    all_issues = analyze_benchmark(data)
    results = {}

    for item in data:
        item_id = item.get("id", "")
        issues = all_issues.get(item_id, [])
        status = get_quality_status(issues)

        results[item_id] = {
            "issues": issues,
            "status": status,
            "critical": [],
            "warning": []
        }

        if issues:
            critical, warning = classify_issues(issues)
            results[item_id]["critical"] = critical
            results[item_id]["warning"] = warning

    return results

def generate_quality_report(data: List[Dict]) -> str:
    """
    Tao bao cao chat luong du lieu
    """
    results = analyze_benchmark_with_status(data)

    error_counts = {
        "NGU_CANH": 0,
        "TRI_THUC": 0,
        "NHIEU": 0,
        "MO_HO": 0,
        "CAU_TRUC": 0,
        "TRUNG_LAP": 0,
        "KHO_HIEU": 0,
        "NHAY_CAM": 0,
    }

    status_counts = {
        "OK": 0,
        "OK (warning)": 0,
        "CRITICAL": 0,
    }

    for item_id, info in results.items():
        status_counts[info["status"]] += 1
        for issue in info["issues"]:
            for error_type in error_counts.keys():
                if f"[{error_type}]" in issue:
                    error_counts[error_type] += 1

    total_items = len(data)
    items_ok = status_counts["OK"] + status_counts["OK (warning)"]
    items_critical = status_counts["CRITICAL"]

    report = f"""
================================================================================
BAO CAO CHAT LUONG DU LIEU - GOC NHIN "NGUOI CHUA DOC SACH"
================================================================================

TONG QUAN:
- Tong so cau hoi: {total_items}

PHAN LOAI TRANG THAI:
  ✓ OK (hoan toan tot):        {status_counts['OK']} cau
  ✓ OK (warning - van on):     {status_counts['OK (warning)']} cau ({status_counts['OK (warning)']/total_items*100:.1f}%)
  ✗ CRITICAL (can sua):        {status_counts['CRITICAL']} cau ({status_counts['CRITICAL']/total_items*100:.1f}%)

=> CAU HOI VAN ON (OK + OK warning): {items_ok} ({items_ok/total_items*100:.1f}%)
=> CAN XEM LAI (CRITICAL):           {items_critical} ({items_critical/total_items*100:.1f}%)

--------------------------------------------------------------------------------
CHI TIET LOAI LOI:
--------------------------------------------------------------------------------

[WARNING - Cau hoi van OK, chi la van de cau truc context]:
  - NGU_CANH (Context ngan):    {error_counts['NGU_CANH']} loi
  - TRI_THUC (Tri thuc chung):  {error_counts['TRI_THUC']} loi
  - CAU_TRUC (Cau truc):        {error_counts['CAU_TRUC']} loi

[CRITICAL - Can sua noi dung]:
  - TRUNG_LAP (Trung lap):      {error_counts['TRUNG_LAP']} loi
  - NHIEU (Nhieu du lieu):      {error_counts['NHIEU']} loi
  - MO_HO (Mo ho):              {error_counts['MO_HO']} loi
  - KHO_HIEU (Kho hieu):        {error_counts['KHO_HIEU']} loi
  - NHAY_CAM (Nhay cam):        {error_counts['NHAY_CAM']} loi

================================================================================
"""
    return report

if __name__ == "__main__":
    import os
    import sys
    import argparse

    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description='Data Quality Checker')
    parser.add_argument('--topic', type=str, default='all',
                        choices=['all', 'culture', 'law'],
                        help='Topic to analyze: all, culture, law')
    parser.add_argument('--detail', type=int, default=10,
                        help='Number of detailed issues to show')
    args = parser.parse_args()

    base_dir = r"../data_question_answer"

    benchmark_files = {
        "culture": {
            "name": "Ban sac van hoa Viet Nam",
            "path": os.path.join(base_dir, "ban_sac_van_hoa_viet_nam", "culture_benchmark.json")
        },
        "law": {
            "name": "Phap luat dai cuong",
            "path": os.path.join(base_dir, "bai_giang_phap_luat_dai_cuong", "law_benchmark.json")
        }
    }

    if args.topic == 'all':
        topics_to_analyze = ['culture', 'law']
    else:
        topics_to_analyze = [args.topic]

    all_data = []
    for topic in topics_to_analyze:
        info = benchmark_files[topic]
        if os.path.exists(info["path"]):
            with open(info["path"], 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Da load {len(data)} cau hoi tu {info['name']}")
            all_data.extend(data)
        else:
            print(f"Khong tim thay file: {info['path']}")

    if all_data:
        print(f"\nTong cong: {len(all_data)} cau hoi")

        issues = analyze_benchmark(all_data)

        print(generate_quality_report(all_data))

        print(f"\nCHI TIET {args.detail} CAU DAU CO VAN DE:")
        print("-" * 60)
        count = 0
        for item_id, item_issues in issues.items():
            if count >= args.detail:
                break
            print(f"\n{item_id}:")
            for issue in item_issues:
                print(f"  - {issue}")
            count += 1

        if args.topic == 'all':
            print("\n" + "=" * 60)
            print("THONG KE THEO CHU DE:")
            print("=" * 60)

            for topic in topics_to_analyze:
                info = benchmark_files[topic]
                topic_data = [d for d in all_data if d.get("category") == topic]
                if topic_data:
                    topic_issues = analyze_benchmark(topic_data)
                    print(f"\n{info['name'].upper()}:")
                    print(f"  - Tong so cau: {len(topic_data)}")
                    print(f"  - Cau co van de: {len(topic_issues)} ({len(topic_issues)/len(topic_data)*100:.1f}%)")
    else:
        print("Khong co du lieu de phan tich")
