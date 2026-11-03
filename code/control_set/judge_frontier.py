"""
Exp 6 - Part C: re-judge the 3 frontier models' EXISTING answers (already in
VietCuE_train_5081.json - no new generation needed) on the control_v2_sample
with the SAME official aspect-based GPT-5 judge used for the paper's Table 4 /
test-247 and for Exp 3's control-200. Copied verbatim from Exp 3's rejudge.py.

Requires build_control_sample.py to have been run first.
Requires repo-root .env to have AZURE_OPENAI_API_KEY / _ENDPOINT / _API_VERSION.

Real paid Azure endpoint - pilot with --limit first. Run this AFTER any other
concurrent azure/gpt-5 judge jobs finish (rate-limit contention).

Usage:
    python judge_frontier.py --limit 30      # pilot
    python judge_frontier.py                 # full (n items x 3 models)
    (resumable - rerunning skips pairs already done)
"""
import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / '.env')
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / 'dataset' / 'control_250' / 'control_250.json'
OUT_PATH = ROOT / 'dataset' / 'control_250' / 'control_250_scores.json'

MODELS = ['claude-opus-4.6', 'gemini-2.5-pro', 'gpt-5.4']

# ---------------------------------------------------------------------------
# Copied verbatim from code/judge_ablation/judge_prompts.py
# (prompt_aspect_based / parse_score) and run_judge.py (call_gpt5_azure) - same
# judge as Exp 1/3 and the paper's Table 4.
# ---------------------------------------------------------------------------
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


def parse_score(text: str):
    text = text.strip()
    m = re.search(r'SCORE\s*:\s*([0-5])\b', text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([0-5])\b\s*\.?\s*$', text)
    return int(m.group(1)) if m else None


def call_gpt5_azure(prompt: str, max_tokens: int = 4000) -> str:
    from litellm import completion
    response = completion(
        model='azure/gpt-5',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        temperature=1,  # GPT-5 reasoning models require temperature=1
        max_tokens=max_tokens,
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    )
    return (response.choices[0].message.content or '').strip()

# ---------------------------- end copied section ----------------------------


def build_pairs(records):
    pairs = []
    for rec in records:
        for m in MODELS:
            ans = rec.get(f'{m}_answer')
            if not ans:
                continue
            pairs.append({
                'id': rec['id'],
                'model_name': m,
                'category': rec['category'],
                'subcategory': rec.get('subcategory'),
                'question': rec['question'],
                'gold_answer': rec['gold_answer'],
                'model_answer': ans,
            })
    return pairs


def process_one(rec):
    prompt = prompt_aspect_based(rec['question'], rec['gold_answer'], rec['model_answer'])
    for retry in range(3):
        try:
            raw = call_gpt5_azure(prompt)
            score = parse_score(raw)
            return {
                'id': rec['id'], 'model_name': rec['model_name'],
                'category': rec['category'], 'subcategory': rec.get('subcategory'),
                'score': score, 'raw_response': raw[:500],
                'error': None if score is not None else 'PARSE_FAIL',
            }
        except Exception as e:
            msg = str(e)
            time.sleep(10 * (retry + 1) if ('429' in msg or 'rate' in msg.lower()) else 2)
            if retry == 2:
                return {
                    'id': rec['id'], 'model_name': rec['model_name'],
                    'category': rec['category'], 'subcategory': rec.get('subcategory'),
                    'score': None, 'raw_response': '', 'error': msg[:300],
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--limit', type=int, default=None, help='pilot: only first N pairs')
    args = ap.parse_args()

    if not SAMPLE_PATH.exists():
        raise SystemExit(f'{SAMPLE_PATH} not found - run build_control_sample.py first.')

    records = json.loads(SAMPLE_PATH.read_text(encoding='utf-8'))
    pairs = build_pairs(records)
    if args.limit:
        pairs = pairs[:args.limit]
        print(f'[PILOT MODE] only {len(pairs)} pairs')

    done = {}
    if OUT_PATH.exists():
        saved = json.loads(OUT_PATH.read_text(encoding='utf-8'))
        done = {(d['id'], d['model_name']): d for d in saved.get('details', [])
                if d.get('score') is not None and not d.get('error')}
        print(f'Resumed: {len(done)} already done')

    remaining = [r for r in pairs if (r['id'], r['model_name']) not in done]
    total = len(pairs)
    print(f'{total - len(remaining)}/{total} done, {len(remaining)} remaining')

    if not remaining:
        print('Already complete.')
        return

    lock = threading.Lock()
    results = list(done.values())
    start = time.time()

    def save():
        out = {
            'metadata': {
                'judge': 'azure/gpt-5', 'strategy': 'aspect_based', 'temperature': 1,
                'max_tokens': 4000, 'total': total, 'evaluated': len(results),
                'errors': sum(1 for r in results if r.get('error')),
                'last_update': datetime.now().isoformat(),
            },
            'details': results,
        }
        OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, r): r for r in remaining}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                n = len(results)
                if n % 20 == 0 or n == total:
                    elapsed = time.time() - start
                    rate = (n - len(done)) / elapsed if elapsed else 0
                    eta = (total - n) / rate / 60 if rate else 0
                    print(f'  [{n}/{total}] rate={rate:.2f}/s eta={eta:.1f}min', flush=True)
                if n % 20 == 0:
                    save()

    save()
    elapsed = time.time() - start
    print(f'DONE: {len(results)}/{total} in {elapsed/60:.1f}min -> {OUT_PATH}')


if __name__ == '__main__':
    main()
