"""
Exp 2 (tuqH-W3, TY2X-C2) - Part C: judge the 3 new open-weight models' framing
outputs with the SAME aspect-based GPT-5 judge used everywhere else in the
paper (and in Exp 1/3 of this rebuttal). Judge call/prompt/parser copied
verbatim from code/judge_ablation/{run_judge.py,
judge_prompts.py} - same judge that produced the existing 6-model
framing_all4_summary.json, so the new 3 models are directly comparable.

Requires run_inference.py to have completed (or partially - this only judges
whatever exists in output_framing/).

Usage:
    python judge.py --limit 10          # pilot
    python judge.py                     # judge everything available
    python judge.py --model qwen3.5-397b --version v1_original   # one slice
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

ROOT = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'output_framing'
SCORES_DIR = ROOT / 'dataset' / 'framing_variants' / 'judge_outputs'
SCORES_DIR.mkdir(exist_ok=True)

MODELS = ['qwen3.5-397b', 'qwen3-next-80b', 'llama-3.3-70b']
VERSIONS = ['v1_original', 'v2_vn_explicit', 'v3_tq_explicit']

# ---------------------------------------------------------------------------
# Copied verbatim from judge_ablation/judge_prompts.py + run_judge.py -
# identical judge used for Exp 1, Exp 3, and the paper's Table 4.
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
        temperature=1,
        max_tokens=max_tokens,
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    )
    return (response.choices[0].message.content or '').strip()

# ---------------------------- end copied section ----------------------------


def process_one(rec):
    prompt = prompt_aspect_based(rec['question'], rec['gold_answer'], rec['answer'])
    for retry in range(3):
        try:
            raw = call_gpt5_azure(prompt)
            score = parse_score(raw)
            return {'id': rec['id'], 'category': rec['category'], 'score': score,
                    'raw_response': raw[:500],
                    'error': None if score is not None else 'PARSE_FAIL'}
        except Exception as e:
            msg = str(e)
            time.sleep(10 * (retry + 1) if ('429' in msg or 'rate' in msg.lower()) else 2)
            if retry == 2:
                return {'id': rec['id'], 'category': rec['category'], 'score': None,
                        'raw_response': '', 'error': msg[:300]}


def judge_one_slice(model_name: str, version: str, limit=None, workers: int = 3):
    src = OUT_DIR / version / model_name / 'results.json'
    if not src.exists():
        print(f'  [!] {src} not found - skip (run_inference.py not done for this slice yet)')
        return

    data = json.loads(src.read_text(encoding='utf-8'))
    items = [d for d in data['details'] if (d.get('answer') or '').strip() and not d.get('error')]
    if limit:
        items = items[:limit]
    total = len(items)

    out_file = SCORES_DIR / f'{version}__{model_name}_judged.json'
    done = {}
    if out_file.exists():
        saved = json.loads(out_file.read_text(encoding='utf-8'))
        done = {d['id']: d for d in saved.get('details', [])
                if d.get('score') is not None and not d.get('error')}

    remaining = [it for it in items if it['id'] not in done]
    if not remaining:
        print(f'  [OK] {version}/{model_name}: {total}/{total} judged - skip')
        return

    print(f'\n=== judging {version}/{model_name}: {total - len(remaining)}/{total} done, '
          f'{len(remaining)} remaining ===')

    lock = threading.Lock()
    results = list(done.values())
    start = time.time()

    def save():
        out = {
            'metadata': {
                'model': model_name, 'version': version, 'judge': 'azure/gpt-5',
                'strategy': 'aspect_based', 'temperature': 1, 'max_tokens': 4000,
                'total': total, 'evaluated': len(results),
                'errors': sum(1 for r in results if r.get('error')),
                'last_update': datetime.now().isoformat(),
            },
            'details': sorted(results, key=lambda r: r['id']),
        }
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, it): it for it in remaining}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                n = len(results)
                if n % 20 == 0 or n == total:
                    elapsed = time.time() - start
                    rate = (n - len(done)) / elapsed if elapsed else 0
                    eta = (total - n) / rate / 60 if rate else 0
                    print(f'  [{version}/{model_name}] [{n}/{total}] rate={rate:.2f}/s eta={eta:.1f}min',
                          flush=True)
                if n % 20 == 0:
                    save()

    save()
    print(f'  [DONE] {version}/{model_name}: {len(results)}/{total} in '
          f'{(time.time()-start)/60:.1f}min -> {out_file}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS + ['all'], default='all')
    ap.add_argument('--version', choices=VERSIONS + ['all'], default='all')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    models = MODELS if args.model == 'all' else [args.model]
    versions = VERSIONS if args.version == 'all' else [args.version]

    for model_name in models:
        for version in versions:
            judge_one_slice(model_name, version, args.limit, args.workers)

    print('\nAll requested slices processed.')


if __name__ == '__main__':
    main()
