"""LLM-match judge cho 19 model trong output_test/ (Vietnamese Culture Benchmark, 247 câu).

Pattern: GPT-5 trả lời true/false — câu trả lời model có khớp ý nghĩa với gold không?

Input:
    output_test/{model_name}/results.json
    Mỗi item: {id, category, question, answer (=gold), {model_name}_answer, ...}

Output (folder riêng, không động vào output_test):
    output_lm_match/{model_name}/results.json
    Schema: list of {id, category, question, gold, model_answer, lm_match}

Resume tự động — chạy lại lệnh = skip item đã có lm_match.

Usage:
    pip install openai python-dotenv
    # đặt OPENAI_API_KEY + OPENAI_BASE_URL (proxy OpenAI-compatible) trong .env
    python run_lm_match.py                      # chạy hết 19 model
    python run_lm_match.py --model gpt-5        # chỉ 1 model
    python run_lm_match.py --workers 32         # tăng workers
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
IN_DIR  = ROOT / 'output_test'
OUT_DIR = ROOT / 'output_lm_match'
OUT_DIR.mkdir(exist_ok=True)

JUDGE_MODEL = 'gpt-5'

def build_prompt(question, gold, model_answer):
    return (
        "Bạn là một giám khảo chấm thi chuyên nghiệp. "
        "Nhiệm vụ của bạn là so sánh câu trả lời của thí sinh với đáp án đúng "
        "và đánh giá xem chúng có khớp về mặt ý nghĩa hay không. "
        "Không cần giống từng chữ, chỉ cần đúng ý chính. "
        "Chỉ trả lời đúng một từ duy nhất: true hoặc false.\n"
        f"Câu hỏi: {question}\n"
        f"Đáp án đúng: {gold}\n"
        f"Câu trả lời của thí sinh: {model_answer}\n"
        "\nĐáp án đúng và câu trả lời của thí sinh có khớp ý nghĩa không?"
    )

def _get_client():
    api_key = os.getenv('OPENAI_API_KEY')
    endpoint = os.getenv('OPENAI_BASE_URL')
    return OpenAI(api_key=api_key, base_url=endpoint)

def judge_match(question, gold, model_answer):
    prompt = build_prompt(question, gold, model_answer)
    client = _get_client()
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=1.0,
            )
            out = (r.choices[0].message.content or '').strip().lower()
            return out == 'true'
        except Exception:
            time.sleep(2 ** attempt)
    return None

def judge_one(model_dir: Path, workers: int):
    model_name = model_dir.name
    in_file = model_dir / 'results.json'
    answer_key = f'{model_name}_answer'

    if not in_file.exists():
        print(f'  ⚠️  {in_file} không tồn tại — skip')
        return

    out_dir  = OUT_DIR / model_name
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / 'results.json'

    print(f'\n{"="*60}')
    print(f'Model: {model_name}')

    src = json.loads(in_file.read_text(encoding='utf-8'))

    if out_file.exists():
        results = json.loads(out_file.read_text(encoding='utf-8'))
        done_ids = {r['id'] for r in results if r.get('lm_match') is not None}
    else:
        results = []
        done_ids = set()

    res_by_id = {r['id']: r for r in results}

    todo = [it for it in src if it['id'] not in done_ids]
    if not todo:
        t = sum(1 for r in results if r.get('lm_match') is True)
        print(f'  ✓ Tất cả {len(results)} item đã judged ({100*t/len(results):.1f}% match) — skip')
        return

    print(f'  Cần judge {len(todo)}/{len(src)} item, {workers} workers')
    lock = threading.Lock()
    start = time.time()
    done_count = [0]

    def proc(it):
        ans = (it.get(answer_key) or '').strip()
        rec = {
            'id':           it['id'],
            'category':     it['category'],
            'question':     it['question'],
            'gold':         it['answer'],
            'model_answer': ans,
        }
        if not ans:
            rec['lm_match'] = False
        else:
            rec['lm_match'] = judge_match(it['question'], it['answer'], ans)
        return rec

    def save():
        out_file.write_text(json.dumps(
            sorted(res_by_id.values(), key=lambda r: r['id']),
            ensure_ascii=False, indent=2), encoding='utf-8')

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(proc, it): it for it in todo}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                res_by_id[rec['id']] = rec
                done_count[0] += 1
                if done_count[0] % 20 == 0:
                    elap = time.time() - start
                    rate = done_count[0] / elap
                    eta  = (len(todo) - done_count[0]) / rate / 60 if rate else 0
                    print(f'    [{done_count[0]:3d}/{len(todo)}] rate={rate:.2f}/s eta={eta:.1f}min',
                          flush=True)
                if done_count[0] % 50 == 0:
                    save()

    save()

    final = list(res_by_id.values())
    t = sum(1 for r in final if r.get('lm_match') is True)
    f = sum(1 for r in final if r.get('lm_match') is False)
    n = sum(1 for r in final if r.get('lm_match') is None)
    elap = time.time() - start
    print(f'  DONE: true={t}  false={f}  null={n}  ({100*t/(t+f) if (t+f) else 0:.1f}% match)  '
          f'in {elap/60:.1f}min  → {out_file.relative_to(ROOT)}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='all',
                    help='Model name (folder trong output_test/) hoặc "all"')
    ap.add_argument('--workers', type=int, default=20,
                    help='Concurrent calls (default: 20)')
    args = ap.parse_args()

    dirs = sorted([d for d in IN_DIR.iterdir() if d.is_dir()])
    if args.model != 'all':
        dirs = [d for d in dirs if d.name == args.model]
        if not dirs:
            print(f'❌ Không tìm thấy {args.model} trong {IN_DIR}')
            print(f'Available: {[d.name for d in IN_DIR.iterdir() if d.is_dir()]}')
            return

    print(f'Sẽ judge {len(dirs)} model với {args.workers} workers:')
    for d in dirs:
        print(f'  - {d.name}')

    for d in dirs:
        judge_one(d, args.workers)

    print('\n✅ Tất cả model đã xong.')

if __name__ == '__main__':
    main()
