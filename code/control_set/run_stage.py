"""
Exp 6 - sequential AND-filter pipeline. For stage N, generate the stage's
weak model's answer + judge it (official aspect_based / azure-gpt-5, copied
verbatim from Exp 1/3) ONLY on the candidate-id pool still alive from stage
N-1 (stage 1 = full train-5081). Writes stage{N}_results.json (all attempted
items, with score) and stage{N}_wrong_ids.json (ids with score<=2, i.e. the
input pool for stage N+1).

This sequential design (vs. naively running all 3 models x 5081 items then
intersecting) is cheaper: each later stage only processes the candidates that
survived every earlier stage's "wrong" filter, since final target = candidates
where ALL 3 models are wrong (set intersection - order does not change the
result, only the cost).

Real Azure GPT-5 + real NVIDIA NIM cost - always pilot with --limit first.

Usage:
    python run_stage.py --stage 1 --limit 20     # pilot stage 1
    python run_stage.py --stage 1                # full stage 1 (5081 items)
    python run_stage.py --stage 2                # only stage-1 "wrong" ids
    python run_stage.py --stage 3                # only stage-1 ∩ stage-2 "wrong" ids
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

import openai

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT / '.env')
except ImportError:
    ROOT = Path(__file__).resolve().parents[2]

from weak_models import (NVIDIA_BASE_URL, SYSTEM_PROMPT, MAX_TOKENS, MODELS,
                          STAGE_ORDER, WRONG_THRESHOLD)

HERE = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'dataset' / 'VietCuE_train_5081.json'

# ---------------------------------------------------------------------------
# Judge: copied verbatim from Exp 1/3's rerun_judge.py / rejudge.py
# (code/judge_ablation/{run_judge.py,judge_prompts.py})
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = (
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
            {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        temperature=1,  # GPT-5 reasoning models require temperature=1
        max_tokens=max_tokens,
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    )
    return (response.choices[0].message.content or '').strip()

# ---------------------------- end copied judge section ----------------------


def call_nvidia_nim(model_id: str, api_key: str, question: str) -> str:
    client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': question},
        ],
        max_tokens=MAX_TOKENS,
    )
    return (resp.choices[0].message.content or '').strip()


def load_train():
    items = json.loads(TRAIN_PATH.read_text(encoding='utf-8-sig'))
    return {it['id']: it for it in items}


def get_candidate_ids(stage: int, train_by_id: dict):
    if stage == 1:
        return sorted(train_by_id.keys())
    prev_wrong = ROOT / 'dataset' / 'control_250' / f'stage{stage - 1}_wrong_ids.json'
    if not prev_wrong.exists():
        raise SystemExit(f'{prev_wrong} not found - run stage {stage - 1} first.')
    return json.loads(prev_wrong.read_text(encoding='utf-8'))


def process_one(item, model_id, api_key):
    question = item['question']
    gold = item['answer']

    # 1) generate
    answer = None
    gen_error = None
    for retry in range(3):
        try:
            answer = call_nvidia_nim(model_id, api_key, question)
            break
        except Exception as e:
            gen_error = str(e)[:300]
            time.sleep(10 * (retry + 1))
    if answer is None:
        return {'id': item['id'], 'category': item['category'], 'subcategory': item.get('subcategory'),
                'question': question, 'gold_answer': gold, 'model_answer': '',
                'score': None, 'error': f'GEN_FAIL: {gen_error}'}

    # 2) judge
    prompt = prompt_aspect_based(question, gold, answer)
    for retry in range(3):
        try:
            raw = call_gpt5_azure(prompt)
            score = parse_score(raw)
            return {'id': item['id'], 'category': item['category'], 'subcategory': item.get('subcategory'),
                    'question': question, 'gold_answer': gold, 'model_answer': answer,
                    'score': score, 'raw_judge': raw[:400],
                    'error': None if score is not None else 'PARSE_FAIL'}
        except Exception as e:
            msg = str(e)
            time.sleep(10 * (retry + 1) if ('429' in msg or 'rate' in msg.lower()) else 2)
            if retry == 2:
                return {'id': item['id'], 'category': item['category'], 'subcategory': item.get('subcategory'),
                        'question': question, 'gold_answer': gold, 'model_answer': answer,
                        'score': None, 'error': f'JUDGE_FAIL: {msg[:300]}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True, choices=[1, 2, 3])
    ap.add_argument('--limit', type=int, default=None, help='pilot: first N candidates')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    model_name = STAGE_ORDER[args.stage - 1]
    model_id, key_env = MODELS[model_name]
    api_key = os.getenv(key_env)
    if not api_key:
        raise SystemExit(f'{key_env} not set in .env')

    train_by_id = load_train()
    candidate_ids = get_candidate_ids(args.stage, train_by_id)
    print(f'Stage {args.stage} ({model_name}): {len(candidate_ids)} candidate items')

    if args.limit:
        candidate_ids = candidate_ids[:args.limit]
        print(f'[PILOT] limited to {len(candidate_ids)} items')

    out_path = HERE / f'stage{args.stage}_results.json'
    done = {}
    if out_path.exists():
        saved = json.loads(out_path.read_text(encoding='utf-8'))
        done = {r['id']: r for r in saved.get('details', [])
                if r.get('score') is not None and not r.get('error')}
        print(f'Resumed: {len(done)} already done')

    remaining_ids = [i for i in candidate_ids if i not in done]
    total = len(candidate_ids)
    print(f'{total - len(remaining_ids)}/{total} done, {len(remaining_ids)} remaining')

    lock = threading.Lock()
    results = list(done.values())
    start = time.time()

    def save():
        out = {
            'metadata': {
                'stage': args.stage, 'model_name': model_name, 'model_id': model_id,
                'judge': 'azure/gpt-5', 'strategy': 'aspect_based', 'temperature': 1,
                'total': total, 'evaluated': len(results),
                'errors': sum(1 for r in results if r.get('error')),
                'last_update': datetime.now().isoformat(),
            },
            'details': sorted(results, key=lambda r: r['id']),
        }
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    if remaining_ids:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, train_by_id[i], model_id, api_key): i for i in remaining_ids}
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
        print(f'DONE stage {args.stage}: {len(results)}/{total} in {(time.time()-start)/60:.1f}min')
    else:
        print('Already complete.')

    scored = [r for r in results if r.get('score') is not None]
    wrong_ids = sorted(r['id'] for r in scored if r['score'] <= WRONG_THRESHOLD)
    wrong_path = ROOT / 'dataset' / 'control_250' / f'stage{args.stage}_wrong_ids.json'
    wrong_path.write_text(json.dumps(wrong_ids, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n{model_name}: {len(wrong_ids)}/{len(scored)} scored items are "wrong" '
          f'(score<={WRONG_THRESHOLD}) -> {wrong_path}')


if __name__ == '__main__':
    main()
