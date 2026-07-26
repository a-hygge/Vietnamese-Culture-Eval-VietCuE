"""
Run LLM-as-judge (GPT-5 Azure) trên 247 câu × 4 prompting strategies.

Output: judge_outputs/{strategy}.json — mỗi file 247 scores + reasoning.

Usage:
    python run_judge.py --strategy zero_shot
    python run_judge.py --strategy cot
    python run_judge.py --strategy demonstration
    python run_judge.py --strategy aspect_based
    python run_judge.py --all     # chạy hết 4 strategies tuần tự
    python run_judge.py --strategy zero_shot --limit 5    # test 5 câu đầu
"""
import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    # Load .env ở project root
    root_env = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(root_env):
        load_dotenv(root_env)
except ImportError:
    pass

from judge_prompts import PROMPT_FN, SYSTEM_PROMPT, parse_score

DATA_PATH = r'..\version 02\LLM-as-judge\humancheck\kept_items_full.json'
OUT_DIR = r'../../dataset/judge_ablation_outputs'
os.makedirs(OUT_DIR, exist_ok=True)


def call_gpt5_azure(prompt: str, max_tokens: int = 4000) -> str:
    """Gọi GPT-5 Azure qua LiteLLM. GPT-5 reasoning model cần max_tokens lớn."""
    from litellm import completion
    response = completion(
        model='azure/gpt-5',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        temperature=1,  # GPT-5 reasoning yêu cầu temperature=1
        max_tokens=max_tokens,
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_base=os.getenv('AZURE_OPENAI_ENDPOINT'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
    )
    return (response.choices[0].message.content or '').strip()


def run_strategy(strategy: str, workers: int = 3, limit: int | None = None):
    if strategy not in PROMPT_FN:
        print(f'Unknown strategy: {strategy}. Options: {list(PROMPT_FN.keys())}')
        sys.exit(1)

    print(f'\n{"=" * 70}')
    print(f'Strategy: {strategy}')
    print(f'Workers:  {workers}')
    print(f'{"=" * 70}')

    # Load data
    items = json.load(open(DATA_PATH, encoding='utf-8'))
    items.sort(key=lambda d: d['id'])
    if limit:
        items = items[:limit]
        print(f'⚠️  LIMIT MODE — {limit} câu đầu')
    total = len(items)

    out_file = os.path.join(OUT_DIR, f'{strategy}.json')

    # Resume từ checkpoint
    done = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            saved = json.load(f)
        done = {r['sample_id']: r for r in saved.get('details', [])
                if r.get('score') is not None and not r.get('error')}
        print(f'Resumed: {len(done)} đã xong')

    remaining = [(sid, item) for sid, item in enumerate(items, start=1)
                  if sid not in done]
    if not remaining:
        print('✅ Đã chạy xong hết.')
        return out_file

    # Test connection
    print('Testing GPT-5 Azure...')
    try:
        prompt_fn = PROMPT_FN[strategy]
        test_p = prompt_fn('Test?', 'Test gold', 'Test output')
        test_r = call_gpt5_azure(test_p, max_tokens=500)
        score = parse_score(test_r, strategy)
        print(f'✓ OK. Sample response: {test_r[:80]}... → parsed={score}')
    except Exception as e:
        print(f'❌ Connection test failed: {e}')
        sys.exit(2)

    lock = threading.Lock()
    results = list(done.values())
    start = time.time()
    processed = errors = 0

    def process_one(sid_item):
        sid, item = sid_item
        prompt = PROMPT_FN[strategy](
            item['question'], item['answer'], item['gpt_answer']
        )
        for retry in range(3):
            try:
                raw = call_gpt5_azure(prompt)
                score = parse_score(raw, strategy)
                return {
                    'sample_id': sid,
                    'original_id': item['id'],
                    'category': item['category'],
                    'score': score,
                    'raw_response': raw[:500],  # truncate để file gọn
                    'error': None if score is not None else 'PARSE_FAIL',
                }
            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg or 'rate' in err_msg.lower():
                    time.sleep((retry + 1) * 10)
                else:
                    time.sleep(2)
                if retry == 2:
                    return {
                        'sample_id': sid, 'original_id': item['id'],
                        'category': item['category'], 'score': None,
                        'raw_response': '',
                        'error': err_msg[:300],
                    }

    def save():
        out = {
            'metadata': {
                'strategy': strategy,
                'judge_model': 'azure/gpt-5',
                'base_model': 'gpt_answer (GPT-4o)',
                'data_file': DATA_PATH,
                'total': total,
                'evaluated': len(results),
                'errors': sum(1 for r in results if r.get('error')),
                'last_update': datetime.now().isoformat(),
            },
            'details': sorted(results, key=lambda r: r['sample_id']),
        }
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    # Chạy batches
    batch_size = max(workers * 5, 10)
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start: batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(process_one, x): x for x in batch}
            for fut in as_completed(futures):
                r = fut.result()
                with lock:
                    results.append(r)
                    processed += 1
                    if r.get('error'):
                        errors += 1
                    if processed % 20 == 0:
                        elapsed = time.time() - start
                        rate = processed / elapsed if elapsed else 0
                        eta = (len(remaining) - processed) / rate / 60 if rate else 0
                        print(f'  [{len(results):3d}/{total}] rate={rate:.2f}/s err={errors} eta={eta:.1f}min', flush=True)
        save()

    save()
    elapsed = time.time() - start
    print(f'\n✅ DONE strategy={strategy} — {len(results)}/{total} ({errors} errors) trong {elapsed/60:.1f}min')
    print(f'📁 {out_file}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', choices=list(PROMPT_FN.keys()))
    ap.add_argument('--all', action='store_true', help='Chạy hết 4 strategies tuần tự')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    if args.all:
        for s in PROMPT_FN:
            run_strategy(s, workers=args.workers, limit=args.limit)
    elif args.strategy:
        run_strategy(args.strategy, workers=args.workers, limit=args.limit)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
