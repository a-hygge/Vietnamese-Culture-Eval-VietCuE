"""
Exp 2 (tuqH-W3, TY2X-C2) - Part B: run the 3 genuinely-new open-weight models
(Qwen3.5 397B, Qwen3-Next 80B, Llama 3.3 70B) on the causal-rewrite framing
variants (v1 original, v2 VN-explicit, v3 TQ-explicit) via NVIDIA NIM.

Each model uses its OWN dedicated NVIDIA_API_KEY (not shared rotation, per
user's explicit key->model assignment):
  NVIDIA_API_KEY1 -> qwen3.5-397b-a17b
  NVIDIA_API_KEY2 -> qwen3-next-80b-a3b-instruct
  NVIDIA_API_KEY3 -> meta/llama-3.3-70b-instruct

NIM base URL and call pattern copied from
code/run_models/runner.py's call_nvidia_nim (OpenAI SDK,
base_url override) - same infra already used for the paper's 19-model
leaderboard.

Usage:
    python run_inference.py --limit 10     # pilot
    python run_inference.py                # full: 3 models x 3 versions x 247 items
"""
import argparse
import json
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
    ROOT = Path(__file__).resolve().parents[3]
    load_dotenv(ROOT / '.env')
except ImportError:
    pass

import os

HERE = Path(__file__).resolve().parent
DATASET = Path(__file__).resolve().parents[3] / 'dataset' / 'framing_variants'
OUT_DIR = HERE / 'output_framing'
OUT_DIR.mkdir(exist_ok=True)

NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'
SYSTEM_PROMPT = (
    'Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. '
    'Trả lời ngắn gọn, chính xác, bằng tiếng Việt.'
)
MAX_TOKENS = 4096

# model_name -> (NIM model-id, dedicated API-key env var)
MODELS = {
    'qwen3.5-397b': ('qwen/qwen3.5-397b-a17b', 'NVIDIA_API_KEY1'),
    'qwen3-next-80b': ('qwen/qwen3-next-80b-a3b-instruct', 'NVIDIA_API_KEY2'),
    'llama-3.3-70b': ('meta/llama-3.3-70b-instruct', 'NVIDIA_API_KEY3'),
}

VERSIONS = ['v1_original', 'v2_vn_explicit', 'v3_tq_explicit']


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


def run_one(model_name: str, version: str, limit=None, workers: int = 3):
    model_id, key_env = MODELS[model_name]
    api_key = os.getenv(key_env)
    if not api_key:
        raise SystemExit(f'{key_env} not set in .env')

    items = json.loads((DATASET / f'{version}.json').read_text(encoding='utf-8-sig'))
    if limit:
        items = items[:limit]

    out_dir = OUT_DIR / version / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'results.json'

    done = {}
    if out_file.exists():
        saved = json.loads(out_file.read_text(encoding='utf-8'))
        done = {r['id']: r for r in saved.get('details', [])
                if (r.get('answer') or '').strip() and not r.get('error')}

    remaining = [it for it in items if it['id'] not in done]
    total = len(items)
    if not remaining:
        print(f'  [OK] {model_name}/{version}: {total}/{total} done - skip')
        return

    print(f'\n=== {model_name}/{version}: {total - len(remaining)}/{total} done, '
          f'{len(remaining)} remaining ===')

    lock = threading.Lock()
    results = list(done.values())
    start = time.time()

    def process(it):
        for retry in range(3):
            try:
                ans = call_nvidia_nim(model_id, api_key, it['question'])
                return {'id': it['id'], 'category': it['category'],
                        'question': it['question'], 'gold_answer': it.get('answer', ''),
                        'answer': ans, 'error': None}
            except Exception as e:
                time.sleep(10 * (retry + 1))
                if retry == 2:
                    return {'id': it['id'], 'category': it['category'],
                            'question': it['question'], 'gold_answer': it.get('answer', ''),
                            'answer': '', 'error': str(e)[:300]}

    def save():
        out = {
            'metadata': {
                'model_name': model_name, 'model_id': model_id, 'version': version,
                'provider': 'nvidia_nim', 'max_tokens': MAX_TOKENS,
                'total': total, 'evaluated': len(results),
                'errors': sum(1 for r in results if r.get('error')),
                'last_update': datetime.now().isoformat(),
            },
            'details': sorted(results, key=lambda r: r['id']),
        }
        out_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, it): it for it in remaining}
        for fut in as_completed(futures):
            r = fut.result()
            with lock:
                results.append(r)
                n = len(results)
                if n % 20 == 0 or n == total:
                    elapsed = time.time() - start
                    rate = (n - len(done)) / elapsed if elapsed else 0
                    eta = (total - n) / rate / 60 if rate else 0
                    print(f'  [{model_name}/{version}] [{n}/{total}] rate={rate:.2f}/s eta={eta:.1f}min',
                          flush=True)
                if n % 20 == 0:
                    save()

    save()
    print(f'  [DONE] {model_name}/{version}: {len(results)}/{total} in '
          f'{(time.time()-start)/60:.1f}min')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(MODELS) + ['all'], default='all')
    ap.add_argument('--version', choices=VERSIONS + ['all'], default='all')
    ap.add_argument('--limit', type=int, default=None, help='pilot: first N items per run')
    ap.add_argument('--workers', type=int, default=3)
    args = ap.parse_args()

    models = list(MODELS) if args.model == 'all' else [args.model]
    versions = VERSIONS if args.version == 'all' else [args.version]

    print(f'Running {len(models)} model(s) x {len(versions)} version(s), '
          f'{"PILOT limit=" + str(args.limit) if args.limit else "FULL 247 items"}')

    # BUG (fixed): max_workers=len(models) let the first-submitted model's 3
    # versions grab all 3 slots, starving the other models' v2/v3 entirely.
    # Each model uses its own dedicated key, so all (model, version) slices
    # should run concurrently - cap is just a safety ceiling, not a real limit.
    with ThreadPoolExecutor(max_workers=len(models) * len(versions)) as ex:
        futures = []
        for model_name in models:
            for version in versions:
                futures.append(ex.submit(run_one, model_name, version, args.limit, args.workers))
        for fut in as_completed(futures):
            fut.result()  # surface exceptions

    print('\nAll done.')


if __name__ == '__main__':
    main()
