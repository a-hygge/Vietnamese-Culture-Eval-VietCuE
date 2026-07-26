"""Framing ablation inference — 6 model × 4 dataset (v1-v4) × 247 câu.

Usage:
    pip install openai python-dotenv
    # đặt API keys trong .env
    python run_framing.py
    # hoặc chạy 1 version, 1 model:
    python run_framing.py --version v2_vn_explicit --model claude-opus-4.6

Output:
    output_framing/{version}/{model}/results.json
    Resume tự động — chạy lại lệnh = skip câu đã xong.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
DATA_DIR = ROOT / 'run_models' / 'data' / 'framing'
OUT_ROOT = ROOT / 'output_framing'
OUT_ROOT.mkdir(exist_ok=True)

MODELS = [
    'claude-opus-4.6',
    'claude-sonnet-4.6',
    'gpt-5',
    'gpt-5.4-pro',
    'gemini-3.1-pro-preview',
    'deepseek-r1',
]

VERSIONS = [
    'v1_original',
    'v2_vn_explicit',
    'v3_tq_explicit',
    'v4_vietnam_english',
]

MAX_TOKENS   = 4096
SYSTEM_PROMPT = ('Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. '
                 'Trả lời ngắn gọn, chính xác, bằng tiếng Việt.')

def _get_client():
    api_key = os.getenv('OPENAI_API_KEY')
    endpoint = os.getenv('OPENAI_BASE_URL')
    return OpenAI(api_key=api_key, base_url=endpoint)

def call_model(model_id: str, question: str) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': question},
        ],
        max_tokens=MAX_TOKENS,
    )
    return (resp.choices[0].message.content or '').strip()

def run_one(version: str, model_name: str, workers: int = 3):
    model_id = model_name
    tag = f'{version}/{model_name}'
    src_path = DATA_DIR / f'{version}.json'
    out_dir  = OUT_ROOT / version / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'results.json'

    items = json.loads(src_path.read_text(encoding='utf-8'))
    total = len(items)

    done = {}
    if out_file.exists():
        saved = json.loads(out_file.read_text(encoding='utf-8'))
        done = {r['id']: r for r in saved.get('details', [])
                if (r.get('answer') or '').strip() and not r.get('error')}

    remaining = [it for it in items if it['id'] not in done]
    if not remaining:
        print(f'  [✓] {tag}: {total}/{total} done — skip')
        return

    print(f'\n=== {tag} — {total - len(remaining)}/{total} done, {len(remaining)} remaining ===')

    lock = threading.Lock()
    results = list(done.values())
    start = time.time()

    def process(it):
        ans = call_model(model_id, it['question'])
        return {
            'id':          it['id'],
            'category':    it['category'],
            'question':    it['question'],
            'gold_answer': it.get('answer', ''),
            'answer':      ans,
            'error':       None,
        }

    def save():
        out = {
            'metadata': {
                'model_name':    model_name,
                'model_id':      model_id,
                'version':       version,
                'max_tokens':    MAX_TOKENS,
                'system_prompt': SYSTEM_PROMPT,
                'data_file':     str(src_path),
                'total_data':    total,
                'evaluated':     len(results),
                'errors':        sum(1 for r in results if r.get('error')),
                'last_update':   datetime.now().isoformat(),
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
                n_done = len(results)
                if n_done % 20 == 0 or n_done == total:
                    elap = time.time() - start
                    rate = (n_done - len(done)) / elap if elap else 0
                    eta  = (total - n_done) / rate / 60 if rate else 0
                    print(f'  [{tag}] [{n_done:3d}/{total}] rate={rate:.2f}/s eta={eta:.1f}min',
                          flush=True)
                if n_done % 20 == 0:
                    save()

    save()
    elap = time.time() - start
    print(f'  [DONE] {tag}: {len(results)}/{total} trong {elap/60:.1f}min')

def main():
    ap = argparse.ArgumentParser(description='Framing ablation: 6 model × 4 dataset × 247 câu')
    ap.add_argument('--version', choices=VERSIONS + ['all'], default='all',
                    help='Dataset version (default: all 4)')
    ap.add_argument('--model', choices=MODELS + ['all'], default='all',
                    help='Model name (default: all 6)')
    ap.add_argument('--workers', type=int, default=32,
                    help='Concurrent questions per run (default: 32)')
    ap.add_argument('--list', action='store_true', help='List models + versions then exit')
    args = ap.parse_args()

    if args.list:
        print('Versions:')
        for v in VERSIONS: print(f'  - {v}')
        print('\nModels:')
        proxy_ok = '✓' if os.getenv('LLM_PROXY_API_KEY') else '✗ (missing LLM_PROXY_API_KEY)'
        for m in MODELS:
            print(f'  - {m:<28} [{proxy_ok}]')
        return

    versions = VERSIONS if args.version == 'all' else [args.version]
    models   = MODELS if args.model == 'all' else [args.model]
    runs     = [(v, m) for v in versions for m in models]

    print(f'Running {len(runs)} run(s) sequentially, {args.workers} workers per run')
    print(f'Hyperparam: max_tokens={MAX_TOKENS}')

    for v, m in runs:
        run_one(v, m, args.workers)

    print(f'\n✅ Done.')

if __name__ == '__main__':
    main()
