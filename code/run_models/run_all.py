"""
Chạy NHIỀU model tuần tự qua FOR loop.

Mỗi model output 1 file riêng vào ./outputs/{provider}__{model}.json
Nếu 1 model lỗi → log lại nhưng vẫn tiếp tục chạy model kế tiếp.

Usage:
    # Chạy tất cả models trong PRIORITY_HIGH (mặc định)
    python run_all.py

    # Chạy tất cả 30 models
    python run_all.py --all

    # Chạy 1 nhóm cụ thể
    python run_all.py --group api          # chỉ API models (skip Ollama)
    python run_all.py --group ollama       # chỉ open-source qua Ollama
    python run_all.py --group openai
    python run_all.py --group anthropic
    python run_all.py --group gemini

    # Chạy danh sách tự chọn
    python run_all.py --models gpt-5.5 claude-opus-4.7 gemini-3.1-pro

    # Skip model đã chạy xong (resume cấp ngoài)
    python run_all.py --skip-done

    # Workers per model
    python run_all.py --workers 5
"""
import argparse
import os
import sys
import time
import traceback
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from models_config import MODELS
from runner import run_model, OUTPUT_DIR, safe_filename

API_MODELS = [
    'gpt-5.5', 'gpt-5.4',
    'claude-opus-4.7', 'claude-sonnet-4.6-thinking', 'claude-haiku-4.5',
    'gemini-3.1-pro', 'gemini-2.5-pro',
    'deepseek-v3', 'qwen3-max',
]

OPEN_SOURCE = [
    'llama-3.3-70b',
    'qwen3.5-397b', 'qwen3-next-80b',
    'gpt-oss-120b', 'gpt-oss-20b',
    'gemma-3-27b',
    'deepseek-v3.2-nim',
]

def select_models(args) -> list[str]:
    if args.models:
        return args.models
    if args.all:
        return list(MODELS.keys())
    if args.group == 'api':
        return API_MODELS
    if args.group == 'nim':
        return OPEN_SOURCE
    if args.group == 'openai':
        return [m for m, c in MODELS.items() if c['model'].startswith('openai/')]
    if args.group == 'anthropic':
        return [m for m, c in MODELS.items() if c['model'].startswith('anthropic/')]
    if args.group == 'gemini':
        return [m for m, c in MODELS.items() if c['model'].startswith('gemini/')]
                      
    return list(MODELS.keys())

def is_done(model_name: str) -> bool:
    """File output có và đủ 247 items + 0 errors → coi là done."""
    cfg = MODELS.get(model_name)
    if not cfg:
        return False
    fp = os.path.join(OUTPUT_DIR, f'{safe_filename(cfg["full_name"])}.json')
    if not os.path.exists(fp):
        return False
    try:
        import json
        with open(fp, encoding='utf-8') as f:
            d = json.load(f)
        meta = d.get('metadata', {})
        ok = sum(1 for r in d.get('details', []) if r.get('answer') and not r.get('error'))
        return ok >= meta.get('total_data', 247)
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description='Chạy nhiều model qua FOR loop')
    parser.add_argument('--models', nargs='+', help='Danh sách model name tự chọn')
    parser.add_argument('--all', action='store_true', help='Chạy tất cả 30 models')
    parser.add_argument('--group',
        choices=['api', 'nim', 'openai', 'anthropic', 'gemini'],
        help='Chạy 1 nhóm preset. Mặc định = tất cả 16 models. api = 9 close-source, nim = 7 open-source')
    parser.add_argument('--workers', type=int, default=3, help='Workers / model (default 3)')
    parser.add_argument('--skip-done', action='store_true',
        help='Bỏ qua model đã chạy đủ 247 / 0 errors (resume cấp ngoài)')
    parser.add_argument('--limit', type=int, default=None,
        help='DEBUG: chỉ chạy N câu đầu (truyền xuống runner)')
    parser.add_argument('--stop-on-error', action='store_true',
        help='Dừng nếu 1 model fail (default: log + tiếp tục)')
    args = parser.parse_args()

    todo = select_models(args)
    print(f'\n{"=" * 70}')
    print(f'BATCH RUN — {len(todo)} models')
    print(f'  Group: {args.group or ("all" if args.all else "default=high")}')
    print(f'  Workers/model: {args.workers}')
    print(f'  Skip done: {args.skip_done}')
    print(f'{"=" * 70}\n')
    for i, m in enumerate(todo, 1):
        marker = ' (DONE)' if is_done(m) else ''
        print(f'  {i:2d}. {m}{marker}')
    print()

    log_file = os.path.join(OUTPUT_DIR, f'_run_all_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(f'[{datetime.now().isoformat(timespec="seconds")}] {msg}')

    overall_start = time.time()
    success, skipped, failed = [], [], []

    for i, model_name in enumerate(todo, 1):
        log(f'\n{"#" * 70}')
        log(f'# [{i}/{len(todo)}] {model_name}')
        log(f'{"#" * 70}')

        if args.skip_done and is_done(model_name):
            log(f'⏭️  SKIP — đã có file output đủ 247 / 0 errors')
            skipped.append(model_name)
            continue

        if model_name not in MODELS:
            log(f'❌ Unknown model: {model_name}')
            failed.append((model_name, 'unknown'))
            continue

        try:
            run_model(model_name, max_workers=args.workers, limit=args.limit)
            success.append(model_name)
        except KeyboardInterrupt:
            log('\n⏹️  Interrupted by user')
            break
        except SystemExit as e:
            log(f'❌ {model_name} → SystemExit({e.code}) [thường = test connection failed: thiếu API key hoặc model ID sai]')
            failed.append((model_name, f'SystemExit({e.code})'))
            if args.stop_on_error:
                break
        except Exception as e:
            log(f'❌ {model_name} → {type(e).__name__}: {e}')
            log(traceback.format_exc())
            failed.append((model_name, f'{type(e).__name__}: {str(e)[:200]}'))
            if args.stop_on_error:
                break

    elapsed = time.time() - overall_start
    log(f'\n{"=" * 70}')
    log(f'BATCH SUMMARY — total {elapsed/60:.1f} min')
    log(f'{"=" * 70}')
    log(f'✅ Success ({len(success)}): {", ".join(success) or "—"}')
    log(f'⏭️  Skipped ({len(skipped)}): {", ".join(skipped) or "—"}')
    log(f'❌ Failed  ({len(failed)}):')
    for m, err in failed:
        log(f'    {m}  →  {err}')

    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log(f'\n📁 Log saved: {log_file}')

if __name__ == '__main__':
    main()
