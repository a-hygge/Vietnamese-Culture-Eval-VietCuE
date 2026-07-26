"""
Runner: chạy 1 model trên kept_items_full.json (247 câu).

Mỗi model output 1 file riêng vào ./outputs/{full_name_safe}.json để tránh race
condition khi mở nhiều terminal cùng chạy nhiều model song song.
Sau khi xong hết, dùng `merge_outputs.py` gộp lại format giống tonghop/model1.

Usage:
    # List available models
    python runner.py --list

    # Run 1 model
    python runner.py gpt-5.5 --workers 5
    python runner.py claude-sonnet-4.6 --workers 3
    python runner.py qwen2.5-72b --workers 1            # Ollama: 1 worker thường đủ

    # Resume nếu bị ngắt giữa chừng — chỉ chạy lại câu chưa xong
    python runner.py gpt-5.5 --workers 5
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

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from dotenv import load_dotenv
    load_dotenv()
                                                      
    root_env = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    if os.path.exists(root_env):
        load_dotenv(root_env)
except ImportError:
    pass

from models_config import MODELS, SYSTEM_PROMPT, GOLD_FIELD, list_models

DATA_FILE = os.environ.get(
    'DATA_FILE',
    os.path.join(os.path.dirname(__file__), 'data', 'kept_items_full.json'),
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def safe_filename(s: str) -> str:
    """Slug từ full_name: 'openai/gpt-5.5' → 'openai__gpt-5.5'"""
    return re.sub(r'[^\w\.-]', '_', s).replace('/', '__')

def call_litellm(config, question):
    """Unified call cho tất cả close-source qua LiteLLM.
    Reuse pattern từ Judge-Models/close_source_model_answer.py.
    Hỗ trợ thinking mode (Claude reasoning) + Azure (api_base, api_version) qua kwargs.
    """
    from litellm import completion
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    kwargs = {
        'model': config['model'],
        'messages': messages,
        'max_tokens': config['max_tokens'],
        'api_key': os.getenv(config['api_key_env']),
    }
                                      
    if config.get('api_base_env'):
        kwargs['api_base'] = os.getenv(config['api_base_env'])
    if config.get('api_version_env'):
        kwargs['api_version'] = os.getenv(config['api_version_env'])
                           
    if config.get('thinking'):
        kwargs['thinking'] = config['thinking']
    if config.get('output_config'):
        kwargs['output_config'] = config['output_config']
    response = completion(**kwargs)
    return (response.choices[0].message.content or '').strip()

def call_ollama(config, question):
    """Local Ollama at http://localhost:11434 (legacy, không dùng nữa — thay = nvidia_nim)."""
    import openai
    client = openai.OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    response = client.chat.completions.create(
        model=config['model'],
        messages=messages,
        max_tokens=config['max_tokens'],
    )
    return (response.choices[0].message.content or '').strip()

NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'
_nvidia_keys = None
_nvidia_key_idx = 0
_nvidia_key_lock = threading.Lock()

def _load_nvidia_keys():
    global _nvidia_keys
    if _nvidia_keys is not None:
        return _nvidia_keys
    keys = [v for k, v in os.environ.items() if k.startswith('NVIDIA_API_KEY') and v]
    if not keys:
        raise RuntimeError('Không tìm thấy NVIDIA_API_KEY* trong .env')
    _nvidia_keys = keys
    print(f'  → NVIDIA NIM: loaded {len(keys)} API key(s)')
    return _nvidia_keys

def _nvidia_switch_key():
    global _nvidia_key_idx
    with _nvidia_key_lock:
        _nvidia_key_idx = (_nvidia_key_idx + 1) % len(_nvidia_keys)
        print(f'    ↻ Switched to NVIDIA key #{_nvidia_key_idx + 1}', flush=True)

def call_nvidia_nim(config, question):
    """NVIDIA NIM với multi-key rotation. Stream để tránh timeout dài."""
    import openai
    keys = _load_nvidia_keys()
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    last_err = None
                                               
    for attempt in range(max(5, len(keys) + 2)):
        client = openai.OpenAI(base_url=NVIDIA_BASE_URL, api_key=keys[_nvidia_key_idx])
        try:
            completion = client.chat.completions.create(
                model=config['model'],
                messages=messages,
                max_tokens=config['max_tokens'],
                stream=True,
            )
            result = ''
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
            return result.strip()
        except openai.RateLimitError as e:
            last_err = e
            _nvidia_switch_key()
            time.sleep(2)
        except openai.APIError as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise last_err if last_err else RuntimeError('NVIDIA NIM: tất cả retries đã fail')

def call_vllm(config, question):
    """vLLM remote server qua OpenAI SDK. base_url + (optional) api_key đọc từ .env.

    Hỗ trợ Qwen3 thinking control qua chat_template_kwargs.
    """
    import openai
    base_url = os.getenv(config['base_url_env'])
    if not base_url:
        raise RuntimeError(
            f'❌ Env var {config["base_url_env"]} chưa set — '
            f'cần thêm vào .env (vd: {config["base_url_env"]}=http://1.2.3.4:8000/v1)'
        )
    api_key_env = config.get('api_key_env')
    api_key = os.getenv(api_key_env) if api_key_env else None
    client = openai.OpenAI(api_key=api_key or 'EMPTY', base_url=base_url)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},
    ]
    kwargs = {
        'model': config['model'],
        'messages': messages,
        'max_tokens': config['max_tokens'],
    }
    if config.get('enable_thinking') is False:
        kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or '').strip()

PROVIDERS = {
    'litellm': call_litellm,
    'ollama': call_ollama,
    'nvidia_nim': call_nvidia_nim,
    'vllm': call_vllm,
}

def run_model(model_name: str, max_workers: int = 3, limit: int | None = None):
    if model_name not in MODELS:
        print(f'❌ Unknown model: {model_name}')
        list_models()
        sys.exit(1)

    config = MODELS[model_name]
    full_name = config['full_name']
    provider_fn = PROVIDERS[config['provider']]

    print(f'\n{"=" * 70}')
    print(f'Model:     {model_name}  ({full_name})')
    print(f'Provider:  {config["provider"]}')
    print(f'Model ID:  {config["model"]}')
    print(f'Settings:  max_tokens={config["max_tokens"]}, workers={max_workers}')
    print(f'{"=" * 70}')

    with open(DATA_FILE, encoding='utf-8') as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
        print(f'⚠️ DEBUG MODE — chỉ chạy {limit} câu đầu')
    total = len(data)

    output_file = os.path.join(OUTPUT_DIR, f'{safe_filename(full_name)}.json')

    done = {}
    if os.path.exists(output_file):
        with open(output_file, encoding='utf-8') as f:
            saved = json.load(f)
        done = {r['id']: r for r in saved.get('details', [])}
                                                               
        done = {k: v for k, v in done.items() if v.get('answer') and not v.get('error')}
        print(f'Resumed: {len(done)} đã xong, còn {total - len(done)}')

    remaining = [item for item in data if item['id'] not in done]
    if not remaining:
        print('✅ Đã chạy xong tất cả 247 câu rồi!')
        return output_file

    print('Testing API connection...', flush=True)
    try:
        test_q = 'Thủ đô Việt Nam là gì? Trả lời ngắn 1 câu.'
        test_a = provider_fn(config, test_q)
        print(f'✓ Test OK: {test_a[:80]}', flush=True)
    except Exception as e:
        print(f'❌ Test FAILED: {type(e).__name__}: {e}')
        sys.exit(2)

    lock = threading.Lock()
    results = list(done.values())
    processed = 0
    errors = 0
    start = time.time()

    def process_one(item):
        question = item.get('question', '')
        if not question:
            return None
        for retry in range(3):
            try:
                answer = provider_fn(config, question)
                return {
                    'id': item['id'],
                    'category': item.get('category'),
                    'question': question,
                    'gold_answer': item.get(GOLD_FIELD, ''),
                    'answer': answer,
                    'error': None,
                }
            except Exception as e:
                err = str(e)
                if '429' in err or 'rate' in err.lower():
                    time.sleep((retry + 1) * 10)
                else:
                    time.sleep(2)
                if retry == 2:
                    return {
                        'id': item['id'],
                        'category': item.get('category'),
                        'question': question,
                        'gold_answer': item.get(GOLD_FIELD, ''),
                        'answer': '',
                        'error': err[:300],
                    }

    def save():
        out = {
            'metadata': {
                'model_name': model_name,
                'full_name': full_name,
                'model_id': config['model'],
                'provider': config['provider'],
                'max_tokens': config['max_tokens'],
                'system_prompt': SYSTEM_PROMPT,
                'data_file': DATA_FILE,
                'total_data': total,
                'evaluated': len(results),
                'errors': sum(1 for r in results if r.get('error')),
                'last_update': datetime.now().isoformat(),
            },
            'details': sorted(results, key=lambda r: r['id']),
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    batch_size = max(max_workers * 5, 10)
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process_one, item): item for item in batch}
            for fut in as_completed(futures):
                r = fut.result()
                if r is None:
                    continue
                with lock:
                    results.append(r)
                    processed += 1
                    if r.get('error'):
                        errors += 1
                    if processed % 20 == 0:
                        elapsed = time.time() - start
                        rate = processed / elapsed if elapsed else 0
                        eta_min = (len(remaining) - processed) / rate / 60 if rate else 0
                        print(f'[{len(results):3d}/{total}] rate={rate:.2f}/s err={errors} eta={eta_min:.1f}min', flush=True)
        save()                            

    save()
    elapsed = time.time() - start
    print(f'\n{"=" * 70}')
    print(f'✅ DONE! {len(results)}/{total} ({errors} errors) trong {elapsed/60:.1f}min')
    print(f'📁 Saved: {output_file}')
    print(f'{"=" * 70}')

    return output_file

def main():
    parser = argparse.ArgumentParser(description='Run 1 LLM trên kept_items_full.json (247 câu)')
    parser.add_argument('model', nargs='?', help='Model name (xem `--list`)')
    parser.add_argument('--workers', type=int, default=3, help='Parallel workers (default 3)')
    parser.add_argument('--list', action='store_true', help='Liệt kê models')
    parser.add_argument('--limit', type=int, default=None, help='DEBUG: chỉ chạy N câu đầu')
    args = parser.parse_args()

    if args.list or not args.model:
        list_models()
        return

    run_model(args.model, max_workers=args.workers, limit=args.limit)

if __name__ == '__main__':
    main()
