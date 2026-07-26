"""
MODELS dict cho test set kept_items_full.json (247 câu).

16 models đại diện cho leaderboard chính:
  - 9 close-source qua LiteLLM (GPT, Claude, Gemini, DeepSeek, Qwen)
  - 7 open-source qua NVIDIA NIM (Llama, Qwen3, gpt-oss, Gemma, DeepSeek-v3.2)

Convention đặt tên: "{provider_short}/{model_id}" — match format tonghop/model1.

3 providers:
  - 'litellm'    : close-source — dùng `litellm.completion(model="<namespace>/<id>", api_key=...)`
  - 'nvidia_nim' : open-source qua NIM — multi-key rotation NVIDIA_API_KEY* + OpenAI SDK + base_url override
  - 'vllm'       : remote vLLM server (OpenAI-compatible) — base_url + api_key đọc từ .env

API keys cần có trong .env:
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
  DEEPSEEK_API_KEY, DASHSCOPE_API_KEY (Qwen),
  NVIDIA_API_KEY, NVIDIA_API_KEY_2, ... (multi-key rotation cho NIM)
"""

GOLD_FIELD = 'answer'

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. "
    "Trả lời ngắn gọn, chính xác, bằng tiếng Việt."
)

MODELS = {
                                                                  
    'gpt-5.5': {
        'full_name': 'openai/gpt-5.5', 'provider': 'litellm',
        'model': 'openai/gpt-5.5', 'api_key_env': 'OPENAI_API_KEY',
        'max_tokens': 600,
    },
    'gpt-5.4': {
        'full_name': 'openai/gpt-5.4', 'provider': 'litellm',
        'model': 'openai/gpt-5.4', 'api_key_env': 'OPENAI_API_KEY',
        'max_tokens': 600,
    },

    'claude-opus-4.7': {
        'full_name': 'anthropic/claude-opus-4-7', 'provider': 'litellm',
        'model': 'anthropic/claude-opus-4-7', 'api_key_env': 'ANTHROPIC_API_KEY',
        'max_tokens': 600,
    },
    'claude-sonnet-4.6-thinking': {
        'full_name': 'anthropic/claude-sonnet-4-6-thinking-high', 'provider': 'litellm',
        'model': 'anthropic/claude-sonnet-4-6', 'api_key_env': 'ANTHROPIC_API_KEY',
        'max_tokens': 16000,                           
        'thinking': {'type': 'adaptive'},
        'output_config': {'effort': 'high'},
    },
    'claude-haiku-4.5': {
        'full_name': 'anthropic/claude-haiku-4-5', 'provider': 'litellm',
        'model': 'anthropic/claude-haiku-4-5', 'api_key_env': 'ANTHROPIC_API_KEY',
        'max_tokens': 600,
    },

    'gemini-3.1-pro': {
        'full_name': 'gemini/gemini-3.1-pro-preview', 'provider': 'litellm',
        'model': 'gemini/gemini-3.1-pro-preview', 'api_key_env': 'GOOGLE_API_KEY',
        'max_tokens': 600,
    },
    'gemini-2.5-pro': {
        'full_name': 'gemini/gemini-2.5-pro', 'provider': 'litellm',
        'model': 'gemini/gemini-2.5-pro', 'api_key_env': 'GOOGLE_API_KEY',
        'max_tokens': 600,
    },

    'deepseek-v3': {
        'full_name': 'deepseek-ai/deepseek-v3', 'provider': 'litellm',
        'model': 'deepseek/deepseek-chat', 'api_key_env': 'DEEPSEEK_API_KEY',
        'max_tokens': 600,
    },
    'qwen3-max': {
        'full_name': 'qwen/qwen3-max', 'provider': 'litellm',
        'model': 'dashscope/qwen-max', 'api_key_env': 'DASHSCOPE_API_KEY',
        'max_tokens': 600,
    },

    'llama-3.3-70b': {
        'full_name': 'meta/llama-3.3-70b-instruct', 'provider': 'nvidia_nim',
        'model': 'meta/llama-3.3-70b-instruct',
        'max_tokens': 600,
    },
    'qwen3.5-397b': {
        'full_name': 'qwen/qwen3.5-397b-a17b', 'provider': 'nvidia_nim',
        'model': 'qwen/qwen3.5-397b-a17b',
        'max_tokens': 600,
    },
    'qwen3-next-80b': {
        'full_name': 'qwen/qwen3-next-80b-a3b-instruct', 'provider': 'nvidia_nim',
        'model': 'qwen/qwen3-next-80b-a3b-instruct',
        'max_tokens': 600,
    },
    'gpt-oss-120b': {
        'full_name': 'openai/gpt-oss-120b', 'provider': 'nvidia_nim',
        'model': 'openai/gpt-oss-120b',
        'max_tokens': 600,
    },
    'gpt-oss-20b': {
        'full_name': 'openai/gpt-oss-20b', 'provider': 'nvidia_nim',
        'model': 'openai/gpt-oss-20b',
        'max_tokens': 600,
    },
    'gemma-3-27b': {
        'full_name': 'google/gemma-3-27b-it', 'provider': 'nvidia_nim',
        'model': 'google/gemma-3-27b-it',
        'max_tokens': 600,
    },
    'deepseek-v3.2-nim': {
        'full_name': 'deepseek-ai/deepseek-v3.2', 'provider': 'nvidia_nim',
        'model': 'deepseek-ai/deepseek-v3.2',
        'max_tokens': 600,
    },

    'qwen3-4b': {
        'full_name': 'qwen/qwen3-4b', 'provider': 'vllm',
        'model': 'Qwen/Qwen3-4B',                                               
        'base_url_env': 'VLLM_BASE_URL',
        'max_tokens': 8000,
        'enable_thinking': False,                                                      
    },
}

def list_models():
    """Print available models in tabular form."""
    print(f'\n{"name":<28s} {"provider":<12s} {"model_id":<46s}')
    print('-' * 88)
    for name, cfg in MODELS.items():
        marker = ' 🧠' if cfg.get('thinking') else ''
        print(f'{name:<28s} {cfg["provider"]:<12s} {cfg["model"]:<46s}{marker}')
    print(f'\nTotal: {len(MODELS)} models')

if __name__ == '__main__':
    list_models()
