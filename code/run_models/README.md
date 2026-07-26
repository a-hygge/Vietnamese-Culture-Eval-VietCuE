# Multi-model inference runner — 247-item VietCuE test set

Self-contained runner for generating Vietnamese answers from the 16 evaluated LLMs on the 247-item test set.

## 1. Setup

### Python deps
```bash
pip install litellm openai python-dotenv
```
- `litellm` handles all closed-source providers (OpenAI, Anthropic, Gemini, DeepSeek, Qwen, Grok) through a unified API.
- `openai` is required separately for NVIDIA NIM (NIM uses the OpenAI SDK with a `base_url` override).

### API keys — create a `.env` file in this folder
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...           # for Qwen API
XAI_API_KEY=xai-...                # for Grok

# NVIDIA NIM — used for all open-source models (Llama, Qwen, gpt-oss, Gemma, DeepSeek, Mistral, Nemotron)
# Multi-key rotation supported — fill in multiple keys to auto-switch on rate-limit (429).
NVIDIA_API_KEY=nvapi-...
NVIDIA_API_KEY_2=nvapi-...
NVIDIA_API_KEY_3=nvapi-...
```
Fill in only the keys you have; missing keys simply skip the corresponding models.

### Open-source models via NVIDIA NIM
**No local GPU required.** All seven open-source models (Llama 3.3-70B, Qwen3-next-80B, Qwen3.5-397B, gpt-oss-20B/120B, Gemma-3-27B, DeepSeek v3.2) run through the NVIDIA NIM API at `https://integrate.api.nvidia.com/v1`.

Free-tier keys: <https://build.nvidia.com> (1000 credits/key/month).

## 2. Running

### List configured models
```bash
python runner.py --list
```

### Option 1: run a single model
```bash
python runner.py gpt-5.5 --workers 5
python runner.py claude-opus-4.7 --workers 3
python runner.py claude-sonnet-4.6 --workers 3
```

### Option 2 (recommended): batch driver for all models
```bash
# All 16 models in one command
python run_all.py --skip-done --workers 5

# Only the 9 closed-source models (GPT, Claude, Gemini, DeepSeek, Qwen)
python run_all.py --group api --workers 5 --skip-done

# Only the 7 open-source models via NVIDIA NIM
python run_all.py --group nim --workers 3 --skip-done

# Hand-pick
python run_all.py --models gpt-5.5 claude-opus-4.7 llama-3.3-70b --workers 5
```

`--skip-done`: skip any model that has already produced 247 completions (per-model resume; safe to re-run).

Tips:
- Closed-source APIs: 3–5 workers are typically fine (rate limits).
- NVIDIA NIM: 2–3 workers; the runner auto-rotates across `NVIDIA_API_KEY*` on 429.
- Interrupting mid-run is safe — re-running the same command resumes from the per-model checkpoint.
- `run_all.py` isolates per-model failures (one failure does not kill the batch); logs land in `outputs/_run_all_log_*.txt`.

### Quick sanity check before a full run
```bash
python runner.py gpt-4.1 --workers 2 --limit 5     # 5 items only, verifies the API key
```

## 3. Per-model output files

```
outputs/
├── openai__gpt-5.5.json
├── anthropic__claude-opus-4-7.json
├── gemini__gemini-3.1-pro-preview.json
├── deepseek-ai__deepseek-v3.json
└── ...
```

Use `merge_outputs.py` to consolidate the per-model JSONs into one analysis-ready frame.

## 4. Output schema (per file)

```json
{
  "metadata": {
    "model_name": "gpt-5.5",
    "full_name": "openai/gpt-5.5",
    "model_id": "gpt-5.5",
    "provider": "openai",
    "temperature": 1,
    "max_tokens": 600,
    "total_data": 247,
    "evaluated": 247,
    "errors": 0,
    "last_update": "<ISO-8601 timestamp>"
  },
  "details": [
    {
      "id": 152,
      "category": "culture",
      "question": "...",
      "gold_answer": "...",
      "answer": "<model output>",
      "error": null
    },
    ...
  ]
}
```

## 5. Models (run any that you have keys for)

| Tier | Models |
|---|---|
| Closed-source via LiteLLM (9) | `gpt-5.5`, `gpt-5.4`, `claude-opus-4.7`, `claude-sonnet-4.6-thinking`, `claude-haiku-4.5`, `gemini-3.1-pro`, `gemini-2.5-pro`, `deepseek-v3`, `qwen3-max` |
| Open-source via NVIDIA NIM (7) | `llama-3.3-70b`, `qwen3.5-397b`, `qwen3-next-80b`, `gpt-oss-120b`, `gpt-oss-20b`, `gemma-3-27b`, `deepseek-v3.2-nim` |
