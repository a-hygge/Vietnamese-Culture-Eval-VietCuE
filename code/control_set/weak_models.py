"""
Exp 6 (thầy's 4th additional request) - config for the 3 "weak" open-source
models used to build the non-frontier-filtered control subset v2.

Per thầy's exact instruction: "dùng 3 open-source cỡ nhỏ (có thể dùng Qwen,
LLama và GPT OSS), chạy trên tập training của mình để chọn ra 250 samples mà
cả 3 cái open-source model trả lời sai".

All 3 already reachable via the NVIDIA NIM infra used for Exp 2 (same
NVIDIA_API_KEY1/2/3 - these are account-level keys, not model-locked, so
reusing them for different models is fine). gpt-oss-20b and llama-3.1-8b were
already run on test-247 for the paper's original 19-model leaderboard
(code/run_models/models_config.py); qwen2.5-7b-instruct is
new (needs pilot verification of the model-id string).

STAGE_ORDER defines the sequential AND-filter order (run_stage.py): model 1
runs on the full train-5081, model 2 only on model-1's "wrong" subset, model 3
only on the model-1 ∩ model-2 "wrong" subset. Order doesn't affect the final
intersection, only cost (whichever order narrows fastest is cheaper) - picked
in the order thầy listed them since we have no prior fail-rate data to
optimize on.

NOTE on Qwen: NVIDIA NIM's public catalog no longer hosts a genuinely small
Qwen model (checked live via client.models.list() - only Qwen3.5-397B,
Qwen3.5-122B, Qwen3-Next-80B remain). Per user's explicit decision, using
Qwen3-Next-80B (the smallest Qwen still available, same model already used in
Exp2) as the "Qwen" slot instead of a small Qwen2.5-7B that no longer exists
on this API.
"""

NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. "
    "Trả lời ngắn gọn, chính xác, bằng tiếng Việt."
)
MAX_TOKENS = 600  # match code/run_models/models_config.py (same config used for the paper's own NIM model answers)

# name -> (NIM model-id, dedicated API-key env var)
MODELS = {
    'qwen3-next-80b': ('qwen/qwen3-next-80b-a3b-instruct', 'NVIDIA_API_KEY2'),  # reuse Exp2's key/model - smallest Qwen left on NIM
    'llama-3.1-8b': ('meta/llama-3.1-8b-instruct', 'NVIDIA_API_KEY_LLAMA31_8B'),
    'gpt-oss-20b': ('openai/gpt-oss-20b', 'NVIDIA_API_KEY_GPTOSS_20B'),
}

STAGE_ORDER = ['qwen3-next-80b', 'llama-3.1-8b', 'gpt-oss-20b']

WRONG_THRESHOLD = 2  # score <= 2 counts as "trả lời sai" (matches paper's Accuracy = CSAT>2)
