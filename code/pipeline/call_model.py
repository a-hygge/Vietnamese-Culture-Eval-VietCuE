import os
import threading
from dotenv import load_dotenv

import config
from openai import OpenAI

load_dotenv(override=True)

_local = threading.local()

def _thread_client() -> OpenAI:
    client = getattr(_local, "client", None)
    if client is None:
        _local.client = OpenAI(
            base_url=os.getenv("LLM_PROXY_ENDPOINT"),
            api_key=os.getenv("LLM_PROXY_API_KEY"),
        )
    return _local.client

def call_model(
    model_name: str,
    prompt: str,
    system_prompt: str | None = config.SYSTEM_PROMPT,
    max_tokens: int = config.MAX_TOKENS,
) -> str:
    client = _thread_client()

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    demo_models = [
        "claude-sonnet-4-6",
        "gpt-5.2",
        "gemini-3-flash-preview",
    ]
    demo_prompt = "Xin chào, bạn là ai?"
    demo_system = "Bạn là một trợ lý AI thân thiện, trả lời ngắn gọn bằng tiếng Việt."

    for m in demo_models:
        print(f"\n=== {m} ===")
        print(call_model(m, demo_prompt, system_prompt=demo_system))
