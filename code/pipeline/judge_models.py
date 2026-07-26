import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import config
from call_model import call_model
from dotenv import load_dotenv

_data_lock = threading.Lock()

load_dotenv()

def _parse_args():
    p = argparse.ArgumentParser(
        description="Generate model answers then CSAT-judge (concurrent API calls per phase).",
    )
    p.add_argument(
        "model",
        help="Model under test (must appear in config.ALLOWED_MODELS)",
    )
    return p.parse_args()

args = _parse_args()
MODEL = args.model
NUM_WORKERS = config.NUM_WORKERS

if MODEL not in config.ALLOWED_MODELS:
    print(f"Error: unknown model {MODEL!r}")
    print(f"Allowed models: {', '.join(config.ALLOWED_MODELS)}")
    sys.exit(1)
if config.DEFAULT_MODEL not in config.ALLOWED_MODELS:
    print(f"Error: DEFAULT_MODEL {config.DEFAULT_MODEL!r} must be in ALLOWED_MODELS")
    print(f"Allowed models: {', '.join(config.ALLOWED_MODELS)}")
    sys.exit(1)
if NUM_WORKERS < 1:
    print("Error: --workers must be >= 1")
    sys.exit(1)

ANSWER_KEY = f"{MODEL}_answer"
SCORE_KEY = f"{MODEL}_csat_score"
REASONING_KEY = f"{MODEL}_csat_reasoning"

OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, MODEL, "results.json")
_SOURCE_DATA_PATH = config.DATA_PATH

def _build_question_input(question: str, context: str | None = None) -> str:
    parts = []
    if context:
        parts.append(f"Ngữ cảnh: {context}")
    parts.append(f"Câu hỏi: {question}")
    return "\n".join(parts)

def _build_judge_input(
    question: str,
    correct_answer: str,
    model_answer: str,
    context: str | None = None,
) -> str:
    parts = []
    if context:
        parts.append(f"Ngữ cảnh: {context}")
    parts.extend([
        f"Câu hỏi: {question}",
        f"Đáp án đúng: {correct_answer}",
        f"Câu trả lời của model: {model_answer}",
        "",
        "Hãy chấm điểm câu trả lời của model theo thang CSAT 1-5 và xuất ra JSON như đã yêu cầu.",
    ])
    return "\n".join(parts)

def _parse_judge_json(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)

def _retry_call(label: str, item_id: str, fn):
    for attempt in range(config.MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            wait = config.RETRY_BASE_DELAY ** attempt
            print(f"  [{label} {item_id}] error: {e!r}, retry in {wait}s")
            time.sleep(wait)
    print(f"  [{label} {item_id}] thất bại sau {config.MAX_RETRIES} lần, trả về None")
    return None

def generate_answer(item: dict) -> dict:
    if item.get(ANSWER_KEY):
        return item
    item_id = str(item.get("id", "?"))
    user_input = _build_question_input(item["question"], item.get("context"))
    answer = _retry_call(
        "generate",
        item_id,
        lambda: call_model(
            MODEL,
            user_input,
            system_prompt=config.SYSTEM_PROMPT,
        ),
    )
    item[ANSWER_KEY] = answer
    return item

def judge_answer(item: dict) -> dict:
    if item.get(SCORE_KEY) is not None:
        return item
    item_id = str(item.get("id", "?"))
    if not item.get(ANSWER_KEY):
        item[SCORE_KEY] = None
        item[REASONING_KEY] = "Không có câu trả lời từ model"
        return item

    judge_input = _build_judge_input(
        item["question"], item["answer"], item[ANSWER_KEY], item.get("context")
    )

    def _do():
        response = call_model(
            config.DEFAULT_MODEL,
            judge_input,
            system_prompt=config.JUDGE_PROMPT,
        )
        parsed = _parse_judge_json(response)
        return int(parsed["score"]), parsed.get("reasoning", "")

    result = _retry_call("judge", item_id, _do)
    if result is None:
        item[SCORE_KEY] = None
        item[REASONING_KEY] = "Judge thất bại sau retry"
    else:
        score, reasoning = result
        item[SCORE_KEY] = score
        item[REASONING_KEY] = reasoning
    return item

def _run_phase(name: str, fn, data: list, done_check) -> list:
    todo = [(i, item) for i, item in enumerate(data) if not done_check(item)]
    print(f"\n=== Phase: {name} | model: {MODEL} | workers: {NUM_WORKERS} ===")
    print(f"  Cần xử lý: {len(todo)}/{len(data)}")
    if not todo:
        return data

    completed = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(fn, item): i for i, item in todo}
        try:
            for future in as_completed(futures):
                i = futures[future]
                data[i] = future.result()
                completed += 1
                print(f"  [{completed}/{len(todo)}] id={data[i].get('id', '?')}")
                if completed % config.SAVE_EVERY == 0:
                    _save(data)
        except BaseException:
            _save(data)
            raise
    return data

def _save(data: list) -> None:
    with _data_lock:
        snapshot = [dict(item) for item in data]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu: {OUTPUT_PATH}")

def _load() -> list:
    if os.path.exists(OUTPUT_PATH):
        print(f"Loading existing results from {OUTPUT_PATH}")
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Loading source data from {config.DATA_PATH}")
    with open(config.DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def _print_stats(data: list) -> None:
    scores = [item[SCORE_KEY] for item in data if isinstance(item.get(SCORE_KEY), int)]
    print(f"\n=== Stats for {MODEL} ===")
    print(f"  Total scored: {len(scores)}/{len(data)}")
    if not scores:
        return
    avg = sum(scores) / len(scores)
    print(f"  Average score: {avg:.2f}")
    for s in (5, 4, 3, 2, 1):
        n = scores.count(s)
        pct = n / len(scores) * 100
        print(f"  {s} điểm: {n} ({pct:.1f}%)")

def main():
    data = _load()

    data = _run_phase(
        "generate",
        generate_answer,
        data,
        done_check=lambda it: it.get(ANSWER_KEY) is not None,
    )
    _save(data)

    data = _run_phase(
        "judge",
        judge_answer,
        data,
        done_check=lambda it: it.get(SCORE_KEY) is not None,
    )
    _save(data)

    _print_stats(data)

if __name__ == "__main__":
    main()
