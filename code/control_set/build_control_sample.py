"""
Exp 6 - Part B: build the final control-v2 sample from the intersection of all
3 weak models' "wrong" sets (stage3_wrong_ids.json = ids where Qwen3-Next-80B
AND Llama-3.1-8B AND gpt-oss-20b all scored <=2).

Reuses the EXISTING {model}_answer fields already present in
VietCuE_train_5081.json for claude-opus-4.6 / gemini-2.5-pro / gpt-5.4 (same
fields Exp 3 used) - no new frontier-model generation needed, only re-judging
with the official aspect-based GPT-5 judge (judge_frontier.py).

Target: 250 items, stratified by the paper's 9 (sub)topics (Society, Religion,
Customs, Education, Arts, Politics, Economy, History, Law - Table 4's "Topic
(n)" breakdown), NOT just the 3 top-level categories (culture/law/history) -
user explicitly corrected this after an earlier version only stratified by
the 3 categories. Quota proportions are hardcoded from the paper's Table 2
test-247 column (Law=38, Society=69, History=72, Religion=24, Customs=14,
Education=10, Arts=9, Politics=7, Economy=4, sum=247) since VietCuE_test_247.json
itself has no subcategory field to recompute this live. If the actual
intersection has fewer than needed in some topic, takes all available in that
topic and reports the shortfall rather than failing (thầy's "250" is a
target, not a hard requirement - the whole point of this experiment is that
the intersection size is unknown until stage 1-3 actually run).

Usage:
    python build_control_sample.py
"""
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 3407  # same convention as Exp 1/3 / paper's Reproducibility Statement
TARGET_N = 250

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / 'dataset' / 'VietCuE_train_5081.json'
TEST_PATH = ROOT / 'dataset' / 'VietCuE_test_247.json'

HERE = Path(__file__).resolve().parent
WRONG_IDS_PATH = ROOT / 'dataset' / 'control_250' / 'stage3_wrong_ids.json'
OUT_PATH = ROOT / 'dataset' / 'control_250' / 'control_250.json'

MODELS = ['claude-opus-4.6', 'gemini-2.5-pro', 'gpt-5.4']

# paper's Table 2 test-247 column, per (sub)topic (Table 4's exact "Topic (n)"
# breakdown: Society, Religion, Customs, Education, Arts, Politics, Economy,
# History, Law) - hardcoded because VietCuE_test_247.json has no subcategory
# field to recompute this live (only top-level category).
TEST247_TOPIC_COUNTS = {
    'Xã hội': 69, 'Tôn giáo': 24, 'Phong tục tập quán': 14, 'Giáo dục': 10,
    'Nghệ thuật': 9, 'Chính trị': 7, 'Kinh tế': 4, 'History': 72, 'Law': 38,
}  # sums to 247


def load_json_bom(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def topic_of(it):
    return it.get('subcategory') or it['category'].capitalize()


def compute_quota(total_n: int):
    grand_total = sum(TEST247_TOPIC_COUNTS.values())
    assert grand_total == 247, f'sanity check failed: {grand_total} != 247'
    quota = {t: round(total_n * n / grand_total) for t, n in TEST247_TOPIC_COUNTS.items()}
    # fix rounding drift so quota sums exactly to total_n
    drift = total_n - sum(quota.values())
    if drift != 0:
        biggest = max(quota, key=quota.get)
        quota[biggest] += drift
    return quota


def main():
    if not WRONG_IDS_PATH.exists():
        raise SystemExit(f'{WRONG_IDS_PATH} not found - run stage 3 first (run_stage.py --stage 3).')

    wrong_ids = set(json.loads(WRONG_IDS_PATH.read_text(encoding='utf-8')))
    print(f'Intersection pool (all 3 weak models wrong): {len(wrong_ids)} items')

    train = load_json_bom(TRAIN_PATH)
    train_by_id = {it['id']: it for it in train}

    pool = [train_by_id[i] for i in wrong_ids if i in train_by_id]
    missing = len(wrong_ids) - len(pool)
    if missing:
        print(f'  [!] {missing} ids from stage3_wrong_ids.json not found in train file')

    by_topic = {}
    for it in pool:
        by_topic.setdefault(topic_of(it), []).append(it)
    print('Pool by (sub)topic:', {k: len(v) for k, v in by_topic.items()})

    n = min(TARGET_N, len(pool))
    quota = compute_quota(n)
    print(f'\nTarget n={n} (requested {TARGET_N}), quota (proportional to paper Table 2 test-247 {TEST247_TOPIC_COUNTS}): {quota}')

    rng = random.Random(SEED)
    sampled = []
    shortfall = {}
    for topic, want in quota.items():
        avail = sorted(by_topic.get(topic, []), key=lambda it: it['id'])
        take = min(want, len(avail))
        if take < want:
            shortfall[topic] = want - take
        sampled.extend(rng.sample(avail, take) if take else [])

    # top up shortfall from topics that still have spare items, to get as close to n as possible
    if shortfall:
        total_short = sum(shortfall.values())
        chosen_ids = {it['id'] for it in sampled}
        spare = []
        for topic, items in by_topic.items():
            for it in sorted(items, key=lambda it: it['id']):
                if it['id'] not in chosen_ids:
                    spare.append(it)
        rng.shuffle(spare)
        topped = spare[:total_short]
        sampled.extend(topped)
        print(f'  [!] Shortfall in {shortfall} - topped up {len(topped)} extra items from other topics')

    print(f'\nFinal sample size: {len(sampled)} (target was {TARGET_N})')
    if len(sampled) < TARGET_N:
        print(f'  [!] Could not reach {TARGET_N} - intersection pool only had {len(pool)} items total. '
              f'This is expected/acceptable: report the actual achieved n honestly.')

    records = []
    missing_answer = {m: 0 for m in MODELS}
    for it in sampled:
        rec = {
            'id': it['id'],
            'category': it['category'].lower(),
            'subcategory': it.get('subcategory'),
            'question': it['question'],
            'gold_answer': it['answer'],
        }
        for m in MODELS:
            ans = it.get(f'{m}_answer')
            if not ans:
                missing_answer[m] += 1
            rec[f'{m}_answer'] = ans
        records.append(rec)

    for m, n_missing in missing_answer.items():
        if n_missing:
            print(f'  [!] {n_missing} items missing existing {m}_answer field')

    records.sort(key=lambda r: r['id'])
    cat_counts = {}
    for r in records:
        cat_counts[r['category']] = cat_counts.get(r['category'], 0) + 1
    print(f'Final category counts in sample: {cat_counts}')

    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
