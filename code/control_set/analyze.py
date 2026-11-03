"""
Exp 6 - Part D: report control_v2 results in the paper's Table 4 format
(CSAT overall + breakdown by topic/subtopic, Accuracy = CSAT>2), for the 3
models with existing answers (Claude Opus 4.6, Gemini 2.5 Pro, GPT-5.4), and
compare against the same 3 models' test-247 numbers (already-judged, from
dataset/judge_scores/).

Requires (in order): run_stage.py --stage 1/2/3, build_control_sample.py,
judge_frontier.py, then this script.

Usage:
    python analyze.py
"""
import json
import random
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 3407  # same convention as Exp 1/3 / paper's Reproducibility Statement
N_BOOTSTRAP = 2000

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'dataset'

SAMPLE_PATH = ROOT / 'dataset' / 'control_250' / 'control_250.json'
SCORES_PATH = ROOT / 'dataset' / 'control_250' / 'control_250_scores.json'

# NOTE: the filter model GPT-5.4 has no leaderboard-judge score on test-247.
# 'gpt-5.4-pro' is a different model (GPT-5.4 Pro), so its test-247 column is
# printed for reference only; the paper reports '---' for GPT-5.4 on test-247.
MODEL_TO_TEST_FILE = {
    'claude-opus-4.6': 'claude-opus-4.6',
    'gemini-2.5-pro': 'gemini-2.5-pro',
    'gpt-5.4': 'gpt-5.4-pro',
}

# topic label used for rows: Culture items use `subcategory` (Vietnamese, as
# stored in VietCuE_train_5081.json); Law/History items have none, so the
# category name itself is the row.
TOPIC_ORDER = ['Xã hội', 'Tôn giáo', 'Phong tục tập quán', 'Giáo dục',
               'Nghệ thuật', 'Chính trị', 'Kinh tế', 'History', 'Law']


def load_json_bom(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def topic_of(rec):
    if rec.get('subcategory'):
        return rec['subcategory']
    return rec['category'].capitalize()


def load_control():
    if not SAMPLE_PATH.exists():
        raise SystemExit(f'{SAMPLE_PATH} not found - run build_control_sample.py first.')
    if not SCORES_PATH.exists():
        raise SystemExit(f'{SCORES_PATH} not found - run judge_frontier.py first.')

    sample = json.loads(SAMPLE_PATH.read_text(encoding='utf-8'))
    sample_by_id = {rec['id']: rec for rec in sample}

    scored = json.loads(SCORES_PATH.read_text(encoding='utf-8'))
    by_model = {m: {} for m in MODEL_TO_TEST_FILE}
    for d in scored['details']:
        if d.get('score') is None or d.get('error'):
            continue
        by_model[d['model_name']][d['id']] = d['score']

    out = {}
    for m in MODEL_TO_TEST_FILE:
        rows = []
        for item_id, score in by_model[m].items():
            rec = sample_by_id.get(item_id)
            if rec is None:
                continue
            rows.append({'id': item_id, 'topic': topic_of(rec), 'score': score})
        out[m] = rows
    return out


def load_test247():
    test = load_json_bom(DATASET / 'VietCuE_test_247.json')
    cat_by_id = {it['id']: it['category'].capitalize() for it in test['items']}

    out = {}
    for m, test_stem in MODEL_TO_TEST_FILE.items():
        judged = load_json_bom(DATASET / 'judge_scores' / f'{test_stem}_judged.json')
        rows = []
        for d in judged['details']:
            if d.get('score') is None or d.get('error'):
                continue
            cat = cat_by_id.get(d['id'])
            if cat is None:
                continue
            rows.append({'id': d['id'], 'topic': cat, 'score': d['score']})
        out[m] = rows
    return out


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, seed=SEED):
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return lo, hi


def summarize(rows, with_ci=False):
    if not rows:
        return None
    scores = [r['score'] for r in rows]
    acc_flags = [1 if s > 2 else 0 for s in scores]
    acc = sum(acc_flags) / len(scores)
    out = {'n': len(scores), 'mean_csat': statistics.mean(scores), 'accuracy': acc}
    if with_ci:
        out['mean_csat_ci95'] = list(bootstrap_ci(scores))
        out['accuracy_ci95'] = list(bootstrap_ci(acc_flags))
    return out


def print_table(title, by_model_rows, topics_present, with_ci=False):
    print(f'\n{"=" * 100}\n{title}\n{"=" * 100}')
    header = f'{"Topic (n)":<22}'
    for m in MODEL_TO_TEST_FILE:
        header += f'{m:>20}'
    print(header)
    print('-' * 100)
    for topic in topics_present:
        line = None
        cells = []
        for m in MODEL_TO_TEST_FILE:
            rows = [r for r in by_model_rows[m] if r['topic'] == topic]
            s = summarize(rows)
            cells.append(s)
        n = next((c['n'] for c in cells if c), 0)
        line = f'{topic + f" ({n})":<22}'
        for c in cells:
            line += f'{c["mean_csat"]:>13.2f}/{c["accuracy"]*100:>4.0f}%' if c else f'{"n/a":>20}'
        print(line)
    print('-' * 100)
    overall = []
    line = f'{"Overall":<22}'
    for m in MODEL_TO_TEST_FILE:
        s = summarize(by_model_rows[m], with_ci=with_ci)
        overall.append(s)
        line += f'{s["mean_csat"]:>13.2f}/{s["accuracy"]*100:>4.0f}%' if s else f'{"n/a":>20}'
    print(line)
    if with_ci:
        for m, s in zip(MODEL_TO_TEST_FILE, overall):
            lo, hi = s['mean_csat_ci95']
            alo, ahi = s['accuracy_ci95']
            print(f'  {m:<18} 95% CI mean_csat=[{lo:.3f}, {hi:.3f}]  accuracy=[{alo*100:.1f}%, {ahi*100:.1f}%]  (bootstrap n={N_BOOTSTRAP})')
    return overall


def main():
    control = load_control()
    test247 = load_test247()

    print('control_v2 sample size per model:', {m: len(v) for m, v in control.items()})
    print('test-247 scored size per model:', {m: len(v) for m, v in test247.items()})

    control_topics = sorted({r['topic'] for rows in control.values() for r in rows},
                             key=lambda t: TOPIC_ORDER.index(t) if t in TOPIC_ORDER else 99)
    test_topics = sorted({r['topic'] for rows in test247.values() for r in rows},
                          key=lambda t: TOPIC_ORDER.index(t) if t in TOPIC_ORDER else 99)

    test_overall = print_table('test-247 (discriminator-filtered) - CSAT (mean) / Accuracy (CSAT>2)',
                                test247, test_topics)
    control_overall = print_table('control-v2 (weak-model-filtered, non-frontier-filtered) - CSAT (mean) / Accuracy (CSAT>2)',
                                   control, control_topics, with_ci=True)

    print(f'\n{"=" * 100}\nSummary: test-247 vs control-v2 (mean CSAT / Accuracy)\n{"=" * 100}')
    results = {}
    for i, m in enumerate(MODEL_TO_TEST_FILE):
        t, c = test_overall[i], control_overall[i]
        print(f'{m:<18} test-247: {t["mean_csat"]:.3f} ({t["accuracy"]*100:.1f}%, n={t["n"]})   '
              f'control-v2: {c["mean_csat"]:.3f} ({c["accuracy"]*100:.1f}%, n={c["n"]})')
        results[m] = {'test247': t, 'control_v2': c}

    test_scores = [r['mean_csat'] for r in test_overall]
    control_scores = [r['mean_csat'] for r in control_overall]
    test_spread = max(test_scores) - min(test_scores)
    control_spread = max(control_scores) - min(control_scores)
    print(f'\nBest-worst spread (mean CSAT): test-247={test_spread:.3f}, control-v2={control_spread:.3f}')
    results['_leaderboard_spread'] = {'test247': test_spread, 'control_v2': control_spread}

    out_path = HERE / 'results.json'
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
