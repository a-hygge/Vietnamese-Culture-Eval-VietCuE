"""
Exp 2 (tuqH-W3, TY2X-C2) - Part D: paired Wilcoxon (v2 vs v1, v3 vs v1) for the
3 new open-weight models, computed the same way as the existing
framing_all4_summary.json, then merged into one combined table spanning all
9 models (6 already-computed + 3 new).

Requires judge.py to have produced judge_scores/{version}__{model}_judged.json
for the 3 new models (v1_original, v2_vn_explicit, v3_tq_explicit).

Usage:
    python analyze.py
"""
import json
import sys
from pathlib import Path

from scipy.stats import wilcoxon

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
EXISTING_SUMMARY = ROOT / 'dataset' / 'framing_variants' / 'framing_summary.json'
SCORES_DIR = ROOT / 'dataset' / 'framing_variants' / 'judge_outputs'

NEW_MODELS = ['qwen3.5-397b', 'qwen3-next-80b', 'llama-3.3-70b']


def load_scores(version, model_name):
    path = SCORES_DIR / f'{version}__{model_name}_judged.json'
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding='utf-8'))
    return {d['id']: d['score'] for d in data['details']
            if d.get('score') is not None and not d.get('error')}


def paired_compare(v1_scores, v_other_scores):
    common = sorted(set(v1_scores) & set(v_other_scores))
    a = [v1_scores[i] for i in common]
    b = [v_other_scores[i] for i in common]
    if len(common) < 10 or all(x == y for x, y in zip(a, b)):
        return None
    stat, p = wilcoxon(b, a)  # b - a, i.e. other - v1
    return {
        'n': len(common),
        'mean_v1': sum(a) / len(a),
        'mean_other': sum(b) / len(b),
        'delta': sum(b) / len(b) - sum(a) / len(a),
        'p': p,
    }


def main():
    existing = json.loads(EXISTING_SUMMARY.read_text(encoding='utf-8'))
    existing_v2 = {r['model']: r for r in existing['paired_vs_v1'] if r['comparison'] == 'v2_vs_v1'}
    existing_v3 = {r['model']: r for r in existing['paired_vs_v1'] if r['comparison'] == 'v3_vs_v1'}

    new_rows = {}
    for model in NEW_MODELS:
        v1s = load_scores('v1_original', model)
        v2s = load_scores('v2_vn_explicit', model)
        v3s = load_scores('v3_tq_explicit', model)
        if not v1s:
            print(f'[!] no v1 scores yet for {model} - skipping (judge.py not done)')
            continue
        r2 = paired_compare(v1s, v2s) if v2s else None
        r3 = paired_compare(v1s, v3s) if v3s else None
        new_rows[model] = {'v2_vs_v1': r2, 'v3_vs_v1': r3}

    print(f'{"model":<20} {"status":<8} {"v2_vs_v1 delta":>15} {"p":>10}   '
          f'{"v3_vs_v1 delta":>15} {"p":>10}')
    print('-' * 90)

    combined = {}
    for model, row in existing_v2.items():
        r2, r3 = existing_v2[model], existing_v3[model]
        print(f'{model:<20} {"OLD":<8} {r2["delta"]:>15.3f} {r2["p"]:>10.3g}   '
              f'{r3["delta"]:>15.3f} {r3["p"]:>10.3g}')
        combined[model] = {
            'status': 'existing',
            'v2_vs_v1': {'delta': r2['delta'], 'p': r2['p'], 'n': r2['n']},
            'v3_vs_v1': {'delta': r3['delta'], 'p': r3['p'], 'n': r3['n']},
        }

    for model, row in new_rows.items():
        r2, r3 = row['v2_vs_v1'], row['v3_vs_v1']
        if r2:
            print(f'{model:<20} {"NEW":<8} {r2["delta"]:>15.3f} {r2["p"]:>10.3g}   '
                  f'{r3["delta"] if r3 else float("nan"):>15.3f} '
                  f'{r3["p"] if r3 else float("nan"):>10.3g}')
        combined[model] = {
            'status': 'new_this_rebuttal',
            'v2_vs_v1': r2,
            'v3_vs_v1': r3,
        }

    n_total = len(combined)
    n_null_v2 = sum(1 for r in combined.values()
                    if r.get('v2_vs_v1') and r['v2_vs_v1']['p'] >= 0.05)
    print(f'\n{n_null_v2}/{n_total} models show NO significant causal effect on v2 '
          f'(adding correct VN marker) at p<0.05.')

    out_path = HERE / 'results.json'
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
