"""Re-check CLC-vs-CSAT monotonicity per 9-subcategory schema on the train split."""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent.parent  # <workspace>/ root
TRAIN = ROOT / 'data_train_v3.json'
GOLD  = ROOT / 'paper_emnlp' / 'analysis_v2' / 'gold_cultural_terms.json'

sys.path.insert(0, str(ROOT / 'paper_emnlp' / 'analysis_v2'))
from exp_c1_clc_v2 import _compile_term_regex, find_terms

SUBCAT_EN = {
    'Xã hội':              'Society',
    'Tôn giáo':            'Religion',
    'Phong tục tập quán':  'Customs',
    'Giáo dục':            'Education',
    'Nghệ thuật':          'Arts',
    'Chính trị':           'Politics',
    'Kinh tế':             'Economy',
}
ORDER = ['Society', 'Religion', 'Customs', 'Education', 'Arts', 'Politics', 'Economy', 'History', 'Law']

def get_bucket(item):
    cat = item.get('category')
    if cat == 'History': return 'History'
    if cat == 'Law':     return 'Law'
    if cat == 'Culture':
        sc = item.get('subcategory')
        return SUBCAT_EN.get(sc, '?')
    return '?'

def main():
    train = json.loads(TRAIN.read_text(encoding='utf-8'))
    gold = json.loads(GOLD.read_text(encoding='utf-8'))
    all_terms = sorted({t.lower() for terms in gold.values() for t in terms},
                       key=len, reverse=True)
    pat = _compile_term_regex(all_terms)

    rows = []
    MODELS = [('claude-opus-4.6', 'Claude'), ('gemini-2.5-pro', 'Gemini'), ('gpt-5.4', 'GPT-5.4')]
    for it in train:
        ref_terms = find_terms(it.get('answer', ''), pat)
        if not ref_terms: continue
        bucket = get_bucket(it)
        for mkey, _ in MODELS:
            ans   = it.get(f'{mkey}_answer', '')
            csat  = it.get(f'{mkey}_csat_score')
            if not ans or csat is None: continue
            out_terms = find_terms(ans, pat)
            clc = len(ref_terms & out_terms) / len(ref_terms)
            rows.append({'bucket': bucket, 'csat': int(csat), 'clc': clc})

    df = pd.DataFrame(rows)
    print(f'Total pairs: {len(df)}')

    print('\nMean CLC at CSAT=1 vs CSAT=5 per sub-topic:')
    print(f'  {"Sub-topic":<12} {"n":>5} {"CLC|CSAT=1":>12} {"CLC|CSAT=5":>12} {"diff":>7} {"monotone?":>11}')
    print('-' * 70)
    all_monotone = True
    for b in ORDER:
        sub = df[df['bucket'] == b]
        if len(sub) == 0: continue
        if (sub.csat == 1).sum() == 0 or (sub.csat == 5).sum() == 0:
            print(f'  {b:<12} {len(sub):>5}  insufficient cells')
            continue
        c1 = sub[sub.csat == 1].clc.mean()
        c5 = sub[sub.csat == 5].clc.mean()
                                      
        means = [sub[sub.csat == k].clc.mean() if (sub.csat == k).sum() > 0 else None for k in [1,2,3,4,5]]
        valid_means = [m for m in means if m is not None]
        mono = all(valid_means[i] <= valid_means[i+1] for i in range(len(valid_means)-1))
        mono_lite = c1 < c5
        if not mono_lite:
            all_monotone = False
        print(f'  {b:<12} {len(sub):>5} {c1:>12.3f} {c5:>12.3f} {c5-c1:>+7.3f} {"YES" if mono_lite else "NO":>11}')

    print(f'\nAll sub-topics show CLC|CSAT=5 > CLC|CSAT=1: {all_monotone}')

if __name__ == '__main__':
    main()
