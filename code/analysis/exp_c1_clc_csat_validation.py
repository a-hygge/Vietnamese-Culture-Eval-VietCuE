"""EXP C1 validation — CLC vs CSAT correlation on FULL train (5,081 items).

Goal: validate that CLC metric correlates with human-judged answer quality (CSAT).

Procedure:
  - 5,081 train items each have answers from 3 models (claude-opus, gemini-2.5-pro, gpt-5.4)
  - Each (item, model) has a CSAT score (1–5) from human/auto grading
  - Compute CLC for each (item, model) using existing gold cultural terms
  - Group by CSAT bin → mean CLC
  - Expected: monotone increase (low CSAT ↔ low CLC, high CSAT ↔ high CLC)

This complements the main CLC analysis (which splits by judge correctness on 247 test set).
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent.parent  # <workspace>/ root
TRAIN = ROOT / 'data_train.json'
GOLD  = ROOT / 'paper_emnlp' / 'analysis_v2' / 'gold_cultural_terms.json'

sys.path.insert(0, str(Path(__file__).parent))
from exp_c1_clc_v2 import _compile_term_regex, find_terms

def main():
    print(f'Loading {TRAIN.name}...')
    train = json.loads(TRAIN.read_text(encoding='utf-8'))
    print(f'  {len(train)} items')

    print(f'Loading gold cultural terms...')
    gold = json.loads(GOLD.read_text(encoding='utf-8'))
    all_terms = sorted({t.lower() for terms in gold.values() for t in terms},
                       key=len, reverse=True)
    pat = _compile_term_regex(all_terms)
    print(f'  {len(all_terms)} unique gold terms')

    print(f'\nComputing CLC for each (item, model)...')
    rows = []
    MODELS = [('claude-opus-4.6', 'Claude Opus 4.6'),
              ('gemini-2.5-pro', 'Gemini 2.5 Pro'),
              ('gpt-5.4', 'GPT-5.4')]
    for it in train:
        ref_terms = find_terms(it.get('answer', ''), pat)
        if not ref_terms:
            continue                                       
        for mkey, mname in MODELS:
            ans   = it.get(f'{mkey}_answer', '')
            csat  = it.get(f'{mkey}_csat_score')
            if not ans or csat is None:
                continue
            out_terms = find_terms(ans, pat)
            clc = len(ref_terms & out_terms) / len(ref_terms)
            rows.append({
                'id': it['id'], 'category': it['category'],
                'model': mname, 'csat': int(csat), 'clc': clc,
                'n_ref_terms': len(ref_terms), 'n_out_terms': len(out_terms),
            })

    df = pd.DataFrame(rows)
    print(f'  total (item, model) pairs with ref_terms > 0: {len(df)}')
    print(f'  CSAT distribution: {dict(df.csat.value_counts().sort_index())}')

    print(f'\n=== Mean CLC by CSAT score ===\n')
    print(f'  {"CSAT":>4} {"n":>5} {"mean CLC":>10} {"std":>8} {"% CLC=0":>9}')
    print('─' * 50)
    for csat in sorted(df.csat.unique()):
        sub = df[df.csat == csat]
        zero_pct = (sub.clc == 0).mean() * 100
        print(f'  {csat:>4} {len(sub):>5} {sub.clc.mean():>10.4f} {sub.clc.std():>8.4f} {zero_pct:>8.1f}%')

    print(f'\n=== Mean CLC by CSAT × model ===\n')
    pivot = df.pivot_table(values='clc', index='csat', columns='model', aggfunc='mean').round(4)
    print(pivot.to_string())

    rho, p = spearmanr(df['csat'], df['clc'])
    tau, p_tau = kendalltau(df['csat'], df['clc'])
    print(f'\n=== Item-level rank correlation (CSAT ↔ CLC) ===')
    print(f'  Spearman ρ = {rho:+.4f}  p = {p:.4g}  (n={len(df)})')
    print(f'  Kendall τ  = {tau:+.4f}  p = {p_tau:.4g}')

    print(f'\n=== Mean CLC by CSAT × domain ===\n')
    pivot_d = df.pivot_table(values='clc', index='csat', columns='category', aggfunc='mean').round(4)
    print(pivot_d.to_string())

    out = ROOT / 'paper_emnlp' / 'analysis_v2' / 'results' / 'clc_csat_validation.json'
    summary = {
        'n_pairs': int(len(df)),
        'csat_means': {int(c): float(df[df.csat==c].clc.mean()) for c in sorted(df.csat.unique())},
        'csat_n':     {int(c): int(len(df[df.csat==c])) for c in sorted(df.csat.unique())},
        'spearman_rho': float(rho), 'spearman_p': float(p),
        'kendall_tau':  float(tau), 'kendall_p':  float(p_tau),
    }
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n✓ Saved {out}')

if __name__ == '__main__':
    main()
