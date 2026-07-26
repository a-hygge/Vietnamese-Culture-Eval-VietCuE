"""Bootstrap 95% CI for CLC gap (correct − incorrect) per domain.

Method: stratified resampling within each domain × judgment, B = 10,000 iter.
Output: results/clc_bootstrap_ci.json + .md table
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from load_data_v2 import load_unified_df

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
GOLD_PATH = ROOT / 'gold_cultural_terms.json'
B = 10_000
RNG = np.random.default_rng(42)

def load_gold():
    g = json.loads(GOLD_PATH.read_text(encoding='utf-8'))
    return sorted({t.lower() for ts in g.values() for t in ts}, key=len, reverse=True)

def make_pat(terms):
    return re.compile(r'(?<![\w])(?:' + '|'.join(re.escape(t) for t in terms) + r')(?![\w])',
                      re.IGNORECASE | re.UNICODE)

def find_terms(text, pat):
    if not isinstance(text, str) or not text.strip():
        return set()
    return {m.group(0) for m in pat.finditer(text.lower())}

def compute_clc(df, pat):
    df = df.copy()
    df['ref_terms'] = df['reference'].apply(lambda t: find_terms(t, pat))
    df['out_terms'] = df['model_answer'].apply(lambda t: find_terms(t, pat))
    df['clc'] = df.apply(
        lambda r: len(r['ref_terms'] & r['out_terms']) / len(r['ref_terms'])
        if r['ref_terms'] else np.nan,
        axis=1,
    )
    return df

def bootstrap_gap(correct, incorrect, b=B):
    """Returns (point_estimate, ci_lo, ci_hi, p_value_one_sided)."""
    correct = np.asarray(correct)
    incorrect = np.asarray(incorrect)
    n_c, n_i = len(correct), len(incorrect)
    point = correct.mean() - incorrect.mean()
    diffs = np.empty(b)
    for i in range(b):
        c_s = RNG.choice(correct, n_c, replace=True)
        i_s = RNG.choice(incorrect, n_i, replace=True)
        diffs[i] = c_s.mean() - i_s.mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_one_sided = (diffs <= 0).mean()
    return point * 100, ci_lo * 100, ci_hi * 100, p_one_sided

def main():
    print('Loading data + gold terms...')
    df = load_unified_df()
    pat = make_pat(load_gold())

    print('Computing CLC...')
    df = compute_clc(df, pat)
    valid = df.dropna(subset=['clc'])
    print(f'  Valid pairs: {len(valid)}')

    print(f'Bootstrapping B={B} per domain ...')
    rows = []
             
    c = valid[valid.judge_binary]['clc']
    i = valid[~valid.judge_binary]['clc']
    pt, lo, hi, p = bootstrap_gap(c, i)
    rows.append({
        'domain':       'Overall',
        'n_correct':    len(c),
        'n_incorrect':  len(i),
        'gap_pp':       round(pt, 2),
        'ci95_lo_pp':   round(lo, 2),
        'ci95_hi_pp':   round(hi, 2),
        'p_one_sided':  float(p),
    })
    print(f'  Overall: {pt:.2f} [{lo:.2f}, {hi:.2f}] p={p:.4g}')

    for dom in ['Law', 'History', 'Culture']:
        sub = valid[valid.domain == dom]
        c = sub[sub.judge_binary]['clc']
        i = sub[~sub.judge_binary]['clc']
        if len(c) < 2 or len(i) < 2:
            print(f'  {dom}: SKIP (n_c={len(c)} n_i={len(i)})')
            continue
        pt, lo, hi, p = bootstrap_gap(c, i)
        rows.append({
            'domain':       dom,
            'n_correct':    len(c),
            'n_incorrect':  len(i),
            'gap_pp':       round(pt, 2),
            'ci95_lo_pp':   round(lo, 2),
            'ci95_hi_pp':   round(hi, 2),
            'p_one_sided':  float(p),
        })
        print(f'  {dom}: gap={pt:.2f}pp [{lo:.2f}, {hi:.2f}], p={p:.4g}  (n_c={len(c)}, n_i={len(i)})')

    out = pd.DataFrame(rows)
    out_path = RESULTS / 'clc_bootstrap_ci.csv'
    out.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    md = ['# CLC Gap — Bootstrap 95% CI (B = 10,000)\n']
    md.append(f'Per (correct − incorrect) CLC%, stratified resample with replacement.')
    md.append('\n| Domain | n_correct | n_incorrect | Gap (p.p.) | 95% CI | p (one-sided) |')
    md.append('|---|---:|---:|---:|---:|---:|')
    for _, r in out.iterrows():
        sig = '***' if r['p_one_sided'] < 0.001 else ('**' if r['p_one_sided'] < 0.01 else ('*' if r['p_one_sided'] < 0.05 else ''))
        md.append(f"| {r['domain']} | {r['n_correct']} | {r['n_incorrect']} | "
                  f"**{r['gap_pp']:+.2f}** | [{r['ci95_lo_pp']:+.2f}, {r['ci95_hi_pp']:+.2f}] | {r['p_one_sided']:.4g} {sig} |")
    md.append('\n*** p<0.001  ** p<0.01  * p<0.05')
    md_path = RESULTS / 'clc_bootstrap_ci.md'
    md_path.write_text('\n'.join(md), encoding='utf-8')
    print(f'Saved: {md_path}')

if __name__ == '__main__':
    main()
