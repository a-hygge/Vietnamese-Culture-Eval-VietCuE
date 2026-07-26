"""Build comparison table: all measured attributes per domain + correlation with difficulty.

Output 2 tables:
  T1 — Per-domain summary stats of all features
  T2 — Spearman correlation of each feature with difficulty
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
PPL_CSV = RESULTS / 'qwen3_8b_ppl_entropy.csv'

from load_data_v2 import load_questions_df, load_unified_df

def main():
    print('Loading data...')
    df = load_unified_df()
    qdf = load_questions_df()
    n_models = df.model.nunique()
    print(f'  {len(qdf)} questions, {n_models} models')

    difficulty_per_q = (1.0 - df.groupby('qid')['score_norm'].mean()).to_dict()
    qdf['difficulty'] = qdf['qid'].map(difficulty_per_q)

    qdf['q_len']     = qdf['question'].apply(lambda q: len(re.findall(r'\w+', q, flags=re.UNICODE)))
    qdf['ref_len']   = qdf['reference'].apply(lambda r: len(re.findall(r'\w+', r, flags=re.UNICODE)))
    qdf['has_year']  = qdf['question'].apply(lambda q: int(bool(re.search(r'\b(1[89]|20)\d{2}\b', q))))
    qdf['has_number']= qdf['question'].apply(lambda q: int(bool(re.search(r'\d{2,}', q))))

    try:
        from underthesea import ner as vi_ner
        def count_ent(t):
            try:
                return sum(1 for x in vi_ner(t) if x[3].startswith('B-'))
            except Exception:
                return 0
    except ImportError:
        def count_ent(t):
            return len(re.findall(r'[A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]\w*', t))
    qdf['n_entity_q']   = qdf['question'].apply(count_ent)
    qdf['n_entity_ref'] = qdf['reference'].apply(count_ent)

    if PPL_CSV.exists():
        ppl_rows = list(csv.DictReader(open(PPL_CSV, encoding='utf-8')))
        ppl_map = {int(r['qid']): (float(r['perplexity']), float(r['mean_entropy']), int(r['n_tokens']))
                   for r in ppl_rows if r['perplexity'] not in ('', 'nan')}
        qdf['perplexity']   = qdf['qid'].map(lambda q: ppl_map.get(q, (None,))[0])
        qdf['mean_entropy'] = qdf['qid'].map(lambda q: ppl_map.get(q, (None, None))[1])
        qdf['n_gen_tokens'] = qdf['qid'].map(lambda q: ppl_map.get(q, (None, None, None))[2])
    else:
        print(f'⚠️  PPL CSV not found at {PPL_CSV}')

    print('\n' + '='*82)
    print('T1 — Per-domain mean of all measured attributes')
    print('='*82)
    feature_cols = ['difficulty', 'q_len', 'ref_len', 'n_entity_q', 'n_entity_ref',
                    'has_year', 'has_number', 'perplexity', 'mean_entropy']
    feature_cols = [c for c in feature_cols if c in qdf.columns]

    domain_summary = qdf.groupby('domain')[feature_cols].mean().round(3)
    overall = qdf[feature_cols].mean().round(3)
    overall.name = 'Overall'
    summary = pd.concat([domain_summary, overall.to_frame().T])
    summary.insert(0, 'n', qdf.groupby('domain').size().to_dict() | {'Overall': len(qdf)})
    summary['n'] = summary['n'].astype(int) if summary['n'].dtype != int else summary['n']
    print(summary.to_string())
    summary.to_csv(RESULTS / 'feature_table_per_domain.csv')
    print(f'\n  → Saved {RESULTS / "feature_table_per_domain.csv"}')

    print('\n' + '='*82)
    print('T2 — Spearman correlation of each feature with Difficulty')
    print('='*82)
    rows = []
    valid = qdf.dropna(subset=['difficulty'])
    for col in feature_cols:
        if col == 'difficulty': continue
        sub = valid.dropna(subset=[col])
        if len(sub) < 5: continue
        rho, p = spearmanr(sub[col], sub['difficulty'])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        rows.append({
            'feature':     col,
            'mean':        round(sub[col].mean(), 3),
            'std':         round(sub[col].std(), 3),
            'min':         round(sub[col].min(), 3),
            'max':         round(sub[col].max(), 3),
            'rho_vs_diff': round(rho, 4),
            'p_value':     f'{p:.4g}',
            'sig':         sig,
        })
    corr_df = pd.DataFrame(rows).sort_values('rho_vs_diff', key=lambda s: s.abs(), ascending=False)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(RESULTS / 'feature_correlation_with_difficulty.csv', index=False)
    print(f'\n  → Saved {RESULTS / "feature_correlation_with_difficulty.csv"}')

    tex = ['\\begin{table}[t]', '\\centering\\small',
           '\\caption{Per-domain mean of all measured question attributes. PPL and entropy from Qwen3-8B.}',
           '\\label{tab:feature_table}',
           '\\begin{tabular}{l' + 'r'*len(feature_cols) + '}',
           '\\toprule',
           'Domain & ' + ' & '.join('\\textbf{' + c.replace('_', '\\_') + '}' for c in feature_cols) + ' \\\\',
           '\\midrule']
    for idx, row in summary.iterrows():
        vals = ' & '.join(f'{row[c]:.2f}' if isinstance(row[c], float) else str(row[c]) for c in feature_cols)
        tex.append(f'{idx} & {vals} \\\\')
    tex += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    (RESULTS / 'feature_table_per_domain.tex').write_text('\n'.join(tex), encoding='utf-8')

if __name__ == '__main__':
    main()
