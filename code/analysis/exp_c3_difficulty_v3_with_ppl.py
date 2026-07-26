"""EXP C3 v3 — Question Difficulty Predictors with PPL + entropy features.

Updates v3:
  - Add 2 new features: perplexity, mean_entropy (from Qwen3-8B)
  - Re-run regression: compare R² (surface-only) vs R² (surface + PPL + entropy)
  - Spearman correlation of PPL/entropy with difficulty
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
PPL_CSV = RESULTS / 'qwen3_8b_ppl_entropy_no_thinking.csv'

from load_data_v2 import load_questions_df, load_unified_df

def main():
    print('Loading data...')
    df = load_unified_df()
    qdf = load_questions_df()
    n_models = df.model.nunique()
    print(f'  {len(qdf)} questions, {n_models} models')

    difficulty_per_q = (1.0 - df.groupby('qid')['score_norm'].mean()).to_dict()
    qdf['difficulty'] = qdf['qid'].map(difficulty_per_q)

    print(f'Loading PPL from {PPL_CSV}...')
    ppl_rows = list(csv.DictReader(open(PPL_CSV, encoding='utf-8')))
    print(f'  {len(ppl_rows)} PPL entries')

    ppl_map = {int(r['qid']): {
        'perplexity':   float(r['perplexity']),
        'mean_entropy': float(r['mean_entropy']),
        'n_tokens':     int(r['n_tokens']),
    } for r in ppl_rows if r['perplexity'] not in ('', 'nan')}

    qdf['perplexity']   = qdf['qid'].map(lambda q: ppl_map.get(q, {}).get('perplexity'))
    qdf['mean_entropy'] = qdf['qid'].map(lambda q: ppl_map.get(q, {}).get('mean_entropy'))

    import re
    qdf['q_len']   = qdf['question'].apply(lambda q: len(re.findall(r'\w+', q, flags=re.UNICODE)))
    qdf['ref_len'] = qdf['reference'].apply(lambda r: len(re.findall(r'\w+', r, flags=re.UNICODE)))
    qdf['has_year']   = qdf['question'].apply(lambda q: int(bool(re.search(r'\b(1[89]|20)\d{2}\b', q))))
    qdf['has_number'] = qdf['question'].apply(lambda q: int(bool(re.search(r'\d{2,}', q))))

    feats = qdf.dropna(subset=['difficulty', 'perplexity', 'mean_entropy']).copy()
    print(f'  Questions with valid difficulty + PPL: {len(feats)}')

    print(f'\n=== Spearman correlation with difficulty ===')
    rho_results = {}
    for feat in ['q_len', 'ref_len', 'has_year', 'has_number', 'perplexity', 'mean_entropy']:
        rho, p = spearmanr(feats[feat], feats['difficulty'])
        rho_results[feat] = (rho, p)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        print(f'  {feat:<15}  rho={rho:+.4f}  p={p:.4g}  {sig}')

    print(f'\n=== Linear regression R² comparison ===')

    surface_cols = ['q_len', 'ref_len', 'has_year', 'has_number']
    surface_plus_domain = surface_cols + ['domain_History', 'domain_Law']

    X_surf = pd.get_dummies(feats[surface_cols + ['domain']], columns=['domain'], drop_first=True).astype(float)
    y = feats['difficulty'].values
    X_scaled = StandardScaler().fit_transform(X_surf)
    r2_surf = LinearRegression().fit(X_scaled, y).score(X_scaled, y)
    print(f'  Surface only (4 features + domain):           R² = {r2_surf:.4f}')

    X_ppl = pd.get_dummies(
        feats[surface_cols + ['perplexity', 'domain']],
        columns=['domain'], drop_first=True,
    ).astype(float)
    X_scaled = StandardScaler().fit_transform(X_ppl)
    r2_ppl = LinearRegression().fit(X_scaled, y).score(X_scaled, y)
    print(f'  Surface + PPL:                                R² = {r2_ppl:.4f}  (Δ = +{r2_ppl-r2_surf:.4f})')

    X_ent = pd.get_dummies(
        feats[surface_cols + ['mean_entropy', 'domain']],
        columns=['domain'], drop_first=True,
    ).astype(float)
    X_scaled = StandardScaler().fit_transform(X_ent)
    r2_ent = LinearRegression().fit(X_scaled, y).score(X_scaled, y)
    print(f'  Surface + entropy:                            R² = {r2_ent:.4f}  (Δ = +{r2_ent-r2_surf:.4f})')

    X_full = pd.get_dummies(
        feats[surface_cols + ['perplexity', 'mean_entropy', 'domain']],
        columns=['domain'], drop_first=True,
    ).astype(float)
    X_scaled = StandardScaler().fit_transform(X_full)
    model = LinearRegression().fit(X_scaled, y)
    r2_full = model.score(X_scaled, y)
    print(f'  Surface + PPL + entropy (FULL):               R² = {r2_full:.4f}  (Δ = +{r2_full-r2_surf:.4f})')

    coef_df = pd.DataFrame({
        'feature': X_full.columns,
        'standardized_coef': model.coef_,
    }).sort_values('standardized_coef', key=lambda s: s.abs(), ascending=False)
    print(f'\n  Standardized coefficients (full model):')
    for _, row in coef_df.iterrows():
        print(f'    {row["feature"]:<25}  {row["standardized_coef"]:+.4f}')

    print(f'\n=== PPL/Entropy distribution ===')
    print(f'  PPL:     min={feats["perplexity"].min():.3f}  median={feats["perplexity"].median():.3f}  max={feats["perplexity"].max():.3f}  mean={feats["perplexity"].mean():.3f}')
    print(f'  Entropy: min={feats["mean_entropy"].min():.3f}  median={feats["mean_entropy"].median():.3f}  max={feats["mean_entropy"].max():.3f}  mean={feats["mean_entropy"].mean():.3f}')

    summary = {
        'n_questions':        int(len(feats)),
        'n_models':           int(n_models),
        'reference_model':    'Qwen3-8B',
        'spearman': {
            feat: {'rho': float(rho), 'p': float(p)}
            for feat, (rho, p) in rho_results.items()
        },
        'r2_comparison': {
            'surface_only':        float(r2_surf),
            'surface_plus_ppl':    float(r2_ppl),
            'surface_plus_entropy': float(r2_ent),
            'surface_plus_both':   float(r2_full),
            'lift_from_ppl_entropy': float(r2_full - r2_surf),
        },
        'top_predictors_full_model': coef_df.head(5).to_dict(orient='records'),
    }
    out = RESULTS / 'difficulty_summary_v3_with_ppl.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n✅ Saved {out}')

if __name__ == '__main__':
    main()
