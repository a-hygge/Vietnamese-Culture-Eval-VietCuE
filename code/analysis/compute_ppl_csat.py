"""Compute PPL x CSAT cross-tabulation for paper Table tab:app_ppl_csat.

Two views:
(1) CSAT bin (0-5) -> mean PPL across (model, question) pairs with that CSAT
(2) PPL quintile -> mean CSAT across all (model, question) pairs in that quintile

Saves CSV + JSON; prints both tables.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent  # <workspace>/ root
ANALYSIS = ROOT / 'paper_emnlp' / 'analysis_v2'
sys.path.insert(0, str(ANALYSIS))

from load_data_v2 import load_unified_df        

PPL_CSV = ANALYSIS / 'results' / 'qwen3_8b_ppl_entropy_no_thinking.csv'
OUT_DIR = ANALYSIS / 'results'

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    df = load_unified_df()
    ppl = pd.read_csv(PPL_CSV)[['qid', 'perplexity', 'mean_entropy']]
    merged = df.merge(ppl, on='qid', how='inner')
    print(f'Merged rows: {len(merged)} (expect ~4693)')
    print(f'Unique qids merged: {merged.qid.nunique()} (expect 247)')
    print(f'Unique models: {merged.model.nunique()}')

    g = merged.groupby('score')['perplexity']
    table1 = pd.DataFrame({
        'n_pairs': g.count(),
        'mean_ppl': g.mean(),
        'std_ppl': g.std(),
        'median_ppl': g.median(),
    }).round(4)

    rho1, p1 = spearmanr(merged.score, merged.perplexity)
    print('\n=== View 1: CSAT bin -> mean PPL (across 4,693 model-item pairs) ===')
    print(table1)
    print(f'Spearman rho(score, PPL) = {rho1:.4f} (p = {p1:.4e})')

    q_ppl = merged.drop_duplicates('qid')[['qid', 'perplexity']]
    q_ppl['quintile'] = pd.qcut(q_ppl.perplexity, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    merged2 = merged.merge(q_ppl[['qid', 'quintile']], on='qid')

    g2 = merged2.groupby('quintile', observed=True)
    table2 = pd.DataFrame({
        'n_questions': g2['qid'].nunique(),
        'n_pairs': g2['score'].count(),
        'ppl_min': g2['perplexity'].min(),
        'ppl_max': g2['perplexity'].max(),
        'mean_csat': g2['score'].mean(),
        'std_csat': g2['score'].std(),
        'pct_csat_le_2': g2.apply(lambda d: float((d.score <= 2).mean() * 100), include_groups=False),
    }).round(4)

    qstat = merged.groupby('qid').agg(mean_csat=('score', 'mean'),
                                       perplexity=('perplexity', 'first')).reset_index()
    rho2, p2 = spearmanr(qstat.perplexity, qstat.mean_csat)
    print('\n=== View 2: PPL quintile (question-level) -> mean CSAT ===')
    print(table2)
    print(f'Question-level Spearman rho(PPL, mean_CSAT_19models) = {rho2:.4f} (p = {p2:.4e})')

    table1.to_csv(OUT_DIR / 'ppl_by_csat.csv')
    table2.to_csv(OUT_DIR / 'csat_by_ppl_quintile.csv')

    summary = {
        'n_pairs_merged': int(len(merged)),
        'n_qids': int(merged.qid.nunique()),
        'n_models': int(merged.model.nunique()),
        'view1_csat_to_ppl': {
            str(k): {col: (float(v) if pd.notna(v) else None) for col, v in row.items()}
            for k, row in table1.iterrows()
        },
        'spearman_pair_level': {'rho': float(rho1), 'p': float(p1), 'n_pairs': int(len(merged))},
        'view2_ppl_quintile_to_csat': {
            str(k): {col: (float(v) if pd.notna(v) else None) for col, v in row.items()}
            for k, row in table2.iterrows()
        },
        'spearman_question_level': {'rho': float(rho2), 'p': float(p2), 'n_questions': int(len(qstat))},
    }
    with open(OUT_DIR / 'ppl_csat_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: ppl_by_csat.csv, csat_by_ppl_quintile.csv, ppl_csat_summary.json')

if __name__ == '__main__':
    main()
