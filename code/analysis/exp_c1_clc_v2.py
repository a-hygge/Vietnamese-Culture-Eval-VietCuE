"""EXP C1 v2 — Cultural Lexicon Coverage analysis on NEW pipeline.

Method identical to v1 (Bhatt & Diaz 2024 §4.3, adapted):
  CLC(output, ref) = |gold_terms_in(output) ∩ gold_terms_in(ref)| / |gold_terms_in(ref)|

Differences from v1:
  - 247 questions instead of 358
  - 6 models instead of 16 (will scale to 16 when remaining models complete)
  - judge = (score >= 3) instead of binary GPT-5 judge
  - Additional: continuous correlation between CLC and ordinal score
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from load_data_v2 import load_unified_df

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
GOLD_PATH = ROOT / 'gold_cultural_terms.json'

def load_gold() -> list[str]:
    gold_dict = json.loads(GOLD_PATH.read_text(encoding='utf-8'))
    all_terms = set()
    for terms in gold_dict.values():
        all_terms.update(t.lower() for t in terms)
    return sorted(all_terms, key=len, reverse=True)

def _compile_term_regex(terms: list[str]) -> re.Pattern:
    sorted_terms = sorted(terms, key=len, reverse=True)
    escaped = [re.escape(t) for t in sorted_terms]
    pattern = r'(?<![\w])(?:' + '|'.join(escaped) + r')(?![\w])'
    return re.compile(pattern, re.IGNORECASE | re.UNICODE)

def find_terms(text: str, pat: re.Pattern) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        return set()
    return {m.group(0) for m in pat.finditer(text.lower())}

def compute_clc(df: pd.DataFrame, pat: re.Pattern) -> pd.DataFrame:
    df = df.copy()
    df['ref_terms'] = df['reference'].apply(lambda t: find_terms(t, pat))
    df['out_terms'] = df['model_answer'].apply(lambda t: find_terms(t, pat))

    def _clc(row):
        if not row['ref_terms']:
            return np.nan
        return len(row['ref_terms'] & row['out_terms']) / len(row['ref_terms'])

    df['clc'] = df.apply(_clc, axis=1)
    return df

def main():
    print('Loading data...')
    df = load_unified_df()
    print(f'  rows: {len(df)}, models: {df.model.nunique()}')

    print('Loading gold terms...')
    gold = load_gold()
    pat = _compile_term_regex(gold)
    print(f'  {len(gold)} unique gold terms')

    print('Computing CLC...')
    df = compute_clc(df, pat)
    valid = df.dropna(subset=['clc']).copy()
    print(f'  Valid (qid, model) pairs (ref has gold term): {len(valid)} / {len(df)}')

    pivot = (valid.groupby(['model', 'domain'])['clc'].mean() * 100).unstack('domain')
    if 'History' not in pivot.columns:
        pivot['History'] = np.nan
    pivot['Overall'] = (valid.groupby('model')['clc'].mean() * 100)
    pivot = pivot.sort_values('Overall', ascending=False)
    pivot = pivot[['History', 'Law', 'Culture', 'Overall']]
    pivot.to_csv(RESULTS / 'clc_pivot_v2.csv')

    latex_tab = pivot.round(1).to_latex(
        float_format='%.1f',
        caption='Cultural Lexicon Coverage (CLC, \\%): mean fraction of gold '
                'cultural terms in the reference that also appear in the model '
                'output, per (model, domain). 247 questions, 6 evaluated models.',
        label='tab:clc_v2',
    )
    (RESULTS / 'clc_pivot_v2.tex').write_text(latex_tab, encoding='utf-8')
    print(f'\nCLC pivot:')
    print(pivot.round(1))

    correct = valid[valid.judge_binary]['clc']
    incorrect = valid[~valid.judge_binary]['clc']
    mw_stat, mw_p = mannwhitneyu(correct, incorrect, alternative='greater')

    by_domain_judge = (
        valid.groupby(['domain', 'judge_binary'])['clc'].mean() * 100
    ).unstack('judge_binary')
    by_domain_judge.columns = ['incorrect_clc_pct', 'correct_clc_pct']
    by_domain_judge['gap_pp'] = (
        by_domain_judge['correct_clc_pct'] - by_domain_judge['incorrect_clc_pct'])
    by_domain_judge.to_csv(RESULTS / 'clc_by_domain_judge_v2.csv')

    rho_score, p_score = spearmanr(valid['clc'], valid['score'])

    per_q = valid.groupby('qid').agg(
        clc_mean=('clc', 'mean'),
        score_mean=('score', 'mean'),
        match_rate=('judge_binary', 'mean'),
        domain=('domain', 'first'),
    ).reset_index()

    rho_q, p_q = spearmanr(per_q['clc_mean'], per_q['match_rate'])
    rho_q_score, p_q_score = spearmanr(per_q['clc_mean'], per_q['score_mean'])

    per_q_paired = (valid.groupby(['qid', 'judge_binary'])['clc'].mean()
                    .unstack('judge_binary').dropna())
    if len(per_q_paired.columns) == 2:
        per_q_paired.columns = ['incorrect', 'correct']
        wilc_stat, wilc_p = wilcoxon(per_q_paired['correct'], per_q_paired['incorrect'])
    else:
        wilc_stat, wilc_p = None, None

    summary = {
        'pipeline_version':                'v2 (247Q ordinal 0-5 judge)',
        'n_gold_terms':                    int(len(gold)),
        'n_models_evaluated':              int(df.model.nunique()),
        'n_valid_pairs':                   int(len(valid)),
        'n_correct_outputs_score_ge_3':    int(len(correct)),
        'n_incorrect_outputs_score_lt_3':  int(len(incorrect)),
        'clc_mean_correct_pct':            float(correct.mean() * 100),
        'clc_mean_incorrect_pct':          float(incorrect.mean() * 100),
        'gap_correct_minus_incorrect_pp':  float((correct.mean() - incorrect.mean()) * 100),
        'mannwhitney_u':                   float(mw_stat),
        'mannwhitney_p_one_sided':         float(mw_p),
        'spearman_clc_vs_score_outputlevel': {'rho': float(rho_score), 'p': float(p_score)},
        'spearman_clc_vs_matchrate_perq':  {'rho': float(rho_q), 'p': float(p_q)},
        'spearman_clc_vs_meanscore_perq':  {'rho': float(rho_q_score), 'p': float(p_q_score)},
        'paired_wilcoxon': (
            {'stat': float(wilc_stat), 'p_two_sided': float(wilc_p),
             'n_paired': int(len(per_q_paired))}
            if wilc_stat is not None else None
        ),
        'by_domain': {
            d: {'correct_clc_pct':   float(row['correct_clc_pct']),
                'incorrect_clc_pct': float(row['incorrect_clc_pct']),
                'gap_pp':            float(row['gap_pp'])}
            for d, row in by_domain_judge.iterrows()
        },
    }
    (RESULTS / 'clc_summary_v2.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSummary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    domain_order = ['History', 'Law', 'Culture']
    pos = np.arange(len(domain_order))
    width = 0.35
    correct_means = [by_domain_judge.loc[d, 'correct_clc_pct'] if d in by_domain_judge.index else 0
                     for d in domain_order]
    incorr_means = [by_domain_judge.loc[d, 'incorrect_clc_pct'] if d in by_domain_judge.index else 0
                    for d in domain_order]
    axes[0].bar(pos - width / 2, correct_means, width, label='Correct (score≥3)',
                color='tab:green', alpha=0.85)
    axes[0].bar(pos + width / 2, incorr_means, width, label='Incorrect (score<3)',
                color='tab:red', alpha=0.85)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(domain_order)
    axes[0].set_ylabel('Cultural Lexicon Coverage (%)')
    axes[0].set_title('CLC: correct vs incorrect outputs, per domain')
    axes[0].legend()
    axes[0].grid(alpha=0.3, axis='y')

    domain_colors = {'History': 'tab:blue', 'Law': 'tab:orange', 'Culture': 'tab:green'}
    for dom, grp in per_q.groupby('domain'):
        axes[1].scatter(grp['clc_mean'], grp['score_mean'], label=dom,
                        alpha=0.5, s=22, color=domain_colors.get(dom, 'gray'))
    axes[1].set_xlabel('Cultural Lexicon Coverage (avg per question)')
    axes[1].set_ylabel('Mean judge score (0-5, avg per question)')
    axes[1].set_title(f'CLC vs Score (per question; ρ={rho_q_score:.2f})')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / 'figures' / 'figure_clc_v2.png', dpi=150)
    fig.savefig(RESULTS / 'figures' / 'figure_clc_v2.pdf')
    plt.close(fig)

    print(f'\n✅ Wrote results to {RESULTS}')

if __name__ == '__main__':
    main()
