"""EXP C2 v2 — Contextualized vs Agnostic split on NEW pipeline.

Method follows Naous et al. 2024 (CAMeL, ACL 2024 §3.2):
> "CAMeL prompts are split across two types: culturally-contextualized
>  prompts (CAMeL-Co) and culturally-agnostic prompts (CAMeL-Ag)."

We adapt to Vietnamese context using regex markers (VN_MARKERS) covering:
geography, people, dynasty, place, law-VN, culture, person.

This is "phase 1" auto-tagging only. User manual review (phase 2) → co_ag_final.csv
will be applied later for paper-final analysis.

Differences from v1:
  - 247 questions instead of 358
  - Compare mean score (0-5) and judge_binary (score>=3) between Co/Ag groups
  - 6 models so far (will scale to 16)
"""
from __future__ import annotations
import re
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from load_data_v2 import load_questions_df, load_unified_df

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

VN_MARKERS = {
    'geo': [
        r'\bở việt nam\b', r'\btại việt nam\b', r'\bcủa việt nam\b',
        r'\bnước ta\b', r'\bviệt nam\b',
    ],
    'people': [
        r'\bngười việt\b', r'\bdân tộc việt\b', r'\bngười việt nam\b',
        r'\bdân tộc kinh\b', r'\b54 dân tộc\b',
    ],
    'dynasty': [
        r'\btriều (lý|trần|lê|nguyễn|đinh|ngô|hồ|mạc|tây sơn)\b',
        r'\bnhà (lý|trần|lê|nguyễn|hồ|mạc)\b',
        r'\bvăn lang\b', r'\bâu lạc\b',
    ],
    'place': [
        r'\b(hà nội|huế|sài gòn|tp\.? ?hcm|đà nẵng|thăng long|'
        r'hải phòng|cần thơ|nha trang|hội an|đà lạt)\b',
        r'\b(sông hồng|sông cửu long|sông mã|sông đà)\b',
    ],
    'law_vn': [
        r'\bhiến pháp (19|20)\d{2}\b',
        r'\bbộ luật (dân sự|hình sự|lao động|hồng đức|tố tụng)\b',
        r'\bluật (hôn nhân|đất đai|lao động|doanh nghiệp|giáo dục|bảo hiểm)\b',
        r'\bquốc hội\b', r'\btòa án nhân dân\b',
    ],
    'culture': [
        r'\b(áo dài|áo bà ba|nón lá|khăn đóng)\b',
        r'\b(phở|bánh chưng|bánh tét|bún chả|chả cá|nem rán|bún bò huế|cao lầu)\b',
        r'\b(ca trù|quan họ|chèo|tuồng|cải lương|nhã nhạc|chầu văn|'
        r'hát xoan|đờn ca tài tử|ví giặm|bài chòi)\b',
        r'\b(đàn bầu|đàn tranh|đàn tỳ bà|đàn nhị|đàn nguyệt)\b',
        r'\b(văn miếu|quốc tử giám|hoàng thành|chùa một cột|lăng bác)\b',
        r'\b(tết nguyên đán|tết trung thu|lễ hội gióng|giỗ tổ hùng vương|đền hùng)\b',
    ],
    'person': [
        r'\b(hồ chí minh|võ nguyên giáp)\b',
        r'\b(lý thái tổ|lý công uẩn|lý thường kiệt|trần hưng đạo|trần quốc tuấn)\b',
        r'\b(lê lợi|nguyễn huệ|quang trung|hai bà trưng|bà triệu|ngô quyền)\b',
        r'\b(an dương vương|hùng vương|đinh tiên hoàng)\b',
        r'\b(trần ngọc thêm|phan ngọc)\b',
    ],
}
ALL_PATTERNS = [(cat, p) for cat, ps in VN_MARKERS.items() for p in ps]

def tag_question(q: str) -> tuple[str, list[str]]:
    q = q.lower()
    matches = []
    for cat, pat in ALL_PATTERNS:
        m = re.search(pat, q)
        if m:
            matches.append(f'{cat}:{m.group(0)}')
    return ('Co' if matches else 'Ag', matches)

def main():
    print('Loading questions + tagging...')
    qdf = load_questions_df()
    qdf['tag_auto'] = qdf['question'].apply(lambda q: tag_question(q)[0])
    qdf['markers'] = qdf['question'].apply(lambda q: ';'.join(tag_question(q)[1]))

    co_ag_auto = qdf[['qid', 'orig_id', 'domain', 'question',
                      'tag_auto', 'markers']].copy()
    co_ag_auto['tag_final'] = co_ag_auto['tag_auto']                               
    co_ag_auto['note'] = ''
    co_ag_auto.to_csv(ROOT / 'co_ag_auto_v2.csv', index=False, encoding='utf-8-sig')

    print(f'\nAuto distribution (n=247):')
    print(qdf['tag_auto'].value_counts().to_string())
    print(f'\nPer-domain auto:')
    print(qdf.groupby(['domain', 'tag_auto']).size().unstack().fillna(0).astype(int))

    print('\nLoading scores + merging...')
    df = load_unified_df()
    df = df.merge(qdf[['qid', 'tag_auto']], on='qid', how='left')

    pivot_score = df.pivot_table(
        values='score', index='model',
        columns=['domain', 'tag_auto'], aggfunc='mean'
    ).round(2)
    pivot_score.to_csv(RESULTS / 'co_ag_pivot_v2_score.csv')

    pivot_binary = (df.pivot_table(
        values='judge_binary', index='model',
        columns=['domain', 'tag_auto'], aggfunc='mean'
    ) * 100).round(1)
    pivot_binary.to_csv(RESULTS / 'co_ag_pivot_v2_binary.csv')

    print(f'\nMean score per model × (domain × tag):')
    print(pivot_score.to_string())
    print(f'\nJudge_binary (score>=3) % per model × (domain × tag):')
    print(pivot_binary.to_string())

    latex_tab = pivot_score.to_latex(
        float_format='%.2f',
        caption='Mean judge score (0-5) per model, split by domain and '
                'Contextualized (Co) vs Agnostic (Ag) tag of the question. '
                'Co/Ag follows Naous et al. (2024) CAMeL methodology, '
                'auto-tagged with VN cultural markers. 6 evaluated models.',
        label='tab:co_ag_v2',
        multicolumn_format='c',
    )
    (RESULTS / 'co_ag_pivot_v2.tex').write_text(latex_tab, encoding='utf-8')

    co_means = df[df.tag_auto == 'Co'].groupby('model')['score'].mean()
    ag_means = df[df.tag_auto == 'Ag'].groupby('model')['score'].mean()
    delta = (co_means - ag_means).sort_values()

    score_co = df[df.tag_auto == 'Co']['score']
    score_ag = df[df.tag_auto == 'Ag']['score']
    mw_stat, mw_p = mannwhitneyu(score_co, score_ag, alternative='two-sided')

    summary = {
        'pipeline_version':     'v2 (247Q, ordinal 0-5)',
        'n_questions':          int(len(qdf)),
        'n_co':                 int((qdf.tag_auto == 'Co').sum()),
        'n_ag':                 int((qdf.tag_auto == 'Ag').sum()),
        'pct_co':               float((qdf.tag_auto == 'Co').mean() * 100),
        'per_domain_distribution': {
            d: {'Co': int(((qdf.domain == d) & (qdf.tag_auto == 'Co')).sum()),
                'Ag': int(((qdf.domain == d) & (qdf.tag_auto == 'Ag')).sum())}
            for d in qdf.domain.unique()
        },
        'mean_score_co':        float(score_co.mean()),
        'mean_score_ag':        float(score_ag.mean()),
        'co_minus_ag':          float(score_co.mean() - score_ag.mean()),
        'mannwhitney_u':        float(mw_stat),
        'mannwhitney_p':        float(mw_p),
        'per_model_delta_co_minus_ag': {
            m: float(d) for m, d in delta.items()
        },
        'note': 'Phase 1 auto-tag only. Manual review pending; phase 2 '
                'will replace tag_final per user expert review.',
    }
    (RESULTS / 'co_ag_summary_v2.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSummary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['tab:red' if d < 0 else 'tab:green' for d in delta.values]
    ax.barh(range(len(delta)), delta.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(delta)))
    ax.set_yticklabels(delta.index)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Mean score(Co) − Mean score(Ag)')
    ax.set_title('Difference in judge score: Contextualized vs Agnostic prompts')
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(RESULTS / 'figures' / 'figure_co_ag_v2.png', dpi=150)
    fig.savefig(RESULTS / 'figures' / 'figure_co_ag_v2.pdf')
    plt.close(fig)

    print(f'\n✅ Wrote results to {RESULTS}')
    print(f'   Auto-tag csv: {ROOT / "co_ag_auto_v2.csv"} (USER REVIEWS THIS)')

if __name__ == '__main__':
    main()
