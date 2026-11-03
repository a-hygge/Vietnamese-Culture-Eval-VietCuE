"""
Judge-validation table (paper Table 3): agreement of the four GPT-5 judge
prompts with the human reference on GPT-4o's answers.

The reference is annotator B only (175 items, sample_id 73-247). The second
annotation file (human_annotations/excluded/annotator_A.xlsx) coincides on every
item with LLM reference scores that were pre-filled in the annotation sheet, so
it is not used as a human reference.

Metrics: Mean, L1 (MAE), L2 (RMSE), EM (exact match, %), <=1pt (%),
quadratic-weighted Cohen's kappa, Krippendorff's interval alpha.

Usage:  python code/judge_ablation/build_table_judge_annotatorB.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
JUDGE_DIR = ROOT / 'dataset' / 'judge_ablation_outputs'
ANNOT_B = ROOT / 'dataset' / 'human_annotations' / 'annotator_B.xlsx'
STRATEGIES = [('zero_shot', 'Zero-shot'), ('cot', 'Chain-of-thought'),
              ('demonstration', 'Few-shot'), ('aspect_based', 'Aspect-based')]


def krippendorff_interval(x, y):
    v = np.concatenate([x, y])
    n = len(v)
    expected = np.sum((v[:, None] - v[None, :]) ** 2) / (n * (n - 1))
    return 1 - np.mean((x - y) ** 2) / expected


df = pd.read_excel(ANNOT_B)
df['human_score'] = pd.to_numeric(df['human_score'], errors='coerce')
human = {int(s): float(h) for s, h in zip(df['sample_id'], df['human_score']) if pd.notna(h)}
print(f'Human reference: annotator B, {len(human)} items, mean {np.mean(list(human.values())):.2f}')

print(f'{"Prompt":18s} {"Mean":>5s} {"L1":>6s} {"L2":>6s} {"EM":>5s} {"<=1pt":>6s} {"kq":>6s} {"alpha":>6s}')
for key, name in STRATEGIES:
    details = json.loads((JUDGE_DIR / f'{key}.json').read_text(encoding='utf-8'))['details']
    judge = {r['sample_id']: r['score'] for r in details if r.get('score') is not None}
    ids = sorted(set(human) & set(judge))
    h = np.array([human[i] for i in ids])
    m = np.array([judge[i] for i in ids], dtype=float)
    print(f'{name:18s} {m.mean():5.2f} {np.mean(abs(h - m)):6.3f} {np.sqrt(np.mean((h - m) ** 2)):6.3f} '
          f'{100 * np.mean(h == m):5.1f} {100 * np.mean(abs(h - m) <= 1):6.1f} '
          f'{cohen_kappa_score(h.astype(int), m.astype(int), weights="quadratic"):6.3f} '
          f'{krippendorff_interval(h, m):6.3f}')
