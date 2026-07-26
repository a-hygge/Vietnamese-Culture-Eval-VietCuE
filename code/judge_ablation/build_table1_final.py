"""
Build Table 1 cuối — format đúng paper VIVID:

  Prompt          | Mean | Cohen's Kappa
  Human Baseline  | x.xx | 1.000
  Zero-shot       | x.xx | x.xxx
  CoT             | x.xx | x.xxx
  Demonstration   | x.xx | x.xxx
  Aspect-based    | x.xx | x.xxx

Input:
  - 4 file judge_outputs/{strategy}.json (GPT-5 chấm 247 câu × 4 strategies)
  - Human scores từ annotator_A.xlsx + annotator_B.xlsx (consensus avg)

Output:
  - table1.csv / table1.md / table1.tex
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

JUDGE_DIR = r'../../dataset/judge_ablation_outputs'
ANNOT_DIR = r'../../dataset/human_annotations'
OUT_DIR = r'../../results'
os.makedirs(OUT_DIR, exist_ok=True)

STRATEGIES = ['zero_shot', 'cot', 'demonstration', 'aspect_based']
DISPLAY_NAME = {
    'zero_shot':     'Zero-shot',
    'cot':           'CoT',
    'demonstration': 'Demonstration',
    'aspect_based':  'Aspect-based',
}


def load_human_consensus():
    """Mean of A & B for overlap; else single annotator."""
    df_a = pd.read_excel(os.path.join(ANNOT_DIR, 'annotator_A.xlsx'))
    df_b = pd.read_excel(os.path.join(ANNOT_DIR, 'annotator_B.xlsx'))
    df_a['human_score'] = pd.to_numeric(df_a['human_score'], errors='coerce')
    df_b['human_score'] = pd.to_numeric(df_b['human_score'], errors='coerce')
    a_map = dict(zip(df_a['sample_id'].astype(int), df_a['human_score']))
    b_map = dict(zip(df_b['sample_id'].astype(int), df_b['human_score']))

    h = {}
    for sid in range(1, 248):
        a = a_map.get(sid)
        b = b_map.get(sid)
        if pd.notna(a) and pd.notna(b):
            h[sid] = (a + b) / 2
        elif pd.notna(a):
            h[sid] = a
        elif pd.notna(b):
            h[sid] = b
    return h


def load_strategy_scores(strategy: str):
    path = os.path.join(JUDGE_DIR, f'{strategy}.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding='utf-8'))
    scores = {r['sample_id']: r['score'] for r in d['details']
              if r.get('score') is not None}
    return scores


def main():
    HUMAN = load_human_consensus()
    print(f'Human consensus: {len(HUMAN)}/247 câu')

    rows = []
    # Row 0: Human Baseline
    h_vals = list(HUMAN.values())
    rows.append({
        'Prompt': 'Human Baseline',
        'Mean':   round(np.mean(h_vals), 2),
        "Cohen's Kappa": 1.000,
        'n_overlap': len(HUMAN),
    })

    for strategy in STRATEGIES:
        sc = load_strategy_scores(strategy)
        if sc is None:
            print(f'⚠️  {strategy}: chưa có output')
            rows.append({'Prompt': DISPLAY_NAME[strategy], 'Mean': '—',
                         "Cohen's Kappa": '—', 'n_overlap': 0})
            continue
        # Overlap with human
        common = sorted(set(HUMAN) & set(sc))
        h_arr = np.array([int(round(HUMAN[s])) for s in common])
        m_arr = np.array([sc[s] for s in common])
        mean = float(np.mean(m_arr))
        kappa = cohen_kappa_score(h_arr, m_arr, weights='quadratic')
        rows.append({
            'Prompt': DISPLAY_NAME[strategy],
            'Mean':   round(mean, 2),
            "Cohen's Kappa": round(kappa, 3),
            'n_overlap': len(common),
        })

    df = pd.DataFrame(rows)

    print(f'\n{"=" * 60}')
    print('TABLE 1 — Evaluation of LLM-as-judge prompting strategies')
    print(f'  Base model: gpt_answer (GPT-4o)')
    print(f'  Judge model: GPT-5 Azure (n=247)')
    print(f'{"=" * 60}')
    print(df.to_string(index=False))

    # === Save ===
    csv_p = os.path.join(OUT_DIR, 'table1.csv')
    df.to_csv(csv_p, index=False, encoding='utf-8-sig')

    md_lines = [
        '# Table 1 — LLM-as-judge prompting strategies',
        '',
        'Evaluation of different prompting strategies for GPT-5 judging GPT-4o outputs on 247-question Vietnamese culture benchmark. Score scale: 0–5. Cohen\'s Kappa: quadratic-weighted, vs Human consensus (avg of 2 annotators).',
        '',
        '| Prompt | Mean | Cohen\'s Kappa |',
        '|---|---:|---:|',
    ]
    for r in rows:
        kp = r["Cohen's Kappa"]
        md_lines.append(f'| {r["Prompt"]} | {r["Mean"]} | {kp} |')
    md_text = '\n'.join(md_lines)
    md_p = os.path.join(OUT_DIR, 'table1.md')
    with open(md_p, 'w', encoding='utf-8') as f:
        f.write(md_text)

    tex = r"""\begin{table}[ht]
\centering
\caption{Evaluation of different LLM-as-a-judge prompting strategies compared to human consensus, on 247 questions of the Vietnamese culture benchmark. Mean is on a 0--5 ordinal scale. Cohen's Kappa is quadratic-weighted, computed vs human consensus (averaged across 2 annotators).}
\label{tab:judge_strategies}
\begin{tabular}{lcc}
\toprule
\textbf{Prompt} & \textbf{Mean} & \textbf{Cohen's Kappa} \\
\midrule
"""
    kappa_key = "Cohen's Kappa"
    for r in rows:
        bold = r"\textbf{" if r['Prompt'] == 'Human Baseline' else ''
        end = '}' if bold else ''
        prompt_str = r['Prompt']
        mean_str = r['Mean']
        kappa_str = r[kappa_key]
        tex += f"{bold}{prompt_str}{end} & {mean_str} & {kappa_str} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    tex_p = os.path.join(OUT_DIR, 'table1.tex')
    with open(tex_p, 'w', encoding='utf-8') as f:
        f.write(tex)

    print(f'\n📁 Saved:')
    print(f'  - {csv_p}')
    print(f'  - {md_p}')
    print(f'  - {tex_p}')


if __name__ == '__main__':
    main()
