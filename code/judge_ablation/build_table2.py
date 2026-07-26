"""
Build Table 2 — leaderboard 6 models theo aspect-based judge (GPT-5 Azure).

Input:  data/judge_outputs_models/{model}_judged.json
Output: results/table2.csv, table2.md, table2.tex
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

JUDGE_DIR = r'../../dataset/judge_scores'
OUT_DIR = r'../../results'
os.makedirs(OUT_DIR, exist_ok=True)

# Pretty display name + ordering by family for Table 2
DISPLAY_NAMES = {
    'claude-opus-4.6':                   'Claude Opus 4.6',
    'gemini-3.1-pro-preview':            'Gemini 3.1 Pro (preview)',
    'gemini-2.5-pro':                    'Gemini 2.5 Pro',
    'qwen_qwen3-next-80b-a3b-instruct':  'Qwen3-Next 80B (A3B)',
    'gpt-oss-120b':                      'GPT-OSS 120B',
    'gpt-oss-20b':                       'GPT-OSS 20B',
}


def main():
    rows = []
    for f in sorted(os.listdir(JUDGE_DIR)):
        if not f.endswith('_judged.json'):
            continue
        d = json.load(open(os.path.join(JUDGE_DIR, f), encoding='utf-8'))
        model_id = d['metadata']['model']
        details = d['details']

        scores = [r['score'] for r in details
                  if r.get('score') is not None and not r.get('error')]
        n_valid = len(scores)
        n_total = len(details)
        n_err = n_total - n_valid

        # Per-category mean
        cat_scores = defaultdict(list)
        for r in details:
            if r.get('score') is not None and not r.get('error'):
                cat_scores[r['category']].append(r['score'])

        rows.append({
            'model':           model_id,
            'display':         DISPLAY_NAMES.get(model_id, model_id),
            'n':               n_valid,
            'n_err':           n_err,
            'mean':            np.mean(scores) if scores else 0.0,
            'std':             np.std(scores) if scores else 0.0,
            'culture_mean':    np.mean(cat_scores['culture']) if cat_scores['culture'] else 0.0,
            'culture_n':       len(cat_scores['culture']),
            'law_mean':        np.mean(cat_scores['law']) if cat_scores['law'] else 0.0,
            'law_n':           len(cat_scores['law']),
            'history_mean':    np.mean(cat_scores['history']) if cat_scores['history'] else 0.0,
            'history_n':       len(cat_scores['history']),
        })

    # Sort by mean desc
    rows.sort(key=lambda r: r['mean'], reverse=True)

    # === CSV ===
    csv_path = os.path.join(OUT_DIR, 'table2.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('rank,model,display,n,n_err,mean,std,culture_mean,culture_n,law_mean,law_n,history_mean,history_n\n')
        for i, r in enumerate(rows, start=1):
            f.write(f'{i},{r["model"]},{r["display"]},{r["n"]},{r["n_err"]},'
                    f'{r["mean"]:.3f},{r["std"]:.3f},'
                    f'{r["culture_mean"]:.3f},{r["culture_n"]},'
                    f'{r["law_mean"]:.3f},{r["law_n"]},'
                    f'{r["history_mean"]:.3f},{r["history_n"]}\n')

    # === Markdown ===
    md_path = os.path.join(OUT_DIR, 'table2.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Table 2 — Vietnamese Culture Benchmark Leaderboard\n\n')
        f.write(f'**Judge**: GPT-5 (Azure) + aspect-based prompting (κ = 0.742 vs human).\n')
        f.write(f'**Test set**: 247 low-CSAT questions (180 culture / 36 law / 31 history).\n')
        f.write(f'**Score**: 0–5 ordinal scale.\n\n')
        f.write('## Overall Leaderboard\n\n')
        f.write('| Rank | Model | Mean ↑ | Std | n |\n')
        f.write('|-----:|:------|------:|----:|--:|\n')
        for i, r in enumerate(rows, start=1):
            f.write(f'| {i} | {r["display"]} | **{r["mean"]:.3f}** | {r["std"]:.3f} | {r["n"]} |\n')
        f.write('\n## Per-Category Breakdown\n\n')
        f.write('| Model | Culture (n=180) | Law (n=36) | History (n=31) |\n')
        f.write('|:------|-----------------:|-----------:|---------------:|\n')
        for r in rows:
            f.write(f'| {r["display"]} | {r["culture_mean"]:.3f} | {r["law_mean"]:.3f} | {r["history_mean"]:.3f} |\n')

    # === LaTeX ===
    tex_path = os.path.join(OUT_DIR, 'table2.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write('\\caption{Vietnamese Culture Benchmark leaderboard. ')
        f.write('Aspect-based GPT-5 judge ($\\kappa = 0.742$ vs human). ')
        f.write('Scores on 0--5 ordinal scale over 247 low-CSAT questions.}\n')
        f.write('\\label{tab:leaderboard}\n')
        f.write('\\small\n')
        f.write('\\begin{tabular}{rlcccc}\n\\toprule\n')
        f.write('Rank & Model & Overall & Culture & Law & History \\\\\n\\midrule\n')
        for i, r in enumerate(rows, start=1):
            f.write(f'{i} & {r["display"]} & '
                    f'\\textbf{{{r["mean"]:.2f}}} & '
                    f'{r["culture_mean"]:.2f} & '
                    f'{r["law_mean"]:.2f} & '
                    f'{r["history_mean"]:.2f} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # === Console print ===
    print(f'\n{"=" * 90}')
    print(f'{"Rank":<5s} {"Model":<35s} {"Mean":>6s} {"Std":>5s} {"n":>4s} {"Cul":>6s} {"Law":>6s} {"His":>6s}')
    print('-' * 90)
    for i, r in enumerate(rows, start=1):
        print(f'{i:<5d} {r["display"]:<35s} {r["mean"]:>6.3f} {r["std"]:>5.3f} {r["n"]:>4d} '
              f'{r["culture_mean"]:>6.3f} {r["law_mean"]:>6.3f} {r["history_mean"]:>6.3f}')
    print(f'\n📁 Saved:\n  {csv_path}\n  {md_path}\n  {tex_path}')


if __name__ == '__main__':
    main()
