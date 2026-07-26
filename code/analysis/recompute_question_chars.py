"""Recompute Table 4 (question characteristics) per the 9-subcategory schema."""
from __future__ import annotations
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent.parent  # <workspace>/ root
TEST_V3 = ROOT / 'version 02' / 'LLM-as-judge' / 'humancheck' / 'kept_items_full_v3.json'
RAW     = ROOT / 'paper_emnlp' / 'analysis_v2' / 'results' / 'difficulty_features_raw_v2.csv'
PPL     = ROOT / 'paper_emnlp' / 'analysis_v2' / 'results' / 'qwen3_8b_ppl_entropy_no_thinking.csv'

SUBCAT_EN = {
    'Xã hội':              'Society',
    'Tôn giáo':            'Religion',
    'Phong tục tập quán':  'Customs',
    'Giáo dục':            'Education',
    'Nghệ thuật':          'Arts',
    'Chính trị':           'Politics',
    'Kinh tế':             'Economy',
}
ROW_ORDER = ['Society', 'Religion', 'Customs', 'Education', 'Arts', 'Politics', 'Economy', 'History', 'Law']

def get_bucket(item):
    cat = item.get('category')
    if cat == 'History': return 'History'
    if cat == 'Law':     return 'Law'
    if cat == 'Culture':
        sc = item.get('subcategory')
        return SUBCAT_EN.get(sc, '?')
    return '?'

def main():
                           
    test = json.loads(TEST_V3.read_text(encoding='utf-8'))
    id2bucket = {it['id']: get_bucket(it) for it in test}

    df = pd.read_csv(RAW)
    df['bucket'] = df['qid'].map(id2bucket)
    df = df[df['bucket'].isin(ROW_ORDER)]

    try:
        ppl_df = pd.read_csv(PPL)
                     
        df = df.merge(ppl_df, on='qid', how='left')
        ppl_col = 'perplexity'
        ent_col = 'mean_entropy'
        print(f'  PPL col: {ppl_col}  Entropy col: {ent_col}')
    except Exception as e:
        print(f'  Could not load PPL/Entropy: {e}')
        ppl_col = None
        ent_col = None

    print(f'\nTotal items joined: {len(df)}')
    print(f'\nPer-bucket counts:')
    for b in ROW_ORDER:
        print(f'  {b:<10}: {(df["bucket"]==b).sum()}')

    print('\n\nPer-subcategory means:')
    hdr = f'  {"Bucket":<10} {"n":>4} {"Diff":>7} {"q_len":>7} {"ref_len":>8} {"ent_q":>7} {"ent_ref":>8} {"%year":>7}'
    if ppl_col: hdr += f' {"PPL":>7}'
    if ent_col: hdr += f' {"Entropy":>9}'
    print(hdr)
    print('-' * len(hdr))
    rows = []
    for b in ROW_ORDER:
        sub = df[df['bucket'] == b]
        if len(sub) == 0:
            continue
        row = {
            'bucket':       b,
            'n':            int(len(sub)),
            'difficulty':   round(sub['difficulty'].mean(), 3),
            'q_len':        round(sub['q_len'].mean(), 1),
            'ref_len':      round(sub['ref_len'].mean(), 1),
            'ent_q':        round(sub['n_entity_q'].mean(), 2),
            'ent_ref':      round(sub['n_entity_ref'].mean(), 2),
            'pct_year':     round(100 * sub['has_year'].mean(), 1),
        }
        if ppl_col and ppl_col in sub.columns:
            row['ppl']     = round(sub[ppl_col].dropna().mean(), 3)
        if ent_col and ent_col in sub.columns:
            row['entropy'] = round(sub[ent_col].dropna().mean(), 3)
        rows.append(row)
        line = f'  {b:<10} {row["n"]:>4} {row["difficulty"]:>7.3f} {row["q_len"]:>7.1f} {row["ref_len"]:>8.1f} {row["ent_q"]:>7.2f} {row["ent_ref"]:>8.2f} {row["pct_year"]:>6.1f}%'
        if 'ppl' in row:     line += f' {row["ppl"]:>7.3f}'
        if 'entropy' in row: line += f' {row["entropy"]:>9.3f}'
        print(line)

    all_row = {
        'bucket':     'Overall',
        'n':          int(len(df)),
        'difficulty': round(df['difficulty'].mean(), 3),
        'q_len':      round(df['q_len'].mean(), 1),
        'ref_len':    round(df['ref_len'].mean(), 1),
        'ent_q':      round(df['n_entity_q'].mean(), 2),
        'ent_ref':    round(df['n_entity_ref'].mean(), 2),
        'pct_year':   round(100 * df['has_year'].mean(), 1),
    }
    if ppl_col: all_row['ppl']     = round(df[ppl_col].dropna().mean(), 3)
    if ent_col: all_row['entropy'] = round(df[ent_col].dropna().mean(), 3)
    rows.append(all_row)
    line = f'  {"Overall":<10} {all_row["n"]:>4} {all_row["difficulty"]:>7.3f} {all_row["q_len"]:>7.1f} {all_row["ref_len"]:>8.1f} {all_row["ent_q"]:>7.2f} {all_row["ent_ref"]:>8.2f} {all_row["pct_year"]:>6.1f}%'
    if 'ppl' in all_row:     line += f' {all_row["ppl"]:>7.3f}'
    if 'entropy' in all_row: line += f' {all_row["entropy"]:>9.3f}'
    print(line)

    out = ROOT / 'paper_emnlp' / 'analysis_v2' / 'results' / 'question_chars_9subcat.json'
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSaved -> {out}')

if __name__ == '__main__':
    main()
