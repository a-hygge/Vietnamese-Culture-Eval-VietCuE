"""
Cultural Lexicon Coverage (CLC) with and without question-echo terms.

CLC(output, gold) = |T(output) & T(gold)| / |T(gold)|, where T(t) is the set of
lexicon terms found in text t; a model's CLC is the mean over items whose gold
contains at least one lexicon term (104 of the 247 test items).

Many gold answers restate entities from the question, so the "non-echo" variant
drops gold terms that also occur in the question and keeps only items with at
least one remaining term (35 items). Reproduces the CLC table of the paper.

Usage:  python code/analysis/clc_question_echo.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'dataset'


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


lexicon = load(DATASET / 'cultural_lexicon_166.json')
terms = sorted({t.lower() for v in lexicon.values() for t in v}, key=len, reverse=True)
pattern = re.compile(r'(?<![\w])(?:' + '|'.join(map(re.escape, terms)) + r')(?![\w])',
                     re.IGNORECASE | re.UNICODE)


def find_terms(text):
    if not isinstance(text, str) or not text.strip():
        return set()
    return {m.group(0) for m in pattern.finditer(text.lower())}


items = {it['id']: it for it in load(DATASET / 'VietCuE_test_247.json')['items']}

rows = []
for f in sorted((DATASET / 'model_outputs').glob('*.json')):
    full, noecho = [], []
    for r in load(f)['details']:
        it = items.get(r['id'])
        if it is None:
            continue
        gold = find_terms(it['gold_answer'])
        out = find_terms(r.get('model_answer'))
        if gold:
            full.append(len(gold & out) / len(gold))
        gold_noecho = gold - find_terms(it['question'])
        if gold_noecho:
            noecho.append(len(gold_noecho & out) / len(gold_noecho))
    rows.append((f.stem, 100 * statistics.mean(full), len(full),
                 100 * statistics.mean(noecho), len(noecho)))

n_gold = sum(1 for it in items.values() if find_terms(it['gold_answer']))
n_echo = sum(1 for it in items.values()
             if find_terms(it['gold_answer'])
             and not find_terms(it['gold_answer']) - find_terms(it['question']))
print(f'items with a lexicon term in the gold: {n_gold}; all gold terms already in the question: {n_echo}')

rank_noecho = {m: i + 1 for i, (m, *_) in enumerate(sorted(rows, key=lambda r: -r[3]))}
print(f'{"model":40s} {"CLC all":>8s} {"CLC non-echo":>13s} {"rank":>5s}')
for m, a, n, b, n2 in sorted(rows, key=lambda r: -r[1]):
    print(f'{m:40s} {a:7.1f}% {b:12.1f}% {rank_noecho[m]:5d}   (n={n}, n_non-echo={n2})')
