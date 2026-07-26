"""Unified data loader v2 — VN Culture Benchmark NEW pipeline.

Test set: 247 low-CSAT questions (kept_items_full.json).
Score: 0-5 ordinal (aspect-based GPT-5 judge).
Models: 6 evaluated so far; will scale to 16 when remaining models complete.

Returns long-format DataFrame: 247 × N_models = (1482 if N=6).

Columns:
    qid, orig_id, domain (History/Law/Culture),
    question, reference (gold), context (None for v2),
    model, model_answer, score (0-5 int), judge_binary (= score >= 3),
    score_norm (= score / 5 in [0,1])
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # <workspace>/ root
KEPT_PATH = ROOT / 'version 02' / 'LLM-as-judge' / 'humancheck' / 'kept_items_full.json'
VIET_DIR = ROOT / 'table1_workspace' / 'data' / 'viet_outputs_247'
JUDGE_DIR = ROOT / 'table1_workspace' / 'data' / 'judge_outputs_models'

MODEL_DISPLAY_V2 = {
               
    'claude-opus-4.6':                          'Claude Opus 4.6',
    'claude-sonnet-4.6':                        'Claude Sonnet 4.6',
                
    'openai_gpt-5':                             'GPT-5',
    'gpt-5.4-pro':                              'GPT-5.4 Pro',
    'gpt-5.4-mini':                             'GPT-5.4 Mini',
    'gpt-5.4-nano':                             'GPT-5.4 Nano',
    'gpt-4o':                                   'GPT-4o',
    'gpt-4o-mini':                              'GPT-4o Mini',
            
    'gemini-3.1-pro-preview':                   'Gemini 3.1 Pro',
    'gemini-2.5-pro':                           'Gemini 2.5 Pro',
    'gemini-2.5-flash':                         'Gemini 2.5 Flash',
              
    'deepseek-r1':                              'DeepSeek-R1',
          
    'qwen_qwen3-next-80b-a3b-instruct':         'Qwen3-Next 80B',
    'qwen_qwen3.5-397b-a17b':                   'Qwen3.5 397B',
          
    'meta_llama-3.3-70b-instruct':              'Llama 3.3 70B',
    'llama-3-2-90b-vision':                     'Llama 3.2 90B Vision',
    'llama-3-1-8b':                             'Llama 3.1 8B',
                        
    'gpt-oss-120b':                             'GPT-OSS 120B',
    'gpt-oss-20b':                              'GPT-OSS 20B',
}

CAT_TO_DOMAIN = {
    'culture': 'Culture',
    'law':     'Law',
    'history': 'History',
}

JUDGE_BINARY_THRESHOLD = 3                                                                  

def load_questions() -> Dict[int, Dict]:
    """Returns {qid_int: {id, category, question, answer (gold), gpt_answer}} for 247 items.

    qid_int is the original id from kept_items_full.json.
    """
    items = json.loads(KEPT_PATH.read_text(encoding='utf-8'))
    return {int(it['id']): it for it in items}

def load_model_answers(model_id: str) -> Dict[int, str]:
    """Load {qid: model_answer} for a given model from viet_outputs_247/."""
    path = VIET_DIR / f'{model_id}.json'
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding='utf-8'))
    return {int(r['id']): r.get('model_answer', '') for r in d['details']}

def load_model_scores(model_id: str) -> Dict[int, int]:
    """Load {qid: score_0_to_5} for a given model from judge_outputs_models/."""
    path = JUDGE_DIR / f'{model_id}_judged.json'
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding='utf-8'))
    out = {}
    for r in d['details']:
        s = r.get('score')
        if s is None or r.get('error'):
            continue
        out[int(r['id'])] = int(s)
    return out

def load_unified_df(models: List[str] = None) -> pd.DataFrame:
    """
    Returns long-format DataFrame: 247 × N_models rows.

    Args:
        models: list of model_id keys (default = all keys in MODEL_DISPLAY_V2)
    """
    if models is None:
        models = list(MODEL_DISPLAY_V2.keys())

    questions = load_questions()
    rows = []
    for model_id in models:
        display = MODEL_DISPLAY_V2.get(model_id, model_id)
        answers = load_model_answers(model_id)
        scores = load_model_scores(model_id)
        for qid_int, item in questions.items():
            score = scores.get(qid_int)
            if score is None:
                continue                       
            rows.append({
                'qid':          qid_int,                                     
                'orig_id':      qid_int,
                'domain':       CAT_TO_DOMAIN[item['category']],
                'question':     item['question'],
                'reference':    item['answer'],
                'context':      '',
                'model':        display,
                'model_id':     model_id,
                'model_answer': answers.get(qid_int, ''),
                'score':        score,                           
                'score_norm':   score / 5.0,                    
                'judge_binary': score >= JUDGE_BINARY_THRESHOLD,
            })
    return pd.DataFrame(rows)

def load_questions_df() -> pd.DataFrame:
    """Returns one row per question (247). Use for feature extraction in C3."""
    qs = load_questions()
    rows = []
    for qid, it in qs.items():
        rows.append({
            'qid':       qid,
            'orig_id':   qid,
            'domain':    CAT_TO_DOMAIN[it['category']],
            'question':  it['question'],
            'reference': it['answer'],
            'context':   '',
        })
    return pd.DataFrame(rows)

EVALUATED_MODELS_V2 = list(MODEL_DISPLAY_V2.values())

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    df = load_unified_df()
    qdf = load_questions_df()
    print(f'Unified DataFrame: {len(df)} rows')
    print(f'Questions: {len(qdf)} (expect 247)')
    n_models = df.model.nunique()
    print(f'Models: {n_models}')
    print(f'Per-question rows: {len(df) // len(qdf)} (expect {n_models})')
    print(f'Domains: {df.domain.value_counts().to_dict()}')
    print(f'\nModel × judge_binary rate:')
    print(df.groupby('model')['judge_binary'].agg(['mean', 'count']).round(3))
    print(f'\nMean score per model:')
    print(df.groupby('model')['score'].mean().round(3).sort_values(ascending=False))
