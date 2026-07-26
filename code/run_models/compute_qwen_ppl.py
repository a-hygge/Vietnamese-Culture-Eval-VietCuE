"""Compute perplexity + token entropy of Qwen3-8B on 247 test questions.

Usage:
    pip install transformers torch accelerate
    python compute_qwen_ppl.py

Output:
    outputs/qwen3_8b_ppl_entropy.csv  with columns:
      qid, category, question, answer, n_tokens, perplexity, mean_entropy

The output is fed back into the difficulty-regression pipeline as a feature (EXP C3).

Hypothesis:
  - High perplexity / entropy = model uncertain about answer = question is harder
  - Low PPL / entropy = model confident = question is easier (or model has knowledge)
"""
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_NAME = 'Qwen/Qwen3-8B'
DATA_PATH = Path('data/kept_items_full.json')                           
OUT_PATH = Path('outputs/qwen3_8b_ppl_entropy_no_thinking.csv')                       
MAX_NEW_TOKENS = 256
BATCH_SIZE = 32                                                  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16                                                                      

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {'role': 'system', 'content': (
            'Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. '
            'Trả lời ngắn gọn, chính xác, bằng tiếng Việt.')},
        {'role': 'user', 'content': question},
    ]
                                                                                          
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

@torch.no_grad()
def generate_and_score_batch(model, tokenizer, questions: list[str]) -> list[tuple]:
    """Batched generation + scoring.

    Returns list of (answer, n_tokens, perplexity, mean_entropy) per question.
    Perplexity = exp(mean NLL of greedy-selected tokens).
    Mean entropy = mean of -sum(p*log p) over generated steps.
    """
                                                                             
    tokenizer.padding_side = 'left'
    prompts = [build_prompt(tokenizer, q) for q in questions]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(DEVICE)
    in_len = inputs.input_ids.shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    gen_out = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=pad_id,
    )
                                                                              
    results = []
    for b in range(len(questions)):
        gen_ids = gen_out.sequences[b, in_len:]               

        eos_hits = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
        n_tokens = int(eos_hits[0].item()) + 1 if len(eos_hits) > 0 else int(gen_ids.shape[0])

        if n_tokens == 0:
            results.append(('', 0, float('nan'), float('nan')))
            continue

        nll_sum = 0.0
        ent_sum = 0.0
        for t in range(min(n_tokens, len(gen_out.scores))):
            logits = gen_out.scores[t][b]                     
            logp = F.log_softmax(logits, dim=-1)
            p = logp.exp()
            chosen_id = int(gen_ids[t].item())
            nll_sum += -float(logp[chosen_id].item())
            ent_sum += float(-(p * logp).sum().item())

        mean_nll = nll_sum / n_tokens
        perplexity = float(torch.exp(torch.tensor(mean_nll)).item())
        mean_entropy = ent_sum / n_tokens
        answer = tokenizer.decode(gen_ids[:n_tokens], skip_special_tokens=True).strip()
        results.append((answer, n_tokens, perplexity, mean_entropy))

    return results

def save_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    print(f'Loading data from {DATA_PATH}...')
    if not DATA_PATH.exists():
                                                         
        alt = Path(r'../dataset/VietCuE_test_247.json')
        if alt.exists():
            print(f'  Using fallback path: {alt}')
            data = json.loads(alt.read_text(encoding='utf-8'))
        else:
            print(f'❌ Data not found. Place kept_items_full.json at: {DATA_PATH.absolute()}')
            sys.exit(1)
    else:
        data = json.loads(DATA_PATH.read_text(encoding='utf-8'))

    print(f'  {len(data)} questions loaded')
    print(f'\nLoading {MODEL_NAME} on {DEVICE} ({DTYPE}) batch={BATCH_SIZE}...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()

    done_qids = set()
    rows = []
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                done_qids.add(int(row['qid']))
                rows.append(row)
        print(f'Resumed: {len(done_qids)} already processed')

    pending = [item for item in data if int(item['id']) not in done_qids]
    print(f'\nGenerating + computing PPL for {len(pending)} items (batch={BATCH_SIZE})...')
    t0 = time.time()
    processed = 0

    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start: batch_start + BATCH_SIZE]
        try:
            batch_results = generate_and_score_batch(
                model, tokenizer, [item['question'] for item in batch])
        except torch.cuda.OutOfMemoryError:
            print(f'  OOM at batch {batch_start}, retrying one-by-one...')
            torch.cuda.empty_cache()
            batch_results = []
            for item in batch:
                try:
                    res = generate_and_score_batch(model, tokenizer, [item['question']])[0]
                except Exception as e:
                    print(f'    qid={item["id"]} ❌ {e}')
                    res = ('', 0, float('nan'), float('nan'))
                batch_results.append(res)
        except Exception as e:
            print(f'  Batch {batch_start} ❌ {e}')
            batch_results = [('', 0, float('nan'), float('nan'))] * len(batch)

        for item, (answer, n_tok, ppl, ent) in zip(batch, batch_results):
            rows.append({
                'qid':          int(item['id']),
                'category':     item['category'],
                'question':     item['question'].replace('\n', ' '),
                'answer':       answer.replace('\n', ' ')[:500],
                'n_tokens':     n_tok,
                'perplexity':   f'{ppl:.4f}',
                'mean_entropy': f'{ent:.4f}',
            })
        processed += len(batch)

        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(pending) - processed) / rate / 60 if rate else 0
        last = batch_results[-1]
        print(f'  [{processed}/{len(pending)}] ppl={last[2]:.2f} ent={last[3]:.2f} | '
              f'rate={rate:.2f}/s eta={eta:.1f}min')

        save_csv(OUT_PATH, rows)

    elapsed = time.time() - t0
    print(f'\n✅ Done {len(rows)}/{len(data)} in {elapsed/60:.1f} min')
    print(f'   Saved to {OUT_PATH.absolute()}')

if __name__ == '__main__':
    main()
