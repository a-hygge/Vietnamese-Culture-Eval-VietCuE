"""
Full SFT — Qwen3-4B (Vietnamese Culture) — bfloat16, TRL only

Setup: HuggingFace Transformers + TRL, full fine-tuning, NO quantization, NO LoRA.
  - Load Qwen/Qwen3-4B in bf16
  - Optimizer: adamw_torch_fused
  - Train on responses only via DataCollatorForCompletionOnlyLM

VRAM estimate (Qwen3-4B full SFT, bf16):
  - Weights: 8GB
  - Gradients: 8GB
  - Adam optimizer states (fp32 m + v): ~16GB → ~32GB total
  - + Activations
  → Needs GPU >= 40GB (A100 40/80GB, H100). RTX 3090 24GB will OOM.

3 steps:
1. Baseline inference: run Qwen3-4B (before fine-tune) on 247 test questions
2. Full SFT: train on 5,081 QA pairs
3. Post-SFT inference: run fine-tuned model on 247 test questions

Installation:
    pip install torch transformers datasets trl accelerate
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL_NAME = os.environ.get('MODEL_NAME', 'Qwen/Qwen3-4B')
MODEL_SLUG = MODEL_NAME.split('/')[-1].lower()                           
os.environ.setdefault('WANDB_PROJECT', 'vn-culture')
SYSTEM_PROMPT = (
    'Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. '
    'Trả lời ngắn gọn, chính xác, bằng tiếng Việt.'
)
TEST_PATH = '../../run_models/data/kept_items_full.json'
TRAIN_PATH = 'data_qa.json'
MAX_SEQ_LENGTH = 1024

def load_model():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), (
        'GPU không hỗ trợ bfloat16. Cần A100 / H100 / RTX 30+ / 40+.'
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        trust_remote_code=True,
    )
    return model, tokenizer

INFERENCE_BATCH_SIZE = 64

@torch.no_grad()
def run_inference(model, tokenizer, items, output_path, max_new_tokens=512):
    model.eval()
    tokenizer.padding_side = 'left'
    results = []
    t0 = time.time()

    for batch_start in range(0, len(items), INFERENCE_BATCH_SIZE):
        batch = items[batch_start: batch_start + INFERENCE_BATCH_SIZE]
        prompts = [
            tokenizer.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM_PROMPT},
                 {'role': 'user',   'content': it['question']}],
                add_generation_prompt=True, tokenize=False,
            )
            for it in batch
        ]
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')
        in_len = inputs.input_ids.shape[1]
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, it in enumerate(batch):
            ans = tokenizer.decode(out[j][in_len:], skip_special_tokens=True).strip()
            results.append({
                'id':          it['id'],
                'category':    it['category'],
                'question':    it['question'],
                'gold_answer': it.get('answer', ''),
                'answer':      ans,
                'error':       None,
            })

        done = min(batch_start + INFERENCE_BATCH_SIZE, len(items))
        rate = done / (time.time() - t0)
        eta = (len(items) - done) / rate / 60
        print(f'  [{done:3d}/{len(items)}] rate={rate:.1f}/s eta={eta:.1f}min', flush=True)

    payload = {
        'metadata': {
            'model':         MODEL_NAME,
            'total_data':    len(items),
            'evaluated':     len(results),
            'system_prompt': SYSTEM_PROMPT,
        },
        'details': results,
    }
    Path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    print(f'Saved {len(results)} to {output_path}')

def build_train_dataset(tokenizer):
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f'Loaded {len(raw_data)} QA pairs for training')

    def make_row(d):
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT},
             {'role': 'user',   'content': d['question']}],
            add_generation_prompt=True, tokenize=False,
        )
        completion = d['answer'] + tokenizer.eos_token
        return {'prompt': prompt, 'completion': completion}

    dataset = Dataset.from_list([make_row(d) for d in raw_data])
    print('\n--- Sample[0] prompt ---')
    print(dataset[0]['prompt'][:400])
    return dataset

class EvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_items):
        self._model = model
        self._tokenizer = tokenizer
        self._test_items = test_items

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        print(f'\n=== Evaluating on test set after epoch {epoch} ===')
        run_inference(self._model, self._tokenizer, self._test_items,
                      f'outputs/{MODEL_SLUG}-sft-epoch{epoch}.json')
        self._model.train()
        return control

def build_trainer(model, tokenizer, dataset, batch_size, grad_acc_steps):
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            max_length=MAX_SEQ_LENGTH,
            completion_only_loss=True,                                               
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_acc_steps,
            warmup_steps=10,
            num_train_epochs=5,
            learning_rate=2e-5,
            logging_steps=10,
            optim='adamw_torch_fused',
            weight_decay=0.01,
            lr_scheduler_type='cosine',
            seed=3407,
            output_dir='outputs/checkpoints',
            save_strategy='no',
            bf16=True,
            fp16=False,
            tf32=True,                                                                  
            gradient_checkpointing=True,
            dataloader_num_workers=4,
            report_to='wandb',
            run_name=f'{MODEL_SLUG}-sft-16bit',
        ),
    )
    return trainer

def log_gpu_before():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    max_mem = round(gpu_stats.total_memory / 1024**3, 3)
    print(f'GPU = {gpu_stats.name}. Max memory = {max_mem} GB.')
    print(f'{start_mem} GB reserved before training.')
    return max_mem

def log_gpu_after(trainer_stats, max_mem):
    used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    runtime = trainer_stats.metrics['train_runtime']
    print(f'Train runtime: {runtime:.1f}s ({runtime/60:.2f} min)')
    print(f'Peak GPU memory: {used_mem} GB ({used_mem/max_mem*100:.1f}% of {max_mem} GB)')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--grad-acc-steps', type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs('outputs', exist_ok=True)

    model, tokenizer = load_model()

    test_items = json.loads(Path(TEST_PATH).read_text(encoding='utf-8'))
    print(f'Loaded {len(test_items)} test questions from {TEST_PATH}')
    baseline_path = f'outputs/{MODEL_SLUG}-baseline.json'
    if not Path(baseline_path).exists():
        run_inference(model, tokenizer, test_items, baseline_path)
    else:
        print(f'Skipping baseline inference ({baseline_path} already exists)')

    model.train()
    dataset = build_train_dataset(tokenizer)
    trainer = build_trainer(model, tokenizer, dataset, args.batch_size, args.grad_acc_steps)
    trainer.add_callback(EvalCallback(model, tokenizer, test_items))
    max_mem = log_gpu_before()
    trainer_stats = trainer.train()
    log_gpu_after(trainer_stats, max_mem)

    run_inference(model, tokenizer, test_items, f'outputs/{MODEL_SLUG}-sft.json')

    model.save_pretrained(f'{MODEL_SLUG}-vn-culture-sft')
    tokenizer.save_pretrained(f'{MODEL_SLUG}-vn-culture-sft')

if __name__ == '__main__':
    main()
