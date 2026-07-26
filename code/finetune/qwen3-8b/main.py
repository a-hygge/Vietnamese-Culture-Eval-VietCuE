"""
Full SFT — Qwen3 8B (Vietnamese Culture) — FULL 16-BIT, 5 EPOCHS

Setup: Unsloth + Full Fine-tuning trên 5,081 cặp QA tiếng Việt.
  - Train cả model (KHÔNG dùng LoRA — phân biệt với run LoRA 3-epoch chat trước đó)
  - Load weights ở bf16 (force, không fallback)
  - Optimizer adamw_torch_fused (không cần bitsandbytes)
  - 5 epoch (khớp cấu hình Qwen3-4B, theo Reproducibility Statement của paper)

VRAM estimate (Qwen3-8B full SFT, bf16):
  - Weights: 16GB
  - Gradients: 16GB
  - Adam optimizer states (fp32 m + v): ~32GB → ~64GB total
  - + Activations + KV cache
  → Cần GPU >= 80GB (A100 80GB, H100 80GB). A100 40GB / RTX 3090 sẽ OOM.

Nếu GPU không đủ: cân nhắc bật `gradient_checkpointing` (đã bật sẵn) + giảm batch + tăng grad accum.

3 bước:
1. Baseline inference: chạy Qwen3-8B (chưa fine-tune) trên 247 câu test
2. Full SFT: train trên 5,081 câu QA — 5 epoch
3. Post-SFT inference: chạy model fine-tuned trên 247 câu test

Installation (chạy 1 lần trước khi run script này):
    pip install --upgrade -qqq uv
    uv pip install -qqq "torch>=2.4.0" "triton>=3.0.0" numpy pillow torchvision \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    uv pip install --upgrade --no-deps "transformers==4.56.2" "tokenizers>=0.22.0,<=0.23.0" \
        "trl==0.22.2" unsloth unsloth_zoo
    # KHÔNG cần bitsandbytes (đã bỏ vì không quantize nữa)
"""

import json
import os
import time
from pathlib import Path

from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
from trl import SFTConfig, SFTTrainer

SYSTEM_PROMPT = (
    'Bạn là trợ lý AI chuyên về văn hóa, pháp luật và giáo dục Việt Nam. '
    'Trả lời ngắn gọn, chính xác, bằng tiếng Việt.'
)

TEST_PATH = '../../run_models/data/kept_items_full.json'
TRAIN_PATH = 'data_qa.json'
MAX_SEQ_LENGTH = 1024

def load_model():
    """Load Qwen3 8B full bfloat16, full SFT, KHÔNG quantization."""
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), (
        'GPU không hỗ trợ bfloat16. Cần A100 / H100.'
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='unsloth/Qwen3-8B',
        dtype=torch.bfloat16,                                        
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,                                     
        load_in_8bit=False,
        full_finetuning=True,                            
    )
    return model, tokenizer

def run_inference(model, tokenizer, items, output_path, max_new_tokens=512):
    results = []
    t0 = time.time()
    for i, it in enumerate(items):
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user',   'content': it['question']},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors='pt', return_dict=True,
        ).to('cuda')
        in_len = inputs.input_ids.shape[1]
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
        ans = tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()
        results.append({
            'id':          it['id'],
            'category':    it['category'],
            'question':    it['question'],
            'gold_answer': it.get('answer', ''),
            'answer':      ans,
            'error':       None,
        })
        if (i + 1) % 20 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(items) - i - 1) / rate / 60
            print(f'  [{i+1:3d}/{len(items)}] rate={rate:.1f}/s eta={eta:.1f}min', flush=True)
    payload = {
        'metadata': {
            'model':         'qwen3-8b',
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
    """Load 5,081 cặp QA + wrap vào chat template Qwen3 (ChatML)."""
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f'Loaded {len(raw_data)} QA pairs for training')

    dataset = Dataset.from_list([
        {
            'messages': [
                {'role': 'system',    'content': SYSTEM_PROMPT},
                {'role': 'user',      'content': d['question']},
                {'role': 'assistant', 'content': d['answer']},
            ]
        }
        for d in raw_data
    ])

    def formatting_prompts_func(examples):
        convos = examples['messages']
        texts = [
            tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
            for c in convos
        ]
        return {'text': texts}

    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print('\n--- Sample[0] ---')
    print(dataset[0]['text'][:600])
    return dataset

def build_trainer(model, tokenizer, dataset):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=4,                                           
            gradient_accumulation_steps=8,                                           
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
            gradient_checkpointing=True,                                     
            report_to='wandb',
            run_name='vn-culture/qwen3-8b-sft-16bit-3ep',
        ),
    )

    qwen3_kwargs = dict(
        instruction_part='<|im_start|>user\n',
        response_part='<|im_start|>assistant\n',
    )
    trainer = train_on_responses_only(trainer, **qwen3_kwargs)

    decoded = tokenizer.decode(trainer.train_dataset[100]['input_ids'])
    print('--- input_ids[100] ---')
    print(decoded[:400])
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

def main():
    os.makedirs('outputs', exist_ok=True)

    model, tokenizer = load_model()

    test_items = json.loads(Path(TEST_PATH).read_text(encoding='utf-8'))
    print(f'Loaded {len(test_items)} test questions from {TEST_PATH}')

    FastLanguageModel.for_inference(model)
    run_inference(model, tokenizer, test_items, 'outputs/qwen3-8b-baseline.json')

    FastLanguageModel.for_training(model)
    dataset = build_train_dataset(tokenizer)
    trainer = build_trainer(model, tokenizer, dataset)

    max_mem = log_gpu_before()
    trainer_stats = trainer.train()
    log_gpu_after(trainer_stats, max_mem)

    FastLanguageModel.for_inference(model)
    run_inference(model, tokenizer, test_items, 'outputs/qwen3-8b-sft-3ep.json')

    model.save_pretrained_merged(
        'qwen3-8b-vn-culture-sft-3ep',
        tokenizer,
        save_method='merged_16bit',
    )

if __name__ == '__main__':
    main()
