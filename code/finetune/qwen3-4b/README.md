# Full SFT — Qwen3 4B (Vietnamese Culture)

Full-parameter supervised fine-tuning of `Qwen3-4B` on the 5,081-item VietCuE training corpus (paper Appendix K).

## Setup
- **Base model:** `unsloth/Qwen3-4B` (loaded in full 16-bit bf16, no quantization)
- **Method:** Full-parameter SFT (not LoRA)
- **Library:** Unsloth + TRL `SFTTrainer`
- **Data:** `data_qa.json` (5,081 items; schema `{id, question, answer}`)
- **Hyperparameters:** 5 epochs, lr 2e-5, per-device batch 32 × grad-accumulation 1 → effective batch 32, `max_seq_length=1024`, `adamw_torch_fused`, `bf16=True`, `gradient_checkpointing=True`, `seed=3407`
- **Hardware:** GPU with ≥ 40 GB VRAM (A100 40/80 GB or H100). An RTX 3090 24 GB will OOM under full-parameter bf16 SFT.
- **Wall-time:** ≈ 90–130 minutes on a single A100 80 GB.

## Run

```bash
bash run.sh
```

Three-phase pipeline:

1. **Step 1 — Baseline inference:** Qwen3-4B (no fine-tune) on the 247-item test set → `outputs/qwen3-4b-baseline.json`
2. **Step 2 — Full SFT:** 5 epochs over the 5,081-item training corpus
3. **Step 3 — Post-SFT inference:** fine-tuned Qwen3-4B on the 247-item test set → `outputs/qwen3-4b-sft.json`

The test set is read from the released dataset bundle: `../../../dataset/VietCuE_test_247.json`. Edit the path constant in `main.py` if your layout differs.

## Outputs

Two JSON files written to `outputs/`:

- `qwen3-4b-baseline.json` (pre-fine-tune answers)
- `qwen3-4b-sft.json` (post-fine-tune answers)

The checkpoint folder `qwen3-4b-vn-culture-sft/` (≈ 8 GB) is a local artefact only; it is not required for downstream analysis.
