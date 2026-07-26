# VietCuE Analysis & Pipeline Code

This bundle contains the inference pipeline, LLM-as-judge, fine-tuning, and paper-specific analysis code accompanying *"Beyond Aggregate Accuracy: A Discriminator-Filtered Vietnamese Benchmark for Multi-Axis Cultural Evaluation of LLMs"*.

## Structure

```
code/
├── pipeline/              # 2-phase generate+judge pipeline (judge_models.py)
├── run_models/            # 19-model inference runner (runner.py + models_config.py)
├── judge_ablation/        # 4-prompt-strategy judge validation (judge-validation table)
├── framing/               # Framing-experiment runner (run_framing.py)
├── lm_match/              # Binary LM-match alternative judge (run_lm_match.py)
├── finetune/
│   ├── qwen3-4b/          # Full SFT Qwen3-4B (5 epochs, bf16)
│   └── qwen3-8b/          # Full SFT Qwen3-8B (5 epochs, bf16)
├── analysis/              # Paper-specific analysis scripts (CLC, Co/Ag, PPL, etc.)
├── data_generation/       # Source QA generation + auto quality-check scripts (§3)
├── README.md              # this file
└── requirements.txt
```

## Quickstart

### 1. Install deps
```bash
pip install -r requirements.txt
pip install -r finetune/requirements.txt   # if running fine-tune
```

### 2. Set up API credentials
```bash
cp pipeline/.env.example pipeline/.env
# edit .env, fill OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY / AZURE_OPENAI_*

cp run_models/.env.example run_models/.env
# multi-provider keys for the 19-model inference
```

### 3. Reproduce paper numbers
Datasets ship separately. Place the dataset bundle alongside the code:
```
<workspace>/
├── code/                  # this bundle
└── dataset/               # released dataset bundle
```

Then:
```bash
cd analysis/
python exp_c1_clc_v2.py       # § 6 CLC table + GPT-5 paradox
python exp_c2_co_ag_v2.py     # § 7.1 Co/Ag observational
python exp_c3_difficulty_v3_with_ppl.py   # § 5 + Appendix G PPL-difficulty
# … see per-script docstrings
```

All scripts use `seed=3407`.

---

## `pipeline/` — generate + judge

| File | Role |
|---|---|
| `config.py` | system prompts, `JUDGE_PROMPT` (CSAT 0–5 rubric), `TEMPERATURE`, `MAX_TOKENS`, worker counts, paths |
| `call_model.py` | unified API wrapper auto-detecting provider by name prefix (`claude-*`, `gpt-*`, `gemini-*`) |
| `judge_models.py` | main pipeline: phase-1 generate, phase-2 judge — `ThreadPoolExecutor` with auto-resume + auto-save every 50 items |
| `sample_data.json` | 10-item smoke test before running on the full corpus |
| `run.sh` | convenience wrapper |
| `.env.example` | API-key template |

## `run_models/` — 19-model inference

Runs each evaluated LLM on the 247-item test set and produces `outputs/{model_name}.json`.

| File | Role |
|---|---|
| `models_config.py` | provider routing for the 19 evaluated models |
| `runner.py` | per-model generation loop with retry / rate-limit handling |
| `run_all.py` | batch driver over all configured models |
| `compute_qwen_ppl.py` | Qwen3-8B per-token PPL on test-set gold answers (no-thinking mode, Appendix H) |
| `merge_outputs.py` | aggregate per-model JSONs into one frame |

## `judge_ablation/` — judge-prompt validation

Compares four GPT-5 judge prompting strategies against human consensus to select the judge used for the main leaderboard.

| File | Role |
|---|---|
| `judge_prompts.py` | the four prompt builders (Zero-shot, CoT, Demonstration, Aspect-based) + rubric + score parser |
| `run_judge.py` | runs one or all four strategies over the 247-item test set → `judge_outputs/{strategy}.json` |
| `build_table1_final.py` | joins judge outputs with `human_annotations/` (2-annotator consensus) to build the judge-validation table |
| `build_table2.py` | early leaderboard build over a 6-model subset (superseded by `run_models/` + `pipeline/` for the full 19-model leaderboard reported in the paper) |

## `framing/` — § 7 controlled-rewrite experiment

`run_framing.py` consumes the four framing variants (v1/v2/v3/v4) in the dataset bundle and produces per-variant judge outputs.

## `lm_match/` — Appendix E alternative judge

`run_lm_match.py` runs the binary LM-match equivalence judge for the metric-robustness check.

## `finetune/` — Appendix K

Two sub-folders, one per scale. Each carries a `main.py`, a `run.sh`, and a sub-folder README. **All configurations use 5 epochs, lr 2×10⁻⁵, bf16, gradient-checkpointing, AdamW fused, single A100 80GB.** Specifics:

- **qwen3-4b/** — Full-parameter SFT, default `--batch-size 32 --grad-acc-steps 1`
- **qwen3-8b/** — Full-parameter SFT, `per_device_train_batch_size=4`, `gradient_accumulation_steps=8` (matches 4B in effective batch)

Both produce two output JSONs (baseline + post-SFT) plus per-epoch judge scores; the released dataset bundle includes the judged outputs (`finetune_outputs/`).

## `analysis/` — paper figures and tables

Self-contained Python scripts that consume the released dataset bundle and produce the numerical results reported in the paper.

| Script | Reproduces |
|---|---|
| `exp_c1_clc_v2.py` | Cultural Lexicon Coverage table (§6) |
| `exp_c1_clc_csat_validation.py` | CLC vs. human-CSAT validation (§6) |
| `clc_csat_per_subcat.py` | Per-sub-topic CLC lifts (Appendix J) |
| `clc_bootstrap_ci.py` | Bootstrap CIs for CLC |
| `exp_c2_co_ag_v2.py` | Observational Co/Ag split (§7.1) |
| `exp_c3_difficulty_v3_with_ppl.py` | Difficulty regression with PPL (§5, Appendix G) |
| `compute_ppl_csat.py` | PPL × CSAT quintile table (Appendix H) |
| `recompute_question_chars.py` | Question characteristic features (Appendix F) |
| `build_feature_table.py` | Per-domain feature table |
| `load_data_v2.py` | Shared dataset loader (called by all scripts) |

Edit the `ROOT` constant at the top of `load_data_v2.py` if the relative layout of `code/` ↔ `dataset/` differs.

## `data_generation/` — source item generation (§3, Dataset Construction)

Scripts used to generate and quality-check the candidate pool before discriminator filtering.

| File | Role |
|---|---|
| `generate_qa_benchmark.py` | GPT-4-based QA generation from the authoritative textbook sources |
| `ban_sac_van_hoa_viet_nam.py` | generation pass specific to the "Bản sắc văn hóa Việt Nam" source |
| `data_quality_checker.py` | rule-based pre-filter for decontextualisation / assumed-knowledge / noise issues |
| `auto_validation_scorer.py` | LLM-based auto-scoring on the four quality criteria (text-based, no-temporal, relevant, objective) ahead of human annotation |

## Reproducibility

- All scripts are deterministic given `seed=3407`.
- Judge runs at `temperature=1`, `max_tokens=4000`, `model=azure/gpt-5` (or the provider you configure).
- LLM provider snapshots may shift over time; the released dataset bundle contains the exact judge outputs used in the paper so the analysis numbers can be reproduced without re-running inference.

## License

Code: MIT. Datasets retain the licences of their primary sources (see dataset bundle README).
