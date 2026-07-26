# VietCuE Dataset

This bundle is the dataset release accompanying the paper *"Beyond Aggregate Accuracy: A Discriminator-Filtered Vietnamese Benchmark for Multi-Axis Cultural Evaluation of LLMs"*.

## Contents

```
dataset/
├── VietCuE_test_247.json         # 247-item test set (discriminator-filtered)
├── VietCuE_train_5081.json       # 5,081-item training corpus
├── cultural_lexicon_166.json     # 166-term cultural lexicon, partitioned by sub-topic
├── model_outputs/                # 19 model raw answers on the 247-item test set
├── judge_scores/                 # 19 aspect-based GPT-5 judge scores
├── judge_ablation_outputs/       # 4 judge-prompt-strategy scores (Table: judge validation)
├── framing_variants/             # 3 framing-rewrite variants (v1–v3) + judged outputs
│   ├── judge_outputs/            #   per-item judged scores, 3 variants x 6 models
│   └── framing_summary.json      #   paired-Wilcoxon summary (all comparisons, all 6 models)
├── lm_match_outputs/             # Binary LM-match judge scores (metric-robustness check)
├── finetune_outputs/             # Per-epoch Qwen3-4B / 8B SFT judge scores
├── human_annotations/            # Anonymised annotator scores + Cohen's κ report
└── README.md                     # this file
```

## Files

### Test set — `VietCuE_test_247.json`
Top-level keys: `dataset`, `description`, `n_items`, `fields`, `items`.
Each `items[i]`:
- `id` (int): stable item identifier (matches across `model_outputs/` and `judge_scores/`)
- `category` (str): one of `Society`, `Religion`, `Customs`, `Education`, `Arts`, `Politics`, `Economy`, `History`, `Law`
- `question` (str): Vietnamese question text
- `gold_answer` (str): Vietnamese reference answer

### Training corpus — `VietCuE_train_5081.json`
Same field schema as the test set; additionally carries pre-test CSAT scores from three reference models (`claude-opus-4.6`, `gemini-2.5-pro`, `gpt-5.4`) used for the independent CLC validation in §6.

### Cultural lexicon — `cultural_lexicon_166.json`
JSON object with three top-level keys: `History`, `Law`, `Culture` (sub-topic partitioning into the 9 leaderboard sub-topics is given in Appendix L of the paper). After de-duplication across the three lists, the lexicon contains **166 unique Vietnamese cultural terms**.

### `model_outputs/`
Nineteen files, one per evaluated model. Each has top-level `metadata` + `details`. Each `details[i]`:
- `id` (int): matches test-set item id
- `question`, `gold_answer` (str): repeated for convenience
- `model_answer` (str): the model's free-form Vietnamese answer
- `category` (str): sub-topic

Naming convention: `<provider>_<model>.json` where `<provider>` is e.g. `openai`, `claude`, `gemini`, `qwen`. The full 19-model name table is in Table 4 of the paper.

### `judge_scores/`
Nineteen files mirroring `model_outputs/`. Each `details[i]`:
- `id` (int): matches test-set item id
- `score` (int 0–5): aspect-based GPT-5 judge score
- `raw_response` (str): full judge response with three sub-aspect scores
- `category`, `error` (optional)

The judge was run with `model=azure/gpt-5`, `temperature=1`, `max_tokens=4000`. Inter-judge stability is documented in Appendix D of the paper.

### `judge_ablation_outputs/`
Four files: `zero_shot.json`, `cot.json`, `demonstration.json`, `aspect_based.json` — GPT-5 judge scores on the same 247 items under each of the four prompting strategies compared in the judge-validation table. Reproduces that table together with `human_annotations/` (see `code/judge_ablation/build_table1_final.py`).

### `framing_variants/`
Three files: `v1_original.json`, `v2_vn_explicit.json`, `v3_tq_explicit.json`. Each is a 247-item parallel set with the variant-specific rewrite of the question prompt; `gold_answer` is unchanged across variants.

Variant semantics:
- `v1`: original question (no marker added)
- `v2`: prepend "`ở Việt Nam`" (correct country marker)
- `v3`: prepend "`ở Trung Quốc`" (wrong-country marker; tests routing vs. passive ignoring)

`judge_outputs/` holds the per-item aspect-based GPT-5 judged answers for all three variants across six models (`claude-opus-4.6`, `claude-sonnet-4.6`, `deepseek-r1`, `gemini-3.1-pro-preview`, `gpt-5`, `gpt-5.4-pro`); file naming is `{version}__{model}_judged.json`. `framing_summary.json` is the pre-computed paired-Wilcoxon summary (v2/v3 vs.\ v1, plus v3-vs-v2, with per-domain breakdowns) for all six models — the paper's causal-rewrite table (§7.2) reports three of these six models.

### `lm_match_outputs/`
Binary LM-match judge results used for the aspect-based-vs.-binary metric-robustness check (Appendix: Metric Robustness). `csat_vs_lmmatch.json`: per-model accuracy under both metrics, rank comparison, Spearman/Pearson correlation. `csat_lmmatch_per_item.json`: aggregate agreement matrix and partial-credit breakdown across the 4,693 (model, question) pairs.

### `finetune_outputs/`
Judge scores for the per-epoch Qwen3-4B / 8B SFT checkpoints (ep1–ep5) plus their respective baselines. Used to reproduce Table 8b/c (Appendix K).

### `human_annotations/`
- `annotator_A.xlsx`, `annotator_B.xlsx`: per-item Vietnamese-linguistics-trained scores on the 247-item test set
- `overlap_A_vs_B.csv`: 103-item overlap subset used to compute Cohen's κ
- `kappa_report.txt`: quadratic-weighted κ = 0.748 (Substantial), plus linear and unweighted variants

Annotator identifiers are anonymised as `A`/`B`; raw demographics are not released.

## Reproducibility

- `category` and `id` fields are stable across all files; join on `id` to align model outputs, judge scores, and gold answers.
- Numerical results reported in the paper (CLC, leaderboard CSAT, framing Δ, fine-tune lift) can be reproduced from these artefacts using the released analysis code (separate bundle).
- All scripts use `seed=3407`.

## Source attribution

The 247 test items and 5,081 training items derive from:
1. **Vietnamese National High School Graduation Examinations** and high-school semester exams (publicly available).
2. **GPT-4-generated questions grounded in five authoritative Vietnamese textbooks** on history, culture, and law (textbook list and author permissions documented in Appendix A of the paper).

## Licence

Released under CC BY-NC-SA 4.0 for non-commercial research use (see [LICENSE](LICENSE)), with the standardised-examination items inheriting the licence of their primary sources. Foreign-reference terms in the cultural lexicon (e.g.\ `Trung Quốc`, `Nhật Bản`) are retained as they appear in Vietnamese cultural QA gold references; see Appendix L of the paper for full lexicon discussion.
