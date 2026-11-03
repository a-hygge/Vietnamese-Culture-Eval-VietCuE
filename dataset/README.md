# VietCuE Dataset

This bundle is the dataset release accompanying the paper *"VietCuE: A Discriminator-Filtered Benchmark and Multi-Axis Protocol for Diagnosing Cultural Knowledge in LLMs"*.

## Contents

```
dataset/
├── VietCuE_test_247.json         # 247-item test set (discriminator-filtered)
├── VietCuE_train_5081.json       # 5,081-item training corpus
├── cultural_lexicon_166.json     # 166-term cultural lexicon, partitioned by sub-topic
├── duplicate_groups.json         # flagged duplicate groups (items are flagged, not deleted)
├── control_250/                  # open-weight-filtered control set + frontier-model scores
├── model_outputs/                # 19 model raw answers on the 247-item test set
├── judge_scores/                 # 19 aspect-based GPT-5 judge scores
├── judge_ablation_outputs/       # 4 judge-prompt-strategy scores (Table: judge validation)
├── framing_variants/             # 3 framing-rewrite variants (v1–v3) + judged outputs
│   ├── judge_outputs/            #   per-item judged scores, 3 variants x 9 models
│   ├── framing_summary.json      #   paired-Wilcoxon summary, first 6 models
│   └── framing_summary_9models.json  # v2/v3 vs v1 summary for all 9 models (paper's rewrite table)
├── lm_match_outputs/             # Binary LM-match judge scores (metric-robustness check)
├── finetune_outputs/             # Per-epoch Qwen3-4B / 8B SFT judge scores
├── human_annotations/            # Anonymised human reference scores (annotator B)
│   └── excluded/                 #   annotation files not used as a human reference (see its README)
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
Same field schema as the test set; additionally carries the answers of the three filter models (`claude-opus-4.6`, `gemini-2.5-pro`, `gpt-5.4`) and the 1–5 score each model gave **its own** answer during filtering (`{model}_csat_score`, `{model}_csat_reasoning`). These are LLM self-scores, not human scores; the paper uses them for the CLC validation.

### Duplicate groups — `duplicate_groups.json`
Items are flagged rather than deleted, so ids stay stable and all paper results include every item.
- `D001`–`D130`: near-duplicate groups in the training corpus (172 LLM-proposed pairs, 139 confirmed by hand; 265 items). Keeping one item per group removes 135 items.
- `D131`: an exact duplicate question pair in the training corpus (ids 61, 1452).
- `D132`: a near-duplicate train–test pair (train 4235, test 4238).

### Control set — `control_250/`
- `control_250.json`: 250 training items, stratified by sub-topic, that three small open-weight models (Qwen3-Next-80B, Llama-3.1-8B, GPT-OSS-20B) all answer with score ≤ 2; includes the three filter models' existing answers.
- `control_250_scores.json`: those answers re-scored by the leaderboard judge.
- `stage{1,2,3}_wrong_ids.json`: the sequential filtering stages.
Code: `code/control_set/`.

### Cultural lexicon — `cultural_lexicon_166.json`
JSON object with three top-level keys: `History`, `Law`, `Culture`. After de-duplication across the three lists, the lexicon contains **166 unique Vietnamese cultural terms**. CLC with and without question-echo terms is computed by `code/analysis/clc_question_echo.py`.

### `model_outputs/`
Nineteen files, one per evaluated model. Each has top-level `metadata` + `details`. Each `details[i]`:
- `id` (int): matches test-set item id
- `question`, `gold_answer` (str): repeated for convenience
- `model_answer` (str): the model's free-form Vietnamese answer
- `category` (str): sub-topic

Naming convention: `<provider>_<model>.json` where `<provider>` is e.g. `openai`, `claude`, `gemini`, `qwen`. The full 19-model name table is in the paper's leaderboard table.

Note: the leaderboard scores of `gpt-oss-120b` and `gpt-oss-20b` were computed on an earlier generation run whose answers were not retained; the answers released here come from a later run with the same models. For `claude-opus-4.6` and `gemini-2.5-pro`, the answers are the filter-stage answers.

### `judge_scores/`
Nineteen files mirroring `model_outputs/`. Each `details[i]`:
- `id` (int): matches test-set item id
- `score` (int 0–5): aspect-based GPT-5 judge score
- `raw_response` (str): full judge response with three sub-aspect scores
- `category`, `error` (optional)

The judge was run with `model=azure/gpt-5`, `temperature=1`, `max_tokens=4000`. Judge stability is documented in the paper.

### `judge_ablation_outputs/`
Four files: `zero_shot.json`, `cot.json`, `demonstration.json`, `aspect_based.json` — GPT-5 judge scores on GPT-4o's answers to the 247 test items under each of the four prompting strategies compared in the judge-validation table. Reproduces that table together with `human_annotations/annotator_B.xlsx` (see `code/judge_ablation/build_table_judge_annotatorB.py`).

### `framing_variants/`
Three files: `v1_original.json`, `v2_vn_explicit.json`, `v3_tq_explicit.json`. Each is a 247-item parallel set with the variant-specific rewrite of the question prompt; `gold_answer` is unchanged across variants.

Variant semantics:
- `v1`: original question (no marker added)
- `v2`: prepend "`ở Việt Nam`" (correct country marker)
- `v3`: prepend "`ở Trung Quốc`" (wrong-country marker; tests routing vs. passive ignoring)

`judge_outputs/` holds the per-item aspect-based GPT-5 judge scores for all three variants across nine models (`claude-opus-4.6`, `claude-sonnet-4.6`, `deepseek-r1`, `gemini-3.1-pro-preview`, `gpt-5`, `gpt-5.4-pro`, `qwen3.5-397b`, `qwen3-next-80b`, `llama-3.3-70b`); file naming is `{version}__{model}_judged.json`. `framing_summary.json` is the paired-Wilcoxon summary (v2/v3 vs.\ v1, plus v3-vs-v2, with per-domain breakdowns) for the first six models; `framing_summary_9models.json` gives v2/v3 vs.\ v1 for all nine models, as in the paper's rewrite table (code: `code/framing/openweight/`).

### `lm_match_outputs/`
Binary LM-match judge results used for the aspect-based-vs.-binary metric-robustness check (Appendix: Metric Robustness). `csat_vs_lmmatch.json`: per-model accuracy under both metrics, rank comparison, Spearman/Pearson correlation. `csat_lmmatch_per_item.json`: aggregate agreement matrix and partial-credit breakdown across the 4,693 (model, question) pairs.

### `finetune_outputs/`
Judge scores for the per-epoch Qwen3-4B / 8B SFT checkpoints (ep1–ep5) plus their respective baselines (used for the paper's fine-tuning table). Files without an epoch suffix come from earlier exploratory runs and are not reported in the paper.

### `human_annotations/`
- `annotator_B.xlsx`: human CSAT scores (0–5) for GPT-4o's answers to test items with sample_id 73–247 (175 items), by an annotator with a Vietnamese-linguistics background. This is the human reference for the judge-validation table.
- `excluded/`: a second annotation file and the agreement report computed from it. They are **not** used as a human reference; see `excluded/README.md`.

Annotator identifiers are anonymised; raw demographics are not released.

## Reproducibility

- `category` and `id` fields are stable across all files; join on `id` to align model outputs, judge scores, and gold answers.
- Numerical results reported in the paper (CLC, leaderboard CSAT, framing Δ, fine-tune lift) can be reproduced from these artefacts with the scripts in `code/`.
- Sampling-based analyses (control-set sampling, bootstrap intervals) use `seed=3407`. No seed was passed to the generation or judge APIs.

## Source attribution

The 247 test items and 5,081 training items derive from:
1. **Vietnamese National High School Graduation Examinations** and high-school semester exams (publicly available).
2. **GPT-4-generated questions grounded in five authoritative Vietnamese textbooks** on history, culture, and law (textbook list in the paper's data-collection appendix).

## Licence

Released under CC BY-NC-SA 4.0 for non-commercial research use (see [LICENSE](LICENSE)), with the standardised-examination items inheriting the licence of their primary sources. Foreign-reference terms in the cultural lexicon (e.g.\ `Trung Quốc`, `Nhật Bản`) are retained as they appear in Vietnamese cultural QA gold references; see the paper's lexicon appendix for discussion.
