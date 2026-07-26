# VietCuE

Dataset and code release accompanying *"Beyond Aggregate Accuracy: A Discriminator-Filtered Vietnamese Benchmark for Multi-Axis Cultural Evaluation of LLMs"*.

## About

Most Vietnamese-culture LLM benchmarks report a single aggregate accuracy number. VietCuE instead pairs a **discriminator-filtered** test set (items a frontier-LLM panel cannot already answer, then unanimously verified by four human annotators) with a **multi-axis** evaluation protocol that decomposes "accuracy" into three independent signals:

1. **Aspect-based rubric score** — a 0–5 GPT-5 judge scoring content match, keyword coverage, and factual accuracy separately.
2. **Cultural Lexicon Coverage (CLC)** — whether a model's answer actually uses the culturally-specific terms present in the gold reference, independent of the judge.
3. **Cue-dependence** — how much a model's answer changes when an explicit Vietnamese-context marker is added or swapped for a wrong-country marker, measured both observationally (marker-bearing vs. marker-less items) and causally (paired minimal-edit rewrites of the same items).

Decomposing accuracy this way surfaces patterns a single leaderboard number hides — e.g. the top-accuracy model on this benchmark (GPT-5) also has the *lowest* Cultural Lexicon Coverage and the *largest* cue-dependence gap among evaluated models, while Claude models sit at the opposite corner, trading some accuracy for much higher lexical grounding and cue-stability. Full methodology, all findings, and statistical tests are in the paper.

## At a glance

| | |
|---|---|
| Test set | 247 items, discriminator-filtered, 4-annotator unanimous consensus |
| Training corpus | 5,081 items, same source pool and category schema |
| Categories | Culture / History / Law (9 leaderboard sub-topics) |
| Cultural lexicon | 166 unique terms, used for CLC scoring |
| Evaluated models | 19, spanning closed frontier APIs and open-weight models |
| Judge | GPT-5, aspect-based rubric, validated against 2-annotator human consensus (4 prompting strategies compared; Cohen's κ = 0.748 quadratic-weighted vs. human) |
| Causal cue-dependence | 4 paired-rewrite variants (v1 original, v2 correct-country marker, v3 wrong-country marker, v4 English-name marker) |
| Fine-tuning | Full-parameter SFT ablation on Qwen3-4B and Qwen3-8B, 5 epochs |

## Structure

- [`dataset/`](dataset/) — VietCuE test/train sets, cultural lexicon, model outputs, judge scores, human annotations. See [dataset/README.md](dataset/README.md).
- [`code/`](code/) — data generation, inference, LLM-as-judge, fine-tuning, and analysis pipeline. See [code/README.md](code/README.md).

## Quickstart

```bash
git clone https://github.com/a-hygge/Vietnamese-Culture-Eval-VietCuE.git
cd Vietnamese-Culture-Eval-VietCuE/code
pip install -r requirements.txt
```

All released judge/model outputs are pre-computed, so the paper's tables and figures can be reproduced without re-running any LLM inference — see [code/README.md](code/README.md) for the per-script breakdown and how to point the analysis scripts at `dataset/`.

## License

Code: MIT. Dataset: CC BY-NC-SA 4.0 (see [dataset/README.md](dataset/README.md) for licences inherited from primary sources).
