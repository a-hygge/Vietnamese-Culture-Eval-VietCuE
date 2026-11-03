# VietCuE

Dataset and code release accompanying *"VietCuE: A Discriminator-Filtered Benchmark and Multi-Axis Protocol for Diagnosing Cultural Knowledge in LLMs"*.

## About

Most Vietnamese-culture LLM benchmarks report a single aggregate accuracy number. VietCuE instead pairs a **discriminator-filtered** test set (items that three frontier models fail, then reviewed by four annotators) with a **multi-axis** evaluation protocol that decomposes "accuracy" into three signals:

1. **Aspect-based rubric score** - a 0–5 GPT-5 judge scoring content match, keyword coverage, and factual accuracy separately.
2. **Cultural Lexicon Coverage (CLC)** - whether a model's answer actually uses the culturally-specific terms present in the gold reference, independent of the judge.
3. **Cue-dependence** - how much a model's answer changes when an explicit Vietnamese-context marker is added or swapped for a wrong-country marker, measured both observationally (marker-bearing vs. marker-less items) and causally (paired minimal-edit rewrites of the same items).

Decomposing accuracy this way shows what a single leaderboard number hides. For example, lexical-coverage differences between models turn out to be driven mostly by whether answers restate entities already named in the question, and the cue gap seen in an observational split is not reproduced by paired rewrites of the same items. Full methodology and findings are in the paper.

## At a glance

| | |
|---|---|
| Test set | 247 items: frontier-filtered, then 215 accepted 4/4 by four annotators and 32 repaired by two senior annotators |
| Training corpus | 5,081 items, same source pool and category schema; duplicate groups flagged in `dataset/duplicate_groups.json` |
| Control set | 250 training items that three small open-weight models fail (`dataset/control_250/`) |
| Categories | Culture / History / Law (9 leaderboard sub-topics) |
| Cultural lexicon | 166 unique terms, used for CLC scoring |
| Evaluated models | 19, spanning closed frontier APIs and open-weight models |
| Judge | GPT-5, aspect-based rubric; 4 prompting strategies compared against a human annotator on 175 items (quadratic-weighted κ = 0.678 for the aspect-based prompt) |
| Causal cue-dependence | 3 paired-rewrite variants (v1 original, v2 correct-country marker, v3 wrong-country marker), 9 models |
| Fine-tuning | Full-parameter SFT ablation on Qwen3-4B and Qwen3-8B, 5 epochs |

## Structure

- [`dataset/`](dataset/) - VietCuE test/train sets, cultural lexicon, model outputs, judge scores, human annotations. See [dataset/README.md](dataset/README.md).
- [`code/`](code/) - data generation, inference, LLM-as-judge, fine-tuning, and analysis pipeline. See [code/README.md](code/README.md).

## Quickstart

```bash
git clone https://github.com/a-hygge/Vietnamese-Culture-Eval-VietCuE.git
cd Vietnamese-Culture-Eval-VietCuE/code
pip install -r requirements.txt
```

All released judge/model outputs are pre-computed, so the paper's tables and figures can be reproduced without re-running any LLM inference - see [code/README.md](code/README.md) for the per-script breakdown and how to point the analysis scripts at `dataset/`.

## Citation

Citation details (BibTeX) will be added here once the paper is published. In the meantime, please cite the paper by its title: *"VietCuE: A Discriminator-Filtered Benchmark and Multi-Axis Protocol for Diagnosing Cultural Knowledge in LLMs."*

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC BY-NC-SA 4.0 (see [dataset/LICENSE](dataset/LICENSE) and [dataset/README.md](dataset/README.md) for licences inherited from primary sources).
