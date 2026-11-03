# Excluded annotation files

These files are kept for transparency but are **not** used as a human reference.

- `annotator_A.xlsx`: scores for sample_id 1–175. They coincide on all 175 items with
  reference scores that an LLM had pre-filled in the annotation sheet, so they are not an
  independent human judgement.
- `kappa_report_A_vs_B.txt`, `overlap_A_vs_B.csv`: agreement between annotator A and
  annotator B on the 103 overlapping items. Because of the issue above, the reported
  κ = 0.748 is not an inter-annotator agreement and is not reported in the paper.

The judge-validation table in the paper uses annotator B only
(`../annotator_B.xlsx`, 175 items); see `code/judge_ablation/build_table_judge_annotatorB.py`.
