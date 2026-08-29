# Final Experiment Notes

This export keeps only the final artifacts used for the manuscript.

## Data

- `data/corpus/`: cleaned legal text sources.
- `data/benchmark/evalData_source_checked_350_v2.json`: full reviewed
  benchmark.
- `data/benchmark/evalData_population_in_scope_320_v2.source_checked.json`:
  in-scope answer evaluation set.
- `data/benchmark/evalData_population_oos_30_v2.source_checked.json`:
  out-of-scope questions kept separately for scope behavior.
- `data/review/`: audit, coverage, lexical-leakage, and legal review records.

## Results

- `results/chunking_ablation/`: final chunking strategy metrics.
- `results/retrieval_ablation/`: final retrieval strategy metrics.
- `results/index_metadata/`: chunk JSON files, manifests, and index statistics.
- `results/final_paper_analysis.json`: aggregate analysis used to check paper
  tables.

## Figures

The external figures used in the manuscript are in `figures/`:

- `question_category_distribution_350.png`
- `chunking_answer_metrics.png`
- `retrieval_answer_metrics.png`

Generated FAISS indexes and pickle files are excluded from this public export.
