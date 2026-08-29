# Nepali Legal RAG Ablation

This repository contains the final benchmark data, experiment code, figures,
and result artifacts for a Nepali legal retrieval-augmented generation ablation
study.

The experiments compare chunking and retrieval strategies for legal question
answering over the Muluki Dewani (Civil) Code, 2017 and the Domestic Violence
(Offense and Punishment) Act, 2009.

## Repository Layout

```text
data/
  benchmark/       Final benchmark JSON files
  corpus/          Cleaned legal text sources
  review/          Audit, coverage, leakage, and legal review records

src/
  evaluation/      Answer and retrieval evaluation scripts
  experiments/     Final chunking and retrieval ablation scripts

results/
  chunking_ablation/      Final chunking metrics and per-strategy outputs
  retrieval_ablation/     Final retrieval metrics and per-strategy outputs
  index_metadata/         Chunk metadata, manifests, and index statistics
  final_paper_analysis.*  Final aggregate analysis used for paper tables

figures/           Three external figures used in the manuscript
docs/              Benchmark and experiment notes
```

Generated FAISS index binaries and pickle files are not included. The exported
index metadata keeps only chunk JSON, manifest JSON, and index statistics.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Some scripts require external LLM API keys. Keep keys in a local `.env` file
and do not commit it.

## Main Files

- Full source-checked benchmark:
  `data/benchmark/evalData_source_checked_350_v2.json`
- In-scope answer evaluation set:
  `data/benchmark/evalData_population_in_scope_320_v2.source_checked.json`
- Out-of-scope set:
  `data/benchmark/evalData_population_oos_30_v2.source_checked.json`
- Final chunking results:
  `results/chunking_ablation/`
- Final retrieval results:
  `results/retrieval_ablation/`
- Manuscript figures:
  `figures/`

The figure files are:

- `figures/question_category_distribution_350.png`
- `figures/chunking_answer_metrics.png`
- `figures/retrieval_answer_metrics.png`

## Review Note

The benchmark questions and ground-truth answers were reviewed before final
evaluation. Held-out test results and combined supplementary results are kept
separate in the result artifacts.
