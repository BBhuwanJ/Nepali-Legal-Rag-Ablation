# Evaluating Chunking and Retrieval Strategies for Nepali Legal RAG

This repository contains the code, benchmark, experiment outputs, and IEEE manuscript for a Nepali legal Retrieval-Augmented Generation (RAG) study.

The study asks: **how do chunking and retrieval choices affect answers generated from Nepali statutes?** We run two controlled ablations over the same legal corpus and benchmark:

1. four chunking strategies with retrieval held fixed; and
2. four retrieval strategies with chunking held fixed.

On the held-out answerable test set (`N = 217`), hybrid legal-first chunking achieved the highest answer correctness (`0.8894`) and the strongest overall deterministic retrieval results. With hybrid chunks held constant, source-aware routed hybrid retrieval also achieved the highest answer correctness (`0.8894`), Recall@5 (`0.8779`), and retrieval relevance (`0.7714`). BM25 alone produced the best Hit@1 (`0.7880`) and MRR@5 (`0.8414`), showing that exact statutory language remains valuable.

> **Research status:** This repository accompanies a manuscript in preparation. It does not claim publication or acceptance. The system provides access to legal information and is not a substitute for advice from a qualified legal professional.

## Paper

**Title:** *Evaluating Chunking and Retrieval Strategies for Nepali Legal Retrieval-Augmented Generation*

- [IEEE manuscript source](IEEE_LawBot_Chunking_Strategies.tex)
- [Experiment runbook](backend/experiments/RUNBOOK.md)
- [Benchmark protocol](backend/evaluationV2/BENCHMARK_PROTOCOL.md)
- [Gold benchmark audit](backend/evaluationV2/EVALDATA_GOLD_V2_AUDIT.md)
- [Chunking result tables](backend/experiments/chunking_ablation/ieee_tables.txt)
- [Retrieval result tables](backend/experiments/retrieval_ablation/ieee_tables.txt)

## Why this study

Legal RAG systems must retrieve the controlling provision before generating an answer. A fluent response is not reliable when the correct legal authority is absent from its context.

Nepali statutes create specific retrieval problems:

- a fixed text window can split a rule from its condition or exception;
- semantic similarity does not guarantee preservation of a statutory section (`दफा`);
- exact legal terms and section numbers favor lexical retrieval;
- paraphrased questions favor dense semantic retrieval;
- some questions require provisions from more than one Act; and
- answers must preserve Act, part, chapter, and section metadata for citation.

The experiments isolate these design choices instead of changing several parts of the system at once.

## Research questions

### RQ1: Which chunking strategy works best for Nepali statutory text?

We compare recursive character, section-based, semantic, and hybrid legal-first semantic chunking. The retriever, generator, benchmark, embedding model, and `top-k` remain fixed.

### RQ2: Which retrieval strategy works best when chunking is fixed?

We compare FAISS semantic search, BM25, standard hybrid retrieval, and source-aware routed hybrid retrieval. Every method searches the same hybrid legal-first chunk index.

### RQ3: Do the improvements appear in both answer quality and provision retrieval?

We report LLM-judged answer metrics separately from deterministic provision-level retrieval metrics. This prevents a high-quality generated answer from hiding weak evidence retrieval.

## Legal corpus

The knowledge base contains two Nepali statutes:

- **Muluki Dewani (Civil) Code, 2017**
- **Domestic Violence (Offense and Punishment) Act, 2009**

The source text was normalized to Unicode Nepali. Page headers, footers, and extraction artifacts were removed. The pipeline retains source, part (`भाग`), chapter (`परिच्छेद`), and section (`दफा`) metadata.

The embedding model is [`Yunika/sentence-transformer-nepali`](https://huggingface.co/Yunika/sentence-transformer-nepali). Dense vectors are stored and searched with FAISS. BM25 supplies lexical retrieval. Gemini generates answers from the retrieved top-five chunks.

## Benchmark

The frozen gold benchmark contains **350 questions** and **210 unique target provisions**. Its deterministic audit reports no schema or corpus-reference errors and no warnings.

| Split or group | Answerable | Out of scope | Total | Use |
|---|---:|---:|---:|---|
| Development | 103 | 5 | 108 | Development and diagnosis |
| Held-out test | 217 | 25 | 242 | Primary paper results |
| Combined | 320 | 30 | 350 | Supplementary reporting |

Source distribution across all questions:

| Required source | Questions |
|---|---:|
| Civil Code | 180 |
| Domestic Violence Act | 90 |
| Both Acts | 50 |
| Outside the admitted corpus | 30 |

| Difficulty | Questions |
|---|---:|
| Easy | 106 |
| Medium | 131 |
| Hard | 113 |

Ordinary answer and retrieval metrics exclude the 30 out-of-scope questions. Refusal compliance is reported separately. See the [benchmark audit](backend/evaluationV2/EVALDATA_GOLD_V2_AUDIT.md) and [held-out coverage report](backend/evaluationV2/EVALDATA_GOLD_TEST_V2_COVERAGE.md).

![Benchmark question distribution](backend/evaluationV2/charts_900/question_category_distribution_350.png)

## System pipeline

```text
Nepali statutes
      |
      v
Unicode cleanup and statutory metadata extraction
      |
      v
Selected chunking strategy
      |
      v
Nepali sentence embeddings + FAISS index + BM25 index
      |
      v
Selected retriever returns top 5 chunks
      |
      v
Gemini generates an answer with legal citations
      |
      v
LLM-judged answer metrics + deterministic retrieval metrics
```

## Experiment 1: chunking ablation

The chunking strategy changes while the **source-aware routed hybrid retriever** remains fixed.

### Compared strategies

| Strategy | Method |
|---|---|
| Recursive character | Splits text by a size-limited recursive character rule. |
| Section-based | Treats each statutory section (`दफा`) as the main chunk. |
| Semantic | Groups text by embedding similarity without enforcing section boundaries. |
| Hybrid legal-first | Extracts complete sections first, validates structure, and merges only related adjacent sections under size constraints. |

### Index statistics

| Strategy | Chunks | Mean characters | Minimum | Maximum | Multi-section chunks |
|---|---:|---:|---:|---:|---:|
| Recursive character | 556 | 786.1 | 116 | 1,025 | 40.1% |
| Section-based | 745 | 739.4 | 190 | 3,170 | 0.0% |
| Semantic | 1,495 | 289.8 | 100 | 901 | 0.0% |
| **Hybrid legal-first** | **745** | **583.8** | **69** | **1,600** | **4.2%** |

The hybrid design uses a `0.75` cosine-similarity threshold for eligible adjacent sections and preserves a section list in merged-chunk metadata.

## Experiment 2: retrieval ablation

The **hybrid legal-first chunk index** remains fixed while the retrieval method changes.

| Strategy | Method |
|---|---|
| Semantic FAISS only | Dense similarity search only. |
| BM25 only | Sparse lexical ranking only. |
| Hybrid | Combines semantic similarity, BM25, and direct section-number boosting. |
| Source-aware routed hybrid | Adds query-based Act routing and multi-source diversification to the hybrid retriever. |

The hybrid weights adapt to query type. Direct section queries receive a strong exact-section boost. Definition and general queries use different dense-to-sparse weights. Source routing is isolated to the routed-hybrid condition.

## Evaluation metrics

### Answer-quality metrics

Gemini 3.5 Flash evaluates each generated answer at temperature `0.0`. Scores are normalized to the `[0, 1]` range.

| Metric | What it measures |
|---|---|
| Correctness | Agreement with the reference legal answer. |
| Groundedness | Whether claims are supported by the retrieved context. |
| Answer relevance | Whether the response directly addresses the question. |
| Retrieval relevance | Whether the supplied context is relevant to the question. |

These are model-judged metrics. Temperature zero reduces variation but does not guarantee identical judgments across repeated hosted-model runs.

### Deterministic retrieval metrics

Gold source and section identifiers are compared with the retrieved top-five chunks.

| Metric | Meaning |
|---|---|
| Hit@1 | At least one required provision appears at rank 1. |
| Hit@5 | At least one required provision appears in the top 5. |
| Precision@5 | Fraction of retrieved top-five chunks that match required provisions. |
| Recall@5 | Fraction of required provisions recovered in the top 5. |
| MRR@5 | Reciprocal rank of the first relevant provision, capped at rank 5. |
| Complete@5 | All required provisions appear in the top 5. |

Retrieval matching is source-qualified. The same section number in a different Act does not count as correct.

## Main results

Primary results use the held-out answerable test set (`N = 217`). Combined answerable results (`N = 320`) are supplementary.

### Chunking: held-out answer quality

| Chunking strategy | Correctness | Groundedness | Answer relevance | Retrieval relevance |
|---|---:|---:|---:|---:|
| Recursive character | 0.8571 | 0.9972 | 0.9843 | **0.7871** |
| Section-based | 0.8885 | 0.9954 | **0.9871** | 0.7410 |
| Semantic | 0.7751 | 0.9963 | 0.9751 | 0.7447 |
| **Hybrid legal-first** | **0.8894** | **0.9991** | 0.9862 | 0.7714 |

### Chunking: held-out deterministic retrieval

| Chunking strategy | Hit@1 | Hit@5 | Precision@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Recursive character | 0.6682 | 0.8848 | 0.1926 | 0.8272 | 0.7480 |
| Section-based | 0.7281 | 0.9078 | 0.1908 | 0.8618 | 0.7943 |
| Semantic | 0.5530 | 0.6774 | 0.1382 | 0.6336 | 0.5982 |
| **Hybrid legal-first** | **0.7558** | **0.9217** | **0.2147** | **0.8779** | **0.8223** |

![Chunking answer-quality results](backend/experiments/chunking_ablation/chunking_answer_metrics.png)

### Retrieval: held-out answer quality

| Retrieval strategy | Correctness | Groundedness | Answer relevance | Retrieval relevance |
|---|---:|---:|---:|---:|
| Semantic FAISS only | 0.7300 | **1.0000** | 0.9401 | 0.6221 |
| BM25 only | 0.8654 | 0.9972 | **0.9880** | 0.7235 |
| Hybrid | 0.8691 | 0.9871 | 0.9742 | 0.7484 |
| **Source-aware routed hybrid** | **0.8894** | 0.9991 | 0.9862 | **0.7714** |

### Retrieval: held-out deterministic retrieval

| Retrieval strategy | Hit@1 | Hit@5 | Precision@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|---:|---:|
| Semantic FAISS only | 0.5714 | 0.7650 | 0.1735 | 0.7143 | 0.6404 |
| **BM25 only** | **0.7880** | 0.9124 | 0.2074 | 0.8641 | **0.8414** |
| Hybrid | 0.7327 | **0.9217** | 0.2120 | 0.8687 | 0.8073 |
| Source-aware routed hybrid | 0.7558 | **0.9217** | **0.2147** | **0.8779** | 0.8223 |

![Retrieval answer-quality results](backend/experiments/retrieval_ablation/retrieval_answer_metrics.png)

### Supplementary combined results

These values use all 320 answerable development and test questions. They must not be described as held-out results.

| Ablation | Best method | Correctness | Hit@1 | Hit@5 | Precision@5 | Recall@5 | MRR@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Chunking | Hybrid legal-first | 0.8787 | 0.6969 | 0.9031 | 0.2223 | 0.8253 | 0.7809 |
| Retrieval | Source-aware routed hybrid | 0.8787 | 0.6969 | 0.9031 | 0.2223 | 0.8253 | 0.7809 |

The two rows represent the same experimental cell: hybrid legal-first chunks with source-aware routed hybrid retrieval. The populated inputs are reused only after fingerprint checks confirm that the dataset, index, retrieval mode, and implementation match.

## Findings

1. **Legal-first chunking gave the best overall balance.** It had the highest held-out correctness and led every deterministic retrieval metric in the chunking ablation.
2. **Pure semantic chunking was the weakest chunking condition.** It produced many short chunks and had the lowest correctness, Hit@1, Hit@5, Recall@5, and MRR@5.
3. **Section boundaries were a strong baseline.** Section-based chunking nearly matched hybrid answer correctness and was second on most deterministic retrieval metrics.
4. **BM25 remained highly competitive.** It achieved the best retrieval Hit@1 and MRR@5. Exact legal terms and section numbers provide a strong lexical signal.
5. **Routed hybrid retrieval produced the best downstream answers.** It achieved the highest correctness and retrieval relevance while also giving the best Precision@5 and Recall@5.
6. **High groundedness did not imply complete retrieval.** Groundedness was near `1.0` across methods, but retrieval coverage differed. A generator can remain faithful to its context even when that context omits the controlling provision.
7. **Top-five evidence coverage remains a bottleneck.** Hybrid methods improved coverage, but no method recovered every required provision for every question.

## Refusal results

Out-of-scope refusal compliance was weak and is not hidden inside the ordinary answer metrics.

| Retrieval strategy | Held-out refusal rate (`N = 25`) | Combined refusal rate (`N = 30`) |
|---|---:|---:|
| Semantic FAISS only | 0.0800 | 0.1000 |
| BM25 only | 0.0400 | 0.0333 |
| Hybrid | 0.0000 | 0.0333 |
| Source-aware routed hybrid | 0.0400 | 0.0333 |

This experiment was designed mainly to compare answerable-question retrieval. The low refusal rates show that production use needs a separate scope detector or abstention policy.

## Limitations

- The corpus contains only two Nepali statutes. Results should not be generalized to all Nepali law.
- The held-out Civil Code questions cover 107 of 721 provisions (`14.84%`), while the Domestic Violence Act portion covers all 22 provisions.
- The benchmark has substantial lexical overlap with the corpus. The test audit flags 125 of 217 answerable questions above the predeclared 65% token-overlap threshold.
- Answer-quality scores rely on one hosted LLM judge. They are not a substitute for repeated evaluation or independent legal-expert assessment.
- No paired significance claim is made. Small differences, such as hybrid versus section-based correctness, should be interpreted cautiously.
- Retrieval uses one Nepali sentence-embedding model and `k = 5`. Other models and context sizes may change the ranking.
- Cross-references and multi-provision questions remain difficult.
- Refusal behavior is not ready for deployment.
- The experiments evaluate information retrieval and grounded generation, not legal advice quality or real-world legal outcomes.

## Reproducing the experiments

### 1. Set up Python

```powershell
git clone https://github.com/BBhuwanJ/Nepali-Legal-Rag-Ablation.git
Set-Location Nepali-Legal-Rag-Ablation
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Add at least one Gemini API key to `backend/.env`. Do not commit this file. Index construction may download the embedding model on first use.

### 2. Run local, API-free checks

```powershell
python backend/experiments/benchmark_audit.py `
  --dataset backend/evaluationV2/evalData_gold_v2.json `
  --strict

python -m unittest backend.experiments.tests.test_multi_document_corpus

python backend/experiments/retrieval_only_eval.py `
  --dataset backend/evaluationV2/evalData_gold_v2.json `
  --experiment all `
  --k 5 `
  --output-dir backend/experiments/reproduction_retrieval
```

### 3. Run the full frozen experiment

```powershell
python backend/experiments/run_experiment.py `
  --dataset backend/evaluationV2/evalData_gold_v2.json `
  --benchmark-manifest backend/evaluationV2/benchmark_release_manifest.json `
  --restart-population
```

This step calls Gemini for answer generation and judging. It can take hours under API rate limits. Population and evaluation scripts save progress after each question.

For quota-controlled execution, run one strategy at a time. The full commands, resume rules, expected API load, and troubleshooting steps are in the [experiment runbook](backend/experiments/RUNBOOK.md).

### 4. Regenerate paper tables

```powershell
python backend/experiments/chunking_ablation/05_summarize_results.py
python backend/experiments/retrieval_ablation/05_summarize_results.py
```

### 5. Compile the IEEE manuscript

XeLaTeX is required for Nepali Unicode text.

```powershell
xelatex -interaction=nonstopmode -halt-on-error IEEE_LawBot_Chunking_Strategies.tex
xelatex -interaction=nonstopmode -halt-on-error IEEE_LawBot_Chunking_Strategies.tex
```

## Repository structure

```text
.
├── IEEE_LawBot_Chunking_Strategies.tex    # IEEE manuscript
├── backend/
│   ├── data/                              # Nepali legal corpus and indexes
│   ├── evaluationV2/                      # Frozen benchmark, audits, and protocols
│   └── experiments/
│       ├── shared/                        # Shared chunking, retrieval, and evaluation code
│       ├── chunking_ablation/             # Four chunking conditions and results
│       ├── retrieval_ablation/            # Four retrieval conditions and results
│       ├── RUNBOOK.md                     # Full reproduction instructions
│       └── final_paper_analysis.json       # Additional paper analysis
└── frontend/                              # Nepali legal-chatbot interface
```

## Key evidence files

| Purpose | Artifact |
|---|---|
| Frozen benchmark | [`evalData_gold_v2.json`](backend/evaluationV2/evalData_gold_v2.json) |
| Held-out test | [`evalData_gold_test_v2.json`](backend/evaluationV2/evalData_gold_test_v2.json) |
| Benchmark audit | [`EVALDATA_GOLD_V2_AUDIT.md`](backend/evaluationV2/EVALDATA_GOLD_V2_AUDIT.md) |
| Test coverage | [`EVALDATA_GOLD_TEST_V2_COVERAGE.md`](backend/evaluationV2/EVALDATA_GOLD_TEST_V2_COVERAGE.md) |
| Chunking outputs | [`chunking_ablation/results`](backend/experiments/chunking_ablation/results/) |
| Retrieval outputs | [`retrieval_ablation/results`](backend/experiments/retrieval_ablation/results/) |
| Chunking summary | [`chunking_ablation/ieee_tables.txt`](backend/experiments/chunking_ablation/ieee_tables.txt) |
| Retrieval summary | [`retrieval_ablation/ieee_tables.txt`](backend/experiments/retrieval_ablation/ieee_tables.txt) |

## Result interpretation

- Treat the held-out answerable test set (`N = 217`) as the primary result.
- Label `N = 320` answerable results as combined and supplementary.
- Report the 30 out-of-scope questions separately.
- Keep LLM-judged scores separate from deterministic retrieval metrics.
- Do not claim statistical significance without final paired per-question testing.
- Do not describe the manuscript as published, accepted, or under review without separate evidence.

## Citation

A formal BibTeX citation will be added after the manuscript has a public archival record. Until then, cite this repository and include the commit hash used for the experiment.

```bibtex
@misc{joshi_nepali_legal_rag_ablation,
  title        = {Evaluating Chunking and Retrieval Strategies for Nepali Legal Retrieval-Augmented Generation},
  author       = {Joshi, Bhuwan Prasad and Bhandari, Anshu},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/BBhuwanJ/Nepali-Legal-Rag-Ablation},
  note         = {Manuscript in preparation; cite the repository commit used}
}
```

## Responsible use

This system is a research prototype. Generated answers can omit relevant provisions, mishandle cross-references, or respond when they should refuse. Verify every legal answer against the official statute and consult a qualified legal professional for legal advice.
