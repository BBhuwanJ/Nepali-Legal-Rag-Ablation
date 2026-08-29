# Final Paper Analysis

Held-out test results are primary. Candidate-minus-baseline differences use paired questions.

## Chunking ablation

### Held-out retrieval metrics

| Strategy | N | Hit@1 | Hit@5 | Precision@5 | Recall@5 | MRR@5 | Complete@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| recursive | 217 | 0.6682 | 0.8848 | 0.1926 | 0.8272 | 0.7480 | 0.7696 |
| dafa | 217 | 0.7281 | 0.9078 | 0.1908 | 0.8618 | 0.7943 | 0.8157 |
| semantic | 217 | 0.5530 | 0.6774 | 0.1382 | 0.6336 | 0.5982 | 0.5899 |
| hybrid | 217 | 0.7558 | 0.9217 | 0.2147 | 0.8779 | 0.8223 | 0.8341 |

### Paired candidate comparisons

| Metric | Baseline | Difference | 95% CI | Holm p | Rank-biserial |
|---|---|---:|---|---:|---:|
| hit@1 | recursive | 0.0876 | [0.0369, 0.1384] | 0.008533 | 0.542857 |
| hit@1 | dafa | 0.0277 | [-0.0092, 0.0645] | 0.228752 | 0.333333 |
| hit@1 | semantic | 0.2028 | [0.1475, 0.2581] | 0.0 | 0.956522 |
| hit@k | recursive | 0.0369 | [0.0000, 0.0783] | 0.246185 | 0.4 |
| hit@k | dafa | 0.0138 | [-0.0138, 0.0415] | 0.425781 | 0.333333 |
| hit@k | semantic | 0.2442 | [0.1889, 0.3041] | 0.0 | 1.0 |
| precision@k | recursive | 0.0221 | [0.0074, 0.0369] | 0.001788 | 0.454927 |
| precision@k | dafa | 0.0240 | [0.0138, 0.0341] | 0.000156 | 0.714715 |
| precision@k | semantic | 0.0765 | [0.0618, 0.0912] | 0.0 | 1.0 |
| recall@k | recursive | 0.0507 | [0.0161, 0.0876] | 0.016015 | 0.554023 |
| recall@k | dafa | 0.0161 | [-0.0069, 0.0415] | 0.263321 | 0.320261 |
| recall@k | semantic | 0.2442 | [0.1912, 0.2995] | 0.0 | 1.0 |
| mrr@k | recursive | 0.0743 | [0.0389, 0.1116] | 0.000207 | 0.570492 |
| mrr@k | dafa | 0.0280 | [0.0022, 0.0552] | 0.049468 | 0.365722 |
| mrr@k | semantic | 0.2242 | [0.1746, 0.2750] | 0.0 | 0.978018 |
| complete@k | recursive | 0.0645 | [0.0230, 0.1060] | 0.014887 | 0.636364 |
| complete@k | dafa | 0.0184 | [-0.0092, 0.0507] | 0.339355 | 0.333333 |
| complete@k | semantic | 0.2442 | [0.1889, 0.2995] | 0.0 | 1.0 |

## Retrieval ablation

### Held-out retrieval metrics

| Strategy | N | Hit@1 | Hit@5 | Precision@5 | Recall@5 | MRR@5 | Complete@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| faiss_only | 217 | 0.5714 | 0.7650 | 0.1735 | 0.7143 | 0.6404 | 0.6636 |
| bm25_only | 217 | 0.7880 | 0.9124 | 0.2074 | 0.8641 | 0.8414 | 0.8157 |
| hybrid | 217 | 0.7327 | 0.9217 | 0.2120 | 0.8687 | 0.8073 | 0.8157 |
| routed_hybrid | 217 | 0.7558 | 0.9217 | 0.2147 | 0.8779 | 0.8223 | 0.8341 |

### Paired candidate comparisons

| Metric | Baseline | Difference | 95% CI | Holm p | Rank-biserial |
|---|---|---:|---|---:|---:|
| hit@1 | faiss_only | 0.1843 | [0.1244, 0.2442] | 0.0 | 0.8 |
| hit@1 | bm25_only | -0.0323 | [-0.0968, 0.0323] | 0.336289 | -0.132075 |
| hit@1 | hybrid | 0.0230 | [0.0000, 0.0461] | 0.21875 | 0.714286 |
| hit@k | faiss_only | 0.1567 | [0.1106, 0.2074] | 0.0 | 0.944444 |
| hit@k | bm25_only | 0.0092 | [-0.0323, 0.0507] | 1.0 | 0.1 |
| hit@k | hybrid | 0.0000 | [-0.0138, 0.0138] | 1.0 | 0.0 |
| precision@k | faiss_only | 0.0412 | [0.0286, 0.0544] | 0.0 | 0.847843 |
| precision@k | bm25_only | 0.0074 | [-0.0037, 0.0184] | 0.498177 | 0.235294 |
| precision@k | hybrid | 0.0028 | [-0.0037, 0.0092] | 0.635498 | 0.175824 |
| recall@k | faiss_only | 0.1636 | [0.1175, 0.2120] | 0.0 | 0.927845 |
| recall@k | bm25_only | 0.0138 | [-0.0207, 0.0484] | 0.863281 | 0.135484 |
| recall@k | hybrid | 0.0092 | [-0.0092, 0.0277] | 0.863281 | 0.327273 |
| mrr@k | faiss_only | 0.1820 | [0.1348, 0.2295] | 0.0 | 0.84854 |
| mrr@k | bm25_only | -0.0190 | [-0.0669, 0.0290] | 0.395725 | -0.118951 |
| mrr@k | hybrid | 0.0151 | [-0.0012, 0.0310] | 0.034851 | 0.653595 |
| complete@k | faiss_only | 0.1705 | [0.1198, 0.2258] | 0.0 | 0.902439 |
| complete@k | bm25_only | 0.0184 | [-0.0184, 0.0599] | 0.644531 | 0.2 |
| complete@k | hybrid | 0.0184 | [-0.0092, 0.0461] | 0.644531 | 0.4 |

