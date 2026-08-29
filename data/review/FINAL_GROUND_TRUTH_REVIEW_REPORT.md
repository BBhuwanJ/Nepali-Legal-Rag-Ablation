# Final Ground-Truth Review

Dataset: `data/benchmark/evalData_source_checked_350_v2.json`

All answers are grounded only in the admitted Muluki Civil Code and
Domestic Violence Act texts. Exact excerpts remain in each record's
`evidence` field. The benchmark questions and ground-truth answers were
reviewed before final evaluation.

## Drafting methods

| Method | Records |
|---|---:|
| cross_act_synthesis | 45 |
| preserved_reviewed_answer | 103 |
| scope_refusal | 30 |
| single_act_evidence_answer | 172 |

Deterministic audit errors: 0
Deterministic audit warnings: 0

The final answer evaluation uses the 320 in-scope questions. The 30
out-of-scope questions are kept separately for scope/refusal checks.
