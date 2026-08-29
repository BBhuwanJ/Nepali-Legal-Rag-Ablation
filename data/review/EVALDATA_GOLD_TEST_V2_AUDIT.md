# LawBot Benchmark Audit

Dataset: `data\\benchmark\\evalData_gold_test_v2.json`

This report is generated deterministically by `benchmark_audit.py`. It validates
structure and corpus references; it does not replace review by a qualified Nepali
legal professional.

## Summary

| Measure | Count |
|---|---:|
| Total Questions | 242 |
| Answerable Questions | 217 |
| Out Of Scope Questions | 25 |
| Source Qualified Questions | 217 |
| Questions With Evidence | 217 |
| Questions With Reviewer | 242 |
| Unique Target Provisions | 129 |
| Errors | 0 |
| Warnings | 0 |

## Distributions

### Sources

| Value | Count |
|---|---:|
| muluki_ain | 97 |
| domestic_violence | 75 |
| both | 45 |
| none | 25 |

### Domains

| Value | Count |
|---|---:|
| domestic_violence_act | 75 |
| cross_act_legal_remedies | 45 |
| outside_admitted_corpus | 25 |
| property_and_transactions | 20 |
| obligations_contracts_and_torts | 17 |
| civil_rights_capacity_and_obligations | 15 |
| partition | 11 |
| succession | 9 |
| adoption | 8 |
| guardianship_and_curatorship | 8 |
| private_international_law | 5 |
| parent_child_and_family_relations | 4 |

### Difficulty

| Value | Count |
|---|---:|
| hard | 100 |
| medium | 83 |
| easy | 59 |

### Review_Status

| Value | Count |
|---|---:|
| legal_reviewed | 242 |

## Blocking errors

None.

## Warnings

None.
