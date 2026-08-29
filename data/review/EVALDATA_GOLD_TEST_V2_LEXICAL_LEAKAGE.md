# Candidate Benchmark Lexical-Leakage Audit

Dataset: `data\\benchmark\\evalData_gold_test_v2.json`

High-leakage threshold: 65% of non-stopword question tokens appear in gold evidence.

High-leakage records: 125/217 (57.6%).

This diagnostic does not prove invalidity. Direct section lookups naturally overlap,
but high rates in scenario or synthesis questions can inflate BM25 results and must be
reduced through human paraphrasing before test-set admission.

## By source

| Source | N | Mean coverage | High leakage |
|---|---:|---:|---:|
| muluki_ain | 97 | 0.735 | 68 |
| domestic_violence | 75 | 0.742 | 56 |
| both | 45 | 0.340 | 1 |

## High-leakage records

| ID | Source | Type | Coverage |
|---|---|---|---:|
| civil_new_002 | muluki_ain | fact | 0.909 |
| civil_new_003 | muluki_ain | procedure | 0.857 |
| civil_new_004 | muluki_ain | condition | 0.923 |
| civil_new_005 | muluki_ain | scenario | 0.722 |
| civil_new_007 | muluki_ain | fact | 0.900 |
| civil_new_008 | muluki_ain | procedure | 0.833 |
| civil_new_012 | muluki_ain | fact | 0.900 |
| civil_new_013 | muluki_ain | procedure | 0.846 |
| civil_new_014 | muluki_ain | condition | 0.889 |
| civil_new_015 | muluki_ain | scenario | 0.688 |
| civil_new_017 | muluki_ain | fact | 0.909 |
| civil_new_018 | muluki_ain | procedure | 0.846 |
| civil_new_020 | muluki_ain | scenario | 0.750 |
| civil_new_021 | muluki_ain | section_lookup | 0.786 |
| civil_new_024 | muluki_ain | exception | 0.833 |
| civil_new_025 | muluki_ain | scenario | 0.667 |
| civil_new_027 | muluki_ain | fact | 0.923 |
| civil_new_029 | muluki_ain | condition | 0.875 |
| civil_new_030 | muluki_ain | scenario | 0.667 |
| civil_new_031 | muluki_ain | section_lookup | 0.857 |
| civil_new_032 | muluki_ain | fact | 0.917 |
| civil_new_033 | muluki_ain | procedure | 0.857 |
| civil_new_034 | muluki_ain | condition | 0.900 |
| civil_new_036 | muluki_ain | section_lookup | 0.812 |
| civil_new_037 | muluki_ain | exception | 0.692 |
| civil_new_039 | muluki_ain | condition | 0.917 |
| civil_new_041 | muluki_ain | section_lookup | 0.875 |
| civil_new_042 | muluki_ain | fact | 0.923 |
| civil_new_043 | muluki_ain | procedure | 0.800 |
| civil_new_044 | muluki_ain | condition | 1.000 |
| civil_new_045 | muluki_ain | scenario | 0.688 |
| civil_new_046 | muluki_ain | section_lookup | 0.769 |
| civil_new_047 | muluki_ain | fact | 0.909 |
| civil_new_048 | muluki_ain | procedure | 0.929 |
| civil_new_051 | muluki_ain | section_lookup | 0.778 |
| civil_new_053 | muluki_ain | procedure | 0.846 |
| civil_new_054 | muluki_ain | condition | 0.900 |
| civil_new_056 | muluki_ain | section_lookup | 0.714 |
| civil_new_057 | muluki_ain | fact | 0.900 |
| civil_new_058 | muluki_ain | procedure | 0.833 |
| civil_new_059 | muluki_ain | condition | 0.923 |
| civil_new_062 | muluki_ain | fact | 0.900 |
| civil_new_063 | muluki_ain | procedure | 0.833 |
| civil_new_064 | muluki_ain | condition | 0.909 |
| civil_new_066 | muluki_ain | section_lookup | 0.706 |
| civil_new_067 | muluki_ain | fact | 0.929 |
| civil_new_068 | muluki_ain | procedure | 0.917 |
| civil_new_069 | muluki_ain | condition | 0.909 |
| civil_new_070 | muluki_ain | scenario | 0.688 |
| civil_new_071 | muluki_ain | section_lookup | 0.700 |
| civil_new_072 | muluki_ain | fact | 0.923 |
| civil_new_074 | muluki_ain | condition | 0.917 |
| civil_new_077 | muluki_ain | fact | 0.900 |
| civil_new_078 | muluki_ain | procedure | 0.750 |
| civil_new_080 | muluki_ain | scenario | 0.667 |
| civil_new_081 | muluki_ain | section_lookup | 0.688 |
| civil_new_083 | muluki_ain | procedure | 0.867 |
| civil_new_084 | muluki_ain | condition | 0.909 |
| civil_new_086 | muluki_ain | section_lookup | 0.706 |
| civil_new_087 | muluki_ain | fact | 0.933 |
| civil_new_088 | muluki_ain | procedure | 0.846 |
| civil_new_089 | muluki_ain | condition | 0.917 |
| civil_new_090 | muluki_ain | scenario | 0.706 |
| civil_new_091 | muluki_ain | section_lookup | 0.867 |
| civil_new_092 | muluki_ain | fact | 0.909 |
| civil_new_093 | muluki_ain | procedure | 0.833 |
| civil_new_094 | muluki_ain | condition | 0.875 |
| civil_new_096 | muluki_ain | section_lookup | 0.714 |
| dv_new_001 | domestic_violence | definition | 0.800 |
| dv_new_002 | domestic_violence | fact | 0.778 |
| dv_new_004 | domestic_violence | condition | 0.909 |
| dv_new_006 | domestic_violence | exception | 0.769 |
| dv_new_007 | domestic_violence | fact | 0.833 |
| dv_new_009 | domestic_violence | condition | 0.900 |
| dv_new_011 | domestic_violence | definition | 0.727 |
| dv_new_012 | domestic_violence | fact | 0.818 |
| dv_new_013 | domestic_violence | procedure | 0.706 |
| dv_new_014 | domestic_violence | condition | 0.900 |
| dv_new_016 | domestic_violence | definition | 0.769 |
| dv_new_017 | domestic_violence | fact | 0.800 |
| dv_new_018 | domestic_violence | procedure | 0.688 |
| dv_new_019 | domestic_violence | condition | 0.818 |
| dv_new_021 | domestic_violence | definition | 0.700 |
| dv_new_022 | domestic_violence | fact | 0.857 |
| dv_new_024 | domestic_violence | condition | 0.800 |
| dv_new_026 | domestic_violence | exception | 0.714 |
| dv_new_027 | domestic_violence | fact | 0.833 |
| dv_new_028 | domestic_violence | procedure | 0.750 |
| dv_new_029 | domestic_violence | condition | 0.917 |
| dv_new_030 | domestic_violence | scenario | 0.688 |
| dv_new_031 | domestic_violence | definition | 0.812 |
| dv_new_032 | domestic_violence | fact | 0.800 |
| dv_new_033 | domestic_violence | procedure | 0.667 |
| dv_new_034 | domestic_violence | condition | 0.900 |
| dv_new_036 | domestic_violence | definition | 0.800 |
| dv_new_037 | domestic_violence | exception | 0.786 |
| dv_new_038 | domestic_violence | procedure | 0.706 |
| dv_new_039 | domestic_violence | condition | 0.875 |
| dv_new_041 | domestic_violence | definition | 0.786 |
| dv_new_042 | domestic_violence | fact | 0.846 |
| dv_new_043 | domestic_violence | procedure | 0.706 |
| dv_new_044 | domestic_violence | condition | 0.909 |
| dv_new_046 | domestic_violence | definition | 0.786 |
| dv_new_047 | domestic_violence | fact | 0.846 |
| dv_new_048 | domestic_violence | procedure | 0.737 |
| dv_new_049 | domestic_violence | condition | 0.909 |
| dv_new_052 | domestic_violence | fact | 0.846 |
| dv_new_054 | domestic_violence | condition | 0.909 |
| dv_new_056 | domestic_violence | definition | 0.769 |
| dv_new_057 | domestic_violence | fact | 0.846 |
| dv_new_058 | domestic_violence | procedure | 0.667 |
| dv_new_059 | domestic_violence | condition | 0.923 |
| dv_new_061 | domestic_violence | definition | 0.727 |
| dv_new_062 | domestic_violence | fact | 0.818 |
| dv_new_063 | domestic_violence | exception | 0.750 |
| dv_new_064 | domestic_violence | condition | 0.909 |
| dv_new_066 | domestic_violence | definition | 0.750 |
| dv_new_067 | domestic_violence | fact | 0.818 |
| dv_new_068 | domestic_violence | procedure | 0.714 |
| dv_new_069 | domestic_violence | condition | 0.917 |
| dv_new_070 | domestic_violence | scenario | 0.688 |
| dv_new_071 | domestic_violence | definition | 0.727 |
| dv_new_072 | domestic_violence | fact | 0.769 |
| dv_new_074 | domestic_violence | condition | 0.857 |
| cross_new_004 | both | cross_act_synthesis | 0.667 |
