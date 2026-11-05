# Package validation — 2026-11-05

## Results

- Full project test suite: **171 passed** (40 existing pandas deprecation warnings).
- Nine short-seasonality boundary cases passed at lengths 50, 71 and 100.
- Local smoke pilot with explicit numerical overrides: **120/120 jobs, 240/240 series passed**.
- Linux server validation: **120/120 jobs, 7,680/7,680 series passed**.
- The full 120,000-series dataset was **not** generated.

## Server measurements

Python 3.11.16; BeTiSe 0.6.1; Linux x86_64. Dependencies are pinned in
`server-requirements.txt`. The pilot uses 64 series/job, enough to represent every
categorical recipe selected by the production policy (at most 32/job).

| Length | Jobs | Series | Observations | Parquet bytes |
|---|---:|---:|---:|---:|
| short | 40 | 2,560 | 192,367 | 15,395,592 |
| medium | 40 | 2,560 | 1,021,390 | 24,605,649 |
| long | 40 | 2,560 | 14,033,670 | 189,339,404 |

Total Parquet size: **229,340,645 bytes**. Sum of measured generation time:
**251.25 seconds**, excluding validation and startup. Peak RAM was not measured.

## Checked properties

Finite observations; requested length bounds; time order; complete series within
shards; unique IDs within each job; composition and numerical-override provenance;
exact categorical allocations and declared stationarity flags; file SHA-256 hashes.

Scientific representativeness, empirical stationarity, exhaustive coverage of the
entire BeTiSe catalog, and the paper's experiment results were not established by
these checks. Counts follow the library's documented generation labels. The scope
is marked runnable because the recipe executes successfully, not because every
possible realization has been scientifically certified.

Raw sanitized summary and tested workflow hashes: `server-validation.json`.
The server tested the same job definitions and numerical override policy before
the scope status changed from draft to runnable. No PyPI package was modified.
