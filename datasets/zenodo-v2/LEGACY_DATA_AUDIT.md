# Legacy Zenodo ZIP inventory — 2026-11-05

## Scope and method

Source: [Zenodo record 18513505](https://zenodo.org/records/18513505).
The API reports two files:

| Archive | Bytes |
|---|---:|
| `Generated Data.zip` | 10,201,289,436 |
| `combinations.rar` | 13,561,138,190 |

The Mac has about 6 GB available, so the archives were not downloaded in full.
HTTP Range requests retrieved the ZIP central directory (111,005,473 bytes plus
small footer records). Counts below come from all entries in that directory.
`combinations.rar` is **not included** in the counts and has not been audited.

A numbered series CSV is identified by the filename pattern `_\d+\.csv$`.
`metadata.csv` and `metadata.json` are counted separately. This establishes the
number of matching archive entries, not that every file contains a unique, valid
series. Full numerical, duplication and metadata-consistency checks remain pending.

## Observed ZIP distribution

| Folder group | Short | Medium | Long | Total numbered CSVs |
|---|---:|---:|---:|---:|
| stationary | 129,000 | 129,000 | 129,000 | 387,000 |
| deterministic_trend | 24,000 | 24,000 | 24,000 | 72,000 |
| Stochastic Trend | 5,000 | 5,000 | 5,000 | 15,000 |
| Volatility | 4,000 | 4,000 | 4,000 | 12,000 |
| collective_anomaly | 16,000 | 16,000 | 16,000 | 48,000 |
| contextual_anomaly | 16,000 | 16,000 | 16,000 | 48,000 |
| mean_shift | 16,000 | 16,000 | 16,000 | 48,000 |
| point_anomaly | 16,000 | 16,000 | 16,000 | 48,000 |
| trend_shift | 16,000 | 16,000 | 16,000 | 48,000 |
| variance_shift | 16,000 | 16,000 | 16,000 | 48,000 |
| **Total** | **258,000** | **258,000** | **258,000** | **774,000** |

The index contains **774,798 files**: 774,000 numbered series CSVs, 399
`metadata.csv` files and 399 `metadata.json` files.

There are 387 leaf folders with 1,000 numbered series each, and 12 stationary
leaf folders with 32,250 each. The stationary leaves are four base types
(`ar`, `ma`, `arma`, `wn`) across three lengths.

Thus the ZIP's folder-level grouping is exactly balanced: 387,000 stationary-folder
entries and 387,000 entries in other folders. Per length: 129,000 versus 129,000.
This is a folder-level statement, not a fresh scientific assessment of stationarity.
In particular, volatility folders must not be assumed theoretically non-stationary
solely because they are outside the `stationary` folder.

## Small content checks

Two short CSVs were read completely using additional small Range requests. The first
metadata record was parsed from each corresponding metadata JSON prefix:

| Example | CSV rows | Metadata length | Finite values | Stationary flag |
|---|---:|---:|---|---:|
| `stationary/ar_short/ar_1.csv` | 67 | 67 | yes | 1 |
| `collective_anomaly/AR/short/single/middle/ar_collective_anomaly_1.csv` | 93 | 93 | yes | 0 |

Both examples contain `data` plus repeated property flags and a label. No explicit
`time` or `series_id` columns are present in these CSVs; the file path identifies
the series and row order supplies its sequence. The collective-anomaly example has
`collect_anom=1`. These checks corroborate the per-file interpretation for the two
samples, not every numbered CSV. Details are saved in
[legacy-sample-checks.json](legacy-sample-checks.json).

## What this changes

The published Zenodo description states 120,000 distinct series. The ZIP inventory
alone contains 774,000 numbered series entries, before inspecting the RAR archive.
This is a concrete metadata/count discrepancy. It does not establish corruption,
unusable values, uniqueness or the cause of the discrepancy. Confirm identities,
contents and metadata before drawing those conclusions.

The inventory also clarifies the apparent tension between 1,000 per configuration
and binary balance: most leaf folders have 1,000 series, while stationary baseline
folders contain many more. The original ZIP did not use only 1,000 series per
stationary base/length.

Complete per-leaf counts and original API file checksums are saved in
[legacy-zip-counts.json](legacy-zip-counts.json). The large temporary central-directory
index is not committed.

## Proposed compact replacement

The user's new working target is **120,000 series** (updated target):

| Length | Stationary target | Non-stationary target | Total |
|---|---:|---:|---:|
| Short | 20,000 | 20,000 | 40,000 |
| Medium | 20,000 | 20,000 | 40,000 |
| Long | 20,000 | 20,000 | 40,000 |
| **Total** | **60,000** | **60,000** | **120,000** |

This is a new release design, not a count recovered from the old data. The current within-group allocation is defined in `scope.json` and the generated production table. Preserve the property families and representative
categorical diversity within this fixed budget; do not expand every categorical
recipe to 1,000 realizations. That reduction in per-recipe density must be stated
as a difference from the paper-inspired exhaustive draft.

Representative metadata/CSV content checks are above; the combinations archive
inventory remains to be reviewed. Current category/model/variant quotas sum to the agreed budget.
Keep the stationarity definition separate from folder labels. The superseded
1,444,000-series plan is not an active generation target. No production dataset has
been generated or uploaded.
