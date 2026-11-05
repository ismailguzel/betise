# BeTiSe dataset reproduction — 120,000 series

The release target is **120,000 series**, following the agreed published-size target
and the paper's three length groups, binary balance and 1,000-series generation batches.
This supersedes the earlier 60,000-series suggestion and exhaustive 1,444,000-series plan.
The current deliverable is a small reproduction package, not a new bulk-data upload.
The existing Zenodo DOI is retained; no new version or DOI is requested.

| Length | Stationary target | Non-stationary target | Total |
|---|---:|---:|---:|
| Short: 50–100 | 20,000 | 20,000 | 40,000 |
| Medium: 300–500 | 20,000 | 20,000 | 40,000 |
| Long: 1,000–10,000 | 20,000 | 20,000 | 40,000 |
| **Total** | **60,000** | **60,000** | **120,000** |

## Configuration and coverage

- [scope.json](scope.json): model/property definitions, batch quotas and sampling policy.
- [PRODUCTION_PLAN.md](PRODUCTION_PLAN.md): readable allocation table.
- [production-plan.csv](production-plan.csv): every job, seed, recipe allocation and blocker.
- [production-plan.json](production-plan.json): machine-readable plan with native BeTiSe configs.

There are **40 jobs per length × 1,000 series × 3 lengths = 120,000 series**.
A job defines a base/overlay configuration, length group, seed and categorical
sampling policy. Its selected recipes **share** the 1,000 realizations; each recipe
does not receive an additional 1,000. Up to 32 recipes are sampled per job and the
actual per-recipe allocation is recorded in the plan.

Each length has 31 distinct broad scenarios plus nine additional stationary baseline
batches. The additional batches use new seeds and realizations, not duplicated series.
They are explicitly labeled `balance_batch`, not presented as unique combinations.

All named model/property families remain represented, along with contextual anomalies
and the Section 4 stochastic-trend/mean-shift/point-anomaly example. Separate seasonal
point, seasonal collective and additional seasonal contextual combinations are omitted
from this compact release. They remain in scope with zero production batches and
explicit reasons. This is representative coverage, not the full combinatorial catalog.

Section 3.5 specifies 1,000 samples/configuration and binary balance. It does not
provide the original complete configuration manifest. Equal totals per length,
representative base choices, categorical sampling and baseline batch quotas are
explicit **v2 design choices**, not recovered historical settings.

Stationarity assignments follow the library’s generation-label convention and were
checked against the complete validation pilot.
ARCH/GARCH and SARMA are not automatically non-stationary. This is not an empirical or parameter-level certification of stationarity for every
possible realization.
The runner checks actual output labels and recipe counts; it never relabels series
to force the desired totals.

## Prepare configs and run the pilot

From the repository root, in an environment with `betise==0.6.1`:

```bash
python datasets/zenodo-v2/build_plan.py

# Replace with storage on munchlab1 that has sufficient available space.
export DATA_ROOT=/path/on/munchlab1/betise-v2
python datasets/zenodo-v2/prepare.py --profile pilot --output "$DATA_ROOT/pilot"
python datasets/zenodo-v2/run.py "$DATA_ROOT/pilot" --keep-going
```

The pilot runs the same 120 job definitions with **two series/job (240 total)**;
therefore it samples at most two recipes/job and does not validate the complete
candidate categorical coverage. Full production needs a broader recipe-quality pilot.

`prepare.py` saves ordinary native requested-generation configs with absolute output
paths, a manifest and scope snapshot. Prepare them on the machine where they will run.
`run.py` records environment versions, logs, structural validation and SHA-256 hashes.
Rerunning verifies and skips completed jobs. Move incomplete outputs outside `data/`
before retrying; no automatic deletion/overwrite occurs.

`production-plan.json` is a wrapper plan, not a native requested-generation config;
the individual configs written under `configs/` are native CLI inputs. For short
seasonal jobs, numerical overrides also reside in the manifest and are applied by
the dataset adapter. Use the managed runner to preserve those settings, validation
and checkpoint behavior.

## Full production preparation

```bash
python datasets/zenodo-v2/prepare.py --profile candidate --output "$DATA_ROOT/candidate"
python datasets/zenodo-v2/run.py "$DATA_ROOT/candidate"
```

The candidate contains exactly 120,000 planned series. Full generation is a user-run
command; it was not executed while preparing this package. Three problematic short
jobs use explicit, recorded period overrides: single 7; multiple 3 and 7. The published
BeTiSe package remains unchanged. See `REPRODUCE.md` for standalone package commands.

Global series identity is `(job_id, series_id)` because numeric IDs restart per job.
Keep series complete when merging or designing train/test splits. Production should
use the dependency versions tested by the server pilot, not a freshly resolved environment.

## Paper coverage

The model-by-model comparison, verified compound example and remaining differences
are in [PAPER_COVERAGE.md](PAPER_COVERAGE.md). Plan validation rejects missing
required paper base/overlay families or the Section 4 compound example.

## Evidence and publication

- [LEGACY_DATA_AUDIT.md](LEGACY_DATA_AUDIT.md): actual original ZIP inventory; counts differ
  from the Zenodo description. RAR contents are not yet included.
- [PILOT_120K_PLAN.md](PILOT_120K_PLAN.md): historical pilot progression.
- [SERVER_VALIDATION.md](SERVER_VALIDATION.md): final package/server verification.
- [PILOT_REPORT.md](PILOT_REPORT.md): historical broad-scenario diagnostic and blockers.
- [Published paper](https://doi.org/10.3390/cmsf2026016024), Sections 3–4.

Before scaling: inspect representative plots, event-localization labels, numerical
outputs, parameter constraints and statistical diagnostics; measure server disk,
peak RAM and runtime. A structurally valid series is not automatically scientifically
valid, and a statistical test result is not automatic label ground truth.

The existing Zenodo record is older than the normal file-editing window. To retain
its DOI, preserve its archived files and update its description with a link to the
fixed GitHub-hosted reproduction package. No new Zenodo version/DOI is created.
The Zenodo UI requires support involvement to change files of this old record.

Release wording should state that the property families are regenerated using the
updated package and explicit v2 allocation. The numerical series differ from the
original release; the paper's experiment results have not been rerun or revalidated
on this version. A 120,000 total alone does not imply identical original data.
