"""Run prepared BeTiSe jobs sequentially; validate and checkpoint each job."""

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import betise
import numpy as np
import pandas as pd
from betise.run_scenario_generation import LENGTH_PRESETS
from generate import generate_job


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def validate(job, directory):
    config = job["config"]
    files = sorted(directory.glob("*-way/part-*.parquet"))
    summary_path = directory / "request_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary["shard_count"] != len(files) or not files:
        raise ValueError("Missing or extra shards")
    low, high = LENGTH_PRESETS[config["length_category"]]
    seen, rows, lengths = set(), 0, []
    recipe_counts = Counter()
    file_hashes = {"request_summary.json": digest(summary_path)}
    for path in files:
        frame = pd.read_parquet(path)
        if not np.isfinite(frame["data"].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite data in " + str(path))
        for column, expected in [("base_components", config["base_components"]),
                                 ("feature_components", config["features"])]:
            if frame[column].isna().any():
                raise ValueError("Missing composition metadata")
            if any(set(json.loads(value)) != set(expected) for value in frame[column].unique()):
                raise ValueError("Composition metadata mismatch")
        if job.get("params_overrides"):
            if "generation_params_overrides" not in frame:
                raise ValueError("Missing numerical override provenance")
            if any(json.loads(value) != job["params_overrides"]
                   for value in frame["generation_params_overrides"].unique()):
                raise ValueError("Numerical override provenance mismatch")
        for series_id, series in frame.groupby("series_id", sort=False):
            if series_id in seen:
                raise ValueError("Series duplicated or split across shards")
            seen.add(series_id)
            n = len(series)
            if not low <= n <= high:
                raise ValueError("Length outside configured range")
            if not np.array_equal(series["time"].to_numpy(), np.arange(n)):
                raise ValueError("Time is not the ordered sequence 0..length-1")
            if not series["length"].eq(n).all():
                raise ValueError("Length metadata mismatch")
            if "expected_is_stationary" in job:
                if not series["is_stationary"].eq(job["expected_is_stationary"]).all():
                    raise ValueError("Stationarity label does not match the provisional allocation")
            if "recipe_allocations" in job:
                recipes = series["materialized_scenario_id"].unique()
                if len(recipes) != 1:
                    raise ValueError("Recipe metadata changes within a series")
                recipe_counts[recipes[0]] += 1
            lengths.append(n)
        rows += len(frame)
        file_hashes[str(path.relative_to(directory))] = digest(path)
    expected_ids = set(range(1, config["num_series"] + 1))
    if seen != expected_ids or summary["total_series"] != config["num_series"]:
        raise ValueError("Unexpected series IDs or counts")
    if "recipe_allocations" in job:
        expected = {r["recipe_id"]: r["num_series"] for r in job["recipe_allocations"]}
        if dict(recipe_counts) != expected:
            raise ValueError("Actual categorical recipe counts differ from the plan")
    return {"series": len(seen), "rows": rows, "min_length": min(lengths),
            "max_length": max(lengths), "bytes": sum(p.stat().st_size for p in files),
            "sha256": file_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--keep-going", action="store_true", help="Record failures and inspect remaining jobs.")
    parser.add_argument("--job", action="append", help="Run only these job IDs (repeatable).")
    args = parser.parse_args()
    root = args.directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["profile"] == "candidate" and manifest["scope_status"] != "ready_for_generation":
        raise SystemExit("Candidate generation blocked: resolve scope/balance and pilot failures first.")
    if betise.__version__ != manifest["betise_version"]:
        raise SystemExit("BeTiSe version does not match manifest")
    available = {j["id"] for j in manifest["jobs"]}
    if args.job and set(args.job) - available:
        raise SystemExit("Unknown job IDs: " + str(set(args.job) - available))
    jobs = [j for j in manifest["jobs"] if not args.job or j["id"] in args.job]
    (root / "logs").mkdir(exist_ok=True)
    environment = root / "environment.json"
    current_environment = {
        "python": sys.version, "platform": platform.platform(),
        "betise_version": betise.__version__, "betise_path": betise.__file__,
        "pip_freeze": subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
    }
    if environment.exists():
        previous = json.loads(environment.read_text())
        if (previous["python"] != current_environment["python"] or
                previous["pip_freeze"] != current_environment["pip_freeze"]):
            raise SystemExit("Environment changed; use the original environment to resume.")
    else:
        environment.write_text(json.dumps(current_environment, indent=2) + "\n")
    failures = []
    for job in jobs:
        try:
            config_path = Path(job["config_path"])
            if json.loads(config_path.read_text()) != job["config"]:
                raise RuntimeError("Config differs from manifest: " + job["id"])
            directory = Path(job["config"]["output_dir"])
            marker = directory / "validation.json"
            config_hash = digest(config_path)
            if marker.exists():
                previous = json.loads(marker.read_text())
                result = validate(job, directory)
                if previous["config_sha256"] != config_hash or previous["sha256"] != result["sha256"]:
                    raise RuntimeError("Completed job changed: " + job["id"])
                print("Verified, skipping:", job["id"], flush=True)
                continue
            if directory.exists():
                raise RuntimeError("Incomplete output: " + str(directory) +
                                 ". Move it outside data/ before retrying; no automatic overwrite.")
            start = time.perf_counter()
            with (root / "logs" / (job["id"] + ".log")).open("w") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        generate_job(job, config_path)
                    except Exception:
                        traceback.print_exc(file=log)
                        raise
            elapsed = time.perf_counter() - start
            result = validate(job, directory)
            result.update(config_sha256=config_hash, generation_seconds=elapsed,
                          scope="Structural checks only; scientific and visual review pending.")
            marker.write_text(json.dumps(result, indent=2) + "\n")
            print(f"Validated {job['id']}: {result['series']} series, {elapsed:.2f}s", flush=True)
        except Exception as error:
            failures.append({"job_id": job["id"], "error": str(error)})
            print(f"FAILED {job['id']}: {error}", file=sys.stderr, flush=True)
            if not args.keep_going:
                break
    records = []
    for job in manifest["jobs"]:
        marker = Path(job["config"]["output_dir"]) / "validation.json"
        if marker.exists():
            records.append({"job_id": job["id"], **json.loads(marker.read_text())})
    report = {"expected_jobs": len(manifest["jobs"]), "completed_jobs": len(records),
              "total_series": sum(r["series"] for r in records),
              "total_parquet_bytes": sum(r["bytes"] for r in records), "jobs": records,
              "failures_this_run": failures}
    (root / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Completed {len(records)}/{len(manifest['jobs'])} jobs; {report['total_series']} series")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
