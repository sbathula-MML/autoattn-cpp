#!/usr/bin/env python3

import csv
import platform
import random
import re
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "build-release" / "autoattn"
OUT = ROOT / "benchmarks"

LENGTHS = [1024, 2048, 4096]
D = 64
BLOCK = 128
REPS = 5
SEED = 7

if platform.system() != "Darwin":
    raise SystemExit("This benchmark requires macOS /usr/bin/time -l.")

if not APP.exists():
    raise SystemExit("Build the Release executable first.")

rng = random.Random(2026)
records = []

with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory)

    for n in LENGTHS:
        for repetition in range(1, REPS + 1):
            algorithms = ["standard", "tiled"]
            rng.shuffle(algorithms)

            for algorithm in algorithms:
                command = [
                    "/usr/bin/time",
                    "-l",
                    str(APP),
                    "run",
                    "--n", str(n),
                    "--d", str(D),
                    "--seed", str(SEED),
                    "--mode", algorithm,
                    "--budget-bytes", "134217728",
                    "--out", str(folder / "output.csv"),
                    "--meta", str(folder / "metrics.json"),
                ]

                if algorithm == "tiled":
                    command += ["--block", str(BLOCK)]

                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    raise RuntimeError(
                        result.stdout + "\n" + result.stderr
                    )

                match = re.search(
                    r"(\d+)\s+maximum resident set size",
                    result.stderr,
                )

                if not match:
                    raise RuntimeError(
                        "RSS measurement not found:\n"
                        + result.stderr
                    )

                rss_bytes = int(match.group(1))

                records.append({
                    "N": n,
                    "d": D,
                    "algorithm": algorithm,
                    "block_size": (
                        BLOCK if algorithm == "tiled" else 0
                    ),
                    "repetition": repetition,
                    "peak_rss_bytes": rss_bytes,
                    "peak_rss_mib": rss_bytes / (1024 ** 2),
                })

        print(f"Completed memory benchmark: N={n}", flush=True)


raw_file = OUT / "final_memory_runs.csv"

with raw_file.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(records[0].keys())
    )
    writer.writeheader()
    writer.writerows(records)


groups = defaultdict(list)

for row in records:
    groups[(row["N"], row["algorithm"])].append(
        row["peak_rss_mib"]
    )

summary = []

print("\nPeak resident memory (MiB)")
print(
    f"{'N':>6} {'Standard':>12} {'Tiled':>12} "
    f"{'Reduction':>12}"
)

for n in LENGTHS:
    standard = statistics.median(groups[(n, "standard")])
    tiled = statistics.median(groups[(n, "tiled")])

    reduction = 100 * (1 - tiled / standard)

    print(
        f"{n:6d} {standard:12.3f} {tiled:12.3f} "
        f"{reduction:11.2f}%"
    )

    for algorithm in ["standard", "tiled"]:
        values = groups[(n, algorithm)]

        summary.append({
            "N": n,
            "d": D,
            "algorithm": algorithm,
            "repetitions": len(values),
            "median_rss_mib": statistics.median(values),
            "mean_rss_mib": statistics.mean(values),
            "std_rss_mib": statistics.stdev(values),
            "min_rss_mib": min(values),
            "max_rss_mib": max(values),
        })


summary_file = OUT / "final_memory_summary.csv"

with summary_file.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

print("\nSaved:")
print(raw_file)
print(summary_file)
print(f"Total measurements: {len(records)}")
