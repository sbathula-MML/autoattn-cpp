#!/usr/bin/env python3

import csv
import random
import re
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "build-release" / "autoattn"
BENCH = ROOT / "benchmarks"

SEQUENCE_LENGTHS = [128, 512, 1024]
DIMENSIONS = [32, 64]
CAUSAL_MODES = [False, True]
ALGORITHMS = ["standard", "tiled"]
REPETITIONS = 5
BLOCK_SIZE = 128
SEED = 7

if not APP.is_file():
    raise SystemExit(
        "Release executable missing. Build with CMake first."
    )

BENCH.mkdir(exist_ok=True)

raw_path = BENCH / "final_runs.csv"
summary_path = BENCH / "final_summary.csv"

rng = random.Random(2026)
records = []

def execute(n, d, causal, algorithm, tmp):
    command = [
        str(APP), "run",
        "--n", str(n),
        "--d", str(d),
        "--seed", str(SEED),
        "--mode", algorithm,
        "--budget-bytes", "16777216",
        "--out", str(tmp / "output.csv"),
        "--meta", str(tmp / "metrics.json"),
    ]

    if causal:
        command.append("--causal")

    if algorithm == "tiled":
        command += ["--block", str(BLOCK_SIZE)]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"{result.stdout}\n{result.stderr}"
        )

    match = re.search(
        r"Execution time\s*=\s*([0-9.eE+-]+)\s*ms",
        result.stdout,
    )

    if not match:
        raise RuntimeError(
            "Execution time not found:\n" + result.stdout
        )

    return float(match.group(1))


with tempfile.TemporaryDirectory() as directory:
    tmp = Path(directory)

    for n in SEQUENCE_LENGTHS:
        for d in DIMENSIONS:
            for causal in CAUSAL_MODES:

                # Warm up both implementations.
                for algorithm in ALGORITHMS:
                    execute(n, d, causal, algorithm, tmp)

                # Alternate randomized execution order.
                for repetition in range(1, REPETITIONS + 1):
                    order = ALGORITHMS.copy()
                    rng.shuffle(order)

                    for algorithm in order:
                        elapsed = execute(
                            n, d, causal, algorithm, tmp
                        )

                        records.append({
                            "N": n,
                            "d": d,
                            "causal": int(causal),
                            "algorithm": algorithm,
                            "block_size": (
                                BLOCK_SIZE
                                if algorithm == "tiled"
                                else 0
                            ),
                            "repetition": repetition,
                            "seed": SEED,
                            "execution_ms": elapsed,
                        })

                print(
                    f"Completed N={n}, d={d}, "
                    f"causal={int(causal)}",
                    flush=True,
                )


# Save every measurement.
fieldnames = [
    "N", "d", "causal", "algorithm",
    "block_size", "repetition", "seed",
    "execution_ms",
]

with raw_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)


# Aggregate measurements.
groups = defaultdict(list)

for record in records:
    key = (
        record["N"],
        record["d"],
        record["causal"],
        record["algorithm"],
    )
    groups[key].append(record["execution_ms"])

summary = []

for key in sorted(groups):
    n, d, causal, algorithm = key
    values = groups[key]

    summary.append({
        "N": n,
        "d": d,
        "causal": causal,
        "algorithm": algorithm,
        "repetitions": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.stdev(values),
        "min_ms": min(values),
        "max_ms": max(values),
    })

summary_fields = [
    "N", "d", "causal", "algorithm",
    "repetitions", "median_ms",
    "mean_ms", "std_ms", "min_ms", "max_ms",
]

with summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=summary_fields
    )
    writer.writeheader()
    writer.writerows(summary)


print("\nMedian runtime comparison (ms)")
print(
    f"{'N':>6} {'d':>4} {'Causal':>7} "
    f"{'Standard':>12} {'Tiled':>12} {'Speedup':>9}"
)

for n in SEQUENCE_LENGTHS:
    for d in DIMENSIONS:
        for causal in CAUSAL_MODES:
            standard = statistics.median(
                groups[(n, d, int(causal), "standard")]
            )
            tiled = statistics.median(
                groups[(n, d, int(causal), "tiled")]
            )

            print(
                f"{n:6d} {d:4d} {int(causal):7d} "
                f"{standard:12.4f} {tiled:12.4f} "
                f"{standard / tiled:8.3f}x"
            )

print(f"\nSaved raw runs: {raw_path}")
print(f"Saved summary:  {summary_path}")
print(f"Total measured executions: {len(records)}")
