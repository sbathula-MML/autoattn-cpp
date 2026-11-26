#!/usr/bin/env python3

import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "benchmarks" / "final_summary.csv"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

with SOURCE.open(newline="") as file:
    rows = list(csv.DictReader(file))

data = {}

for row in rows:
    key = (
        int(row["N"]),
        int(row["d"]),
        int(row["causal"]),
        row["algorithm"],
    )

    if key in data:
        raise ValueError(f"Duplicate configuration: {key}")

    if int(row["repetitions"]) != 5:
        raise ValueError(f"Unexpected repetition count: {key}")

    data[key] = {
        "median": float(row["median_ms"]),
        "std": float(row["std_ms"]),
    }

if len(data) != 24:
    raise ValueError(
        f"Expected 24 algorithm/workload rows, found {len(data)}"
    )

lengths = [128, 512, 1024]
dimensions = [32, 64]
masks = [0, 1]

# Figure 1: relative speedup across workloads.
fig, ax = plt.subplots(figsize=(8, 5))

for d in dimensions:
    for causal in masks:
        speedups = []

        for n in lengths:
            standard = data[(n, d, causal, "standard")]["median"]
            tiled = data[(n, d, causal, "tiled")]["median"]
            speedups.append(standard / tiled)

        label = f"d={d}, " + (
            "causal" if causal else "noncausal"
        )

        ax.plot(
            lengths,
            speedups,
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )

ax.axhline(1, linestyle="--", color="gray", linewidth=1)
ax.set_xlabel("Sequence length N")
ax.set_ylabel("Speedup (standard / tiled)")
ax.set_title("Tiled Attention Performance Advantage")
ax.set_xticks(lengths)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()

fig.savefig(OUT / "final_speedup.png", dpi=220)
fig.savefig(OUT / "final_speedup.pdf")
plt.close(fig)


# Figure 2: median runtime and sample standard deviation.
labels = []
standard_times = []
tiled_times = []
standard_std = []
tiled_std = []

for n in lengths:
    for d in dimensions:
        for causal in masks:
            labels.append(
                f"N={n}\nd={d}, C={causal}"
            )

            s = data[(n, d, causal, "standard")]
            t = data[(n, d, causal, "tiled")]

            standard_times.append(s["median"])
            tiled_times.append(t["median"])
            standard_std.append(s["std"])
            tiled_std.append(t["std"])

positions = list(range(len(labels)))
width = 0.36

fig, ax = plt.subplots(figsize=(14, 6))

ax.bar(
    [x - width / 2 for x in positions],
    standard_times,
    width,
    label="Standard",
    yerr=standard_std,
    capsize=2,
)

ax.bar(
    [x + width / 2 for x in positions],
    tiled_times,
    width,
    label="Tiled (B=128)",
    yerr=tiled_std,
    capsize=2,
)

ax.set_yscale("log")
ax.set_xticks(positions)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Execution time (ms, log scale)")
ax.set_title(
    "Standard vs. Tiled Attention: "
    "Median Runtime and Variability"
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()

fig.savefig(OUT / "final_runtime.png", dpi=220)
fig.savefig(OUT / "final_runtime.pdf")
plt.close(fig)

print("Generated:")
for name in [
    "final_speedup.png",
    "final_speedup.pdf",
    "final_runtime.png",
    "final_runtime.pdf",
]:
    print(" ", OUT / name)

print("\nVerified 24 configurations with 5 repetitions each.")
