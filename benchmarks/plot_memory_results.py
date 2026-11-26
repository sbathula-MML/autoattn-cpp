#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "benchmarks" / "final_memory_summary.csv"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    (int(row["N"]), row["algorithm"]): row
    for row in rows
}

lengths = [1024, 2048, 4096]

standard = [
    float(data[(n, "standard")]["median_rss_mib"])
    for n in lengths
]

tiled = [
    float(data[(n, "tiled")]["median_rss_mib"])
    for n in lengths
]

standard_std = [
    float(data[(n, "standard")]["std_rss_mib"])
    for n in lengths
]

tiled_std = [
    float(data[(n, "tiled")]["std_rss_mib"])
    for n in lengths
]

reductions = [
    100 * (1 - t / s)
    for s, t in zip(standard, tiled)
]

fig, ax = plt.subplots(figsize=(8, 5))

x = list(range(len(lengths)))
width = 0.33

bars_standard = ax.bar(
    [i - width / 2 for i in x],
    standard,
    width,
    yerr=standard_std,
    capsize=4,
    label="Standard attention"
)

bars_tiled = ax.bar(
    [i + width / 2 for i in x],
    tiled,
    width,
    yerr=tiled_std,
    capsize=4,
    label="Tiled attention (B=128)"
)

for i, reduction in enumerate(reductions):
    ax.text(
        i,
        standard[i] + max(standard_std[i], 0.3) + 2,
        f"{reduction:.1f}% reduction",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in lengths])
ax.set_xlabel("Sequence length N")
ax.set_ylabel("Peak resident memory (MiB)")
ax.set_title("Memory Efficiency of Tiled Attention")
ax.set_ylim(0, max(standard) * 1.20)
ax.legend()
ax.grid(axis="y", alpha=0.25)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(OUT / "final_memory.png", dpi=250)
fig.savefig(OUT / "final_memory.pdf")

plt.close(fig)

print("Generated:")
print(OUT / "final_memory.png")
print(OUT / "final_memory.pdf")
print("\nMeasurements: 5 repetitions per setting")
print("Bars: median process RSS")
print("Error bars: sample standard deviation")
