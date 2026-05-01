#!/usr/bin/env python3
"""
Plot experiment validation BPB frontier from EXPERIMENTS.md.
Usage:
    uv run plot_experiments.py
    # or
    source .venv/bin/activate && python plot_experiments.py
"""
from pathlib import Path
import re
import csv
import io

import numpy as np
import matplotlib.pyplot as plt


def parse_experiments_md(path: Path):
    """Parse the markdown table from EXPERIMENTS.md into a list of dicts."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Collect contiguous lines that look like table rows
    table_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)

    # Drop the markdown separator line  |---|---|...
    data_lines = [
        l for l in table_lines if not re.match(r"^\|\s*[-:]+\s*\|", l)
    ]

    if not data_lines:
        raise ValueError("No markdown table found in EXPERIMENTS.md")

    reader = csv.DictReader(
        io.StringIO("\n".join(data_lines)),
        delimiter="|",
        skipinitialspace=True,
    )

    rows = []
    for raw in reader:
        row = {k.strip(): v.strip() for k, v in raw.items() if k and k.strip()}
        if not row:
            continue
        try:
            num = int(row["#"])
            val_bpb = float(row["val_bpb"])
            steps_raw = row["steps"].replace("~", "").strip()
            steps = int(steps_raw)
            size_raw = row["compressed size"].replace("MB", "").strip()
            size_mb = float(size_raw)
            rows.append(
                {
                    "num": num,
                    "phase": row["phase"],
                    "experiment": row["experiment"],
                    "val_bpb": val_bpb,
                    "delta": row["delta vs baseline"],
                    "steps": steps,
                    "size_mb": size_mb,
                    "valid": row["valid (<16MB)"],
                }
            )
        except Exception as exc:
            # Skip header/footer/separator rows that fail parsing
            continue

    # Sort by discovery / experiment number
    rows.sort(key=lambda r: r["num"])
    return rows


def build_frontier(rows):
    """Lower-is-better cumulative-minimum frontier."""
    y = np.array([r["val_bpb"] for r in rows])
    cummin = np.minimum.accumulate(y)

    frontier_x = []
    frontier_y = []
    for i, r in enumerate(rows):
        if i == 0 or r["val_bpb"] == cummin[i]:
            frontier_x.append(r["num"])
            frontier_y.append(r["val_bpb"])
    return frontier_x, frontier_y


def main():
    md_path = Path(__file__).parent / "EXPERIMENTS.md"
    rows = parse_experiments_md(md_path)

    x = [r["num"] for r in rows]
    y = [r["val_bpb"] for r in rows]
    frontier_x, frontier_y = build_frontier(rows)

    # Colour by phase (hardware info is not in the table, phase is the next-best grouping)
    phases = sorted({r["phase"] for r in rows})
    cmap = plt.cm.tab10
    phase_colour = {phase: cmap(i / max(1, len(phases) - 1)) for i, phase in enumerate(phases)}

    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Scatter per phase so legend entries are clean
    for phase in phases:
        px = [r["num"] for r in rows if r["phase"] == phase]
        py = [r["val_bpb"] for r in rows if r["phase"] == phase]
        ax.scatter(
            px,
            py,
            label=phase,
            color=phase_colour[phase],
            edgecolors="black",
            linewidths=0.4,
            s=70,
            zorder=3,
            alpha=0.9,
        )

    # Frontier line (step-like, post style so it holds until the next new best)
    ax.step(
        frontier_x,
        frontier_y,
        where="post",
        color="#d62728",
        linewidth=2.4,
        label="lower-envelope frontier",
        zorder=4,
    )

    # Emphasise frontier points
    ax.scatter(
        frontier_x,
        frontier_y,
        color="#d62728",
        s=110,
        zorder=5,
        edgecolors="white",
        linewidths=1.2,
    )

    # Annotate best
    best = min(rows, key=lambda r: r["val_bpb"])
    ax.annotate(
        f"best={best['val_bpb']:.5f} ({best['experiment']}, exp {best['num']})",
        xy=(best["num"], best["val_bpb"]),
        xytext=(best["num"] + 1.5, best["val_bpb"] - 0.08),
        fontsize=9,
        color="#333333",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
    )

    # Title / labels styled like the example
    ax.set_title(
        f"Model Experiment Validation BPB Frontier\n"
        f"points={len(rows)}; frontier={len(frontier_x)}; "
        f"best={best['val_bpb']:.5f} ({best['experiment']}, exp {best['num']})",
        loc="left",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
    )
    ax.set_xlabel(
        "experiment metric observations in discovered artifact order",
        fontsize=11,
    )
    ax.set_ylabel("validation bpb, lower is better", fontsize=11)

    # Grid
    ax.grid(True, linestyle="-", alpha=0.25, which="both", color="#555555")
    ax.set_axisbelow(True)

    # Limits with breathing room
    ymin = min(y)
    ymax = max(y)
    pad = (ymax - ymin) * 0.05 if ymax != ymin else 0.1
    ax.set_ylim(ymin - pad, ymax + pad * 2)
    ax.set_xlim(min(x) - 1, max(x) + 2)

    # Legend
    ax.legend(
        title="Phase",
        loc="upper right",
        framealpha=0.95,
        edgecolor="#cccccc",
    )

    # Spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#555555")

    plt.tight_layout()
    out = Path(__file__).parent / "experiments_bpb_frontier.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
