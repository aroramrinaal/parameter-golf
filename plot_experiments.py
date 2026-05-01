#!/usr/bin/env python3
"""
Plot experiment validation BPB frontier from EXPERIMENTS.md.
Over-engineered for clarity and aesthetics.
"""
from pathlib import Path
import re
import csv
import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def parse_experiments_md(path: Path):
    """Parse the markdown table from EXPERIMENTS.md into a list of dicts."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    table_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)

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
            valid = row["valid (<16MB)"].lower()
            rows.append(
                {
                    "num": num,
                    "phase": row["phase"],
                    "experiment": row["experiment"],
                    "val_bpb": val_bpb,
                    "delta": row["delta vs baseline"],
                    "steps": steps,
                    "size_mb": size_mb,
                    "valid": valid,
                }
            )
        except Exception:
            continue

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

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    OUTLIER_THRESHOLD = 2.0
    BASELINE_BPB = 1.36785

    main_rows = [r for r in rows if r["val_bpb"] <= OUTLIER_THRESHOLD]
    outlier_rows = [r for r in rows if r["val_bpb"] > OUTLIER_THRESHOLD]

    if not main_rows:
        raise ValueError("No experiments left after outlier filtering!")

    x = [r["num"] for r in main_rows]
    y = [r["val_bpb"] for r in main_rows]
    frontier_x, frontier_y = build_frontier(main_rows)
    best = min(main_rows, key=lambda r: r["val_bpb"])

    phases = sorted({r["phase"] for r in main_rows})
    phase_colours = dict(
        zip(phases, plt.cm.tab10(np.linspace(0, 0.9, len(phases))))
    )

    def edge_colour(valid):
        v = valid.lower()
        if v == "yes":
            return "#2ca02c"  # green
        if v == "no":
            return "#d62728"  # red
        return "#ff7f0e"      # orange for borderline / anything else

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Phase boundary vertical lines
    prev_phase = main_rows[0]["phase"]
    for r in main_rows[1:]:
        if r["phase"] != prev_phase:
            ax.axvline(
                r["num"] - 0.5,
                color="#cccccc",
                linestyle=":",
                linewidth=1.2,
                alpha=0.9,
                zorder=0,
            )
            prev_phase = r["phase"]

    # Baseline reference
    ax.axhline(
        BASELINE_BPB,
        color="#888888",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )
    ax.text(
        main_rows[-1]["num"] + 0.4,
        BASELINE_BPB + 0.006,
        f"baseline {BASELINE_BPB}",
        fontsize=9,
        color="#888888",
        ha="right",
        va="bottom",
    )

    # ------------------------------------------------------------------
    # Scatter (size = compressed MB)
    # ------------------------------------------------------------------
    for phase in phases:
        pr = [r for r in main_rows if r["phase"] == phase]
        ax.scatter(
            [r["num"] for r in pr],
            [r["val_bpb"] for r in pr],
            s=[max(60, min(350, r["size_mb"] * 18)) for r in pr],
            c=[phase_colours[phase]] * len(pr),
            edgecolors=[edge_colour(r["valid"]) for r in pr],
            linewidths=1.5,
            alpha=0.85,
            zorder=3,
            label=phase,
        )

    # ------------------------------------------------------------------
    # Frontier
    # ------------------------------------------------------------------
    ax.step(
        frontier_x,
        frontier_y,
        where="post",
        color="#d62728",
        linewidth=2.8,
        alpha=0.9,
        zorder=4,
        label="lower-envelope frontier",
    )
    ax.scatter(
        frontier_x,
        frontier_y,
        color="#d62728",
        s=140,
        zorder=5,
        edgecolors="white",
        linewidths=2,
        marker="o",
    )

    # Annotate frontier points
    for fx, fy in zip(frontier_x, frontier_y):
        ax.annotate(
            str(fx),
            (fx, fy),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center",
            fontsize=9,
            color="#d62728",
            fontweight="bold",
            zorder=6,
        )

    # ------------------------------------------------------------------
    # Best highlight
    # ------------------------------------------------------------------
    ax.scatter(
        [best["num"]],
        [best["val_bpb"]],
        color="gold",
        s=400,
        zorder=7,
        edgecolors="#d62728",
        linewidths=2.5,
        marker="*",
    )
    ax.annotate(
        f"BEST  {best['val_bpb']:.5f}\n{best['experiment'][:40]}",
        xy=(best["num"], best["val_bpb"]),
        xytext=(best["num"] - 2, best["val_bpb"] + 0.04),
        fontsize=9,
        color="#333333",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightyellow",
            edgecolor="#d62728",
            alpha=0.95,
        ),
        zorder=8,
    )

    # ------------------------------------------------------------------
    # Outlier notice (axes coordinates so it never clips)
    # ------------------------------------------------------------------
    if outlier_rows:
        note = "⚠ Outliers excluded from view (val_bpb > 2.0):\n" + "\n".join(
            f"  Exp {r['num']}: {r['val_bpb']:.3f} – {r['experiment'][:38]}"
            for r in outlier_rows
        )
        ax.text(
            0.02,
            0.98,
            note,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            color="#664d03",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#fff3cd",
                edgecolor="#ffc107",
                alpha=0.95,
            ),
            zorder=10,
        )

    # ------------------------------------------------------------------
    # Title / axes
    # ------------------------------------------------------------------
    ax.set_title(
        f"Model Experiment Validation BPB Frontier\n"
        f"points={len(rows)} (shown={len(main_rows)}, outliers={len(outlier_rows)}); "
        f"frontier={len(frontier_x)}; best={best['val_bpb']:.5f} "
        f"({best['experiment'][:50]}, exp {best['num']})",
        loc="left",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
        pad=12,
    )
    ax.set_xlabel(
        "experiment metric observations in discovered artifact order",
        fontsize=11,
    )
    ax.set_ylabel(
        "validation bpb, lower is better",
        fontsize=11,
    )

    # Grid
    ax.grid(True, axis="y", linestyle="-", alpha=0.18, color="#555555", zorder=0)
    ax.set_axisbelow(True)

    # Smart limits – adaptive but with hard floor so annotations fit
    y_min, y_max = min(y), max(y)
    y_pad = max(0.04, (y_max - y_min) * 0.12)
    ax.set_ylim(max(1.18, y_min - y_pad), y_max + y_pad + 0.03)
    ax.set_xlim(min(x) - 0.8, max(x) + 2)
    ax.set_xticks(range(min(x), max(x) + 1))

    # ------------------------------------------------------------------
    # Legends (phase + validity)
    # ------------------------------------------------------------------
    phase_leg = ax.legend(
        title="Phase",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9,
    )
    ax.add_artist(phase_leg)

    ax.legend(
        handles=[
            mpatches.Patch(
                edgecolor="#2ca02c", facecolor="white", linewidth=2, label="valid <16 MB"
            ),
            mpatches.Patch(
                edgecolor="#d62728", facecolor="white", linewidth=2, label="invalid ≥16 MB"
            ),
            mpatches.Patch(
                edgecolor="#ff7f0e", facecolor="white", linewidth=2, label="borderline"
            ),
        ],
        title="Validity",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.76),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9,
    )

    # Spines
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color("#555555")
        ax.spines[sp].set_linewidth(1.2)

    plt.tight_layout()
    out = Path(__file__).parent / "experiments_bpb_frontier.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {out.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
