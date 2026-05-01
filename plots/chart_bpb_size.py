"""Chart 2: Validation BPB vs Compressed Model Size with Pareto frontier."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plots.data import build_size_pareto
from plots.style import (
    apply_base_spines,
    phase_colours,
    edge_colour,
    delta_colour,
    setup_figure,
    BASELINE_BPB,
    OUTLIER_THRESHOLD,
)


def _parse_delta_num(delta_str):
    try:
        return float(delta_str.replace("+", "").replace("—", "0").strip())
    except ValueError:
        return 0.0


def draw(rows, out_path):
    main_rows = [r for r in rows if r["val_bpb"] <= OUTLIER_THRESHOLD]
    outlier_rows = [r for r in rows if r["val_bpb"] > OUTLIER_THRESHOLD]
    pareto_rows = build_size_pareto(main_rows)

    if not main_rows:
        raise ValueError("No experiments left after outlier filtering!")

    phases = sorted({r["phase"] for r in main_rows})
    colours = phase_colours(phases)
    pareto_nums = {r["num"] for r in pareto_rows}

    fig, ax = setup_figure(figsize=(14, 9))

    # Limits FIRST so reference lines & text land correctly
    x_vals = [r["size_mb"] for r in main_rows]
    y_vals = [r["val_bpb"] for r in main_rows]
    x_pad = max(1.0, (max(x_vals) - min(x_vals)) * 0.08)
    y_pad = max(0.03, (max(y_vals) - min(y_vals)) * 0.12)
    ax.set_xlim(max(8, min(x_vals) - x_pad), max(x_vals) + x_pad + 1)
    ax.set_ylim(max(1.22, min(y_vals) - y_pad), max(y_vals) + y_pad + 0.03)

    # 16 MB vertical reference line + label
    ax.axvline(
        16.0,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        zorder=1,
    )
    y_top = ax.get_ylim()[1]
    ax.text(
        16.3,
        y_top - 0.015,
        "16 MB limit",
        fontsize=9,
        color="#d62728",
        ha="left",
        va="top",
    )

    # Baseline horizontal reference
    ax.axhline(
        BASELINE_BPB,
        color="#888888",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        zorder=1,
    )
    ax.text(
        ax.get_xlim()[1] - 0.3,
        BASELINE_BPB + 0.004,
        f"baseline BPB {BASELINE_BPB}",
        fontsize=9,
        color="#888888",
        ha="right",
        va="bottom",
    )

    # Scatter per phase (size ∝ training steps)
    for phase in phases:
        pr = [r for r in main_rows if r["phase"] == phase]
        ax.scatter(
            [r["size_mb"] for r in pr],
            [r["val_bpb"] for r in pr],
            s=[max(70, min(380, r["steps"] / 4)) for r in pr],
            c=[colours[phase]] * len(pr),
            edgecolors=[edge_colour(r["valid"]) for r in pr],
            linewidths=1.5,
            alpha=0.85,
            zorder=3,
            label=phase,
        )

    # Tiny experiment number on every point (subtle, for navigation)
    for r in main_rows:
        ax.annotate(
            str(r["num"]),
            (r["size_mb"], r["val_bpb"]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            fontsize=7,
            color="#444444",
            fontweight="normal",
            zorder=6,
        )

    # Delta annotations ONLY on Pareto points (alternate positions to avoid overlap)
    annotated = set()
    for idx, r in enumerate(pareto_rows):
        d = _parse_delta_num(r["delta"])
        sign = "+" if d > 0 else ""
        # alternate placement to reduce collision in the Pareto cluster
        if idx % 2 == 0:
            xytext, ha, va = (10, 10), "left", "bottom"
        else:
            xytext, ha, va = (-10, -12), "right", "top"
        ax.annotate(
            f"Δ{sign}{d:.3f}",
            (r["size_mb"], r["val_bpb"]),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=8,
            color=delta_colour(d),
            fontweight="bold",
            zorder=6,
        )
        annotated.add(r["num"])

    # Pareto frontier
    if pareto_rows:
        px = [r["size_mb"] for r in pareto_rows]
        py = [r["val_bpb"] for r in pareto_rows]
        ax.step(
            px,
            py,
            where="post",
            color="#d62728",
            linewidth=2.8,
            alpha=0.9,
            zorder=4,
            label="size↔bpb Pareto frontier",
        )
        ax.scatter(
            px,
            py,
            color="#d62728",
            s=160,
            zorder=5,
            edgecolors="white",
            linewidths=2,
            marker="D",
        )

    # Best BPB highlight (absolute lowest)
    best = min(main_rows, key=lambda r: r["val_bpb"])
    ax.scatter(
        [best["size_mb"]],
        [best["val_bpb"]],
        color="gold",
        s=500,
        zorder=7,
        edgecolors="#d62728",
        linewidths=2.5,
        marker="*",
    )
    ax.annotate(
        f"LOWEST BPB\n{best['val_bpb']:.5f} (exp {best['num']})",
        xy=(best["size_mb"], best["val_bpb"]),
        xytext=(best["size_mb"] - 3.0, best["val_bpb"] + 0.055),
        fontsize=9,
        color="#333333",
        ha="right",
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5),
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="lightyellow",
            edgecolor="#d62728",
            alpha=0.95,
        ),
        zorder=8,
    )

    # Best valid point (< 16 MB)
    valid_rows = [r for r in main_rows if r["valid"] == "yes"]
    if valid_rows:
        best_valid = min(valid_rows, key=lambda r: r["val_bpb"])
        ax.scatter(
            [best_valid["size_mb"]],
            [best_valid["val_bpb"]],
            color="none",
            s=350,
            zorder=7,
            edgecolors="#2ca02c",
            linewidths=2.5,
            marker="o",
            linestyle="--",
        )
        # Only add delta annotation if not already annotated as Pareto
        if best_valid["num"] not in annotated:
            d = _parse_delta_num(best_valid["delta"])
            sign = "+" if d > 0 else ""
            ax.annotate(
                f"Δ{sign}{d:.3f}",
                (best_valid["size_mb"], best_valid["val_bpb"]),
                textcoords="offset points",
                xytext=(12, 8),
                ha="left",
                fontsize=8,
                color=delta_colour(d),
                fontweight="bold",
                zorder=6,
            )
        ax.annotate(
            f"BEST VALID\n{best_valid['val_bpb']:.5f} (exp {best_valid['num']})",
            xy=(best_valid["size_mb"], best_valid["val_bpb"]),
            xytext=(best_valid["size_mb"] - 2.0, best_valid["val_bpb"] - 0.035),
            fontsize=9,
            color="#333333",
            ha="right",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#e8f5e9",
                edgecolor="#2ca02c",
                alpha=0.95,
            ),
            zorder=8,
        )

    # Outlier notice
    if outlier_rows:
        note = "⚠ Outliers excluded (val_bpb > 2.0):\n" + "\n".join(
            f"  Exp {r['num']}: {r['val_bpb']:.3f} – {r['experiment'][:42]}"
            for r in outlier_rows
        )
        ax.text(
            0.98,
            0.98,
            note,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            color="#664d03",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#fff3cd",
                edgecolor="#ffc107",
                alpha=0.95,
            ),
            zorder=10,
        )

    # Title / axes
    ax.set_title(
        f"Validation BPB vs Compressed Model Size\n"
        f"points={len(rows)} (shown={len(main_rows)}, outliers={len(outlier_rows)}); "
        f"Pareto points={len(pareto_rows)}; lowest BPB={best['val_bpb']:.5f} (exp {best['num']})",
        loc="left",
        fontsize=14,
        fontweight="bold",
        fontfamily="monospace",
        pad=12,
    )
    ax.set_xlabel("compressed model size (MB), lower is better →", fontsize=11)
    ax.set_ylabel("validation bpb, lower is better", fontsize=11)

    ax.grid(True, linestyle="-", alpha=0.18, color="#555555", zorder=0)
    ax.set_axisbelow(True)

    # Legends
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
            mpatches.Patch(edgecolor="#2ca02c", facecolor="white", linewidth=2, label="valid <16 MB"),
            mpatches.Patch(edgecolor="#d62728", facecolor="white", linewidth=2, label="invalid ≥16 MB"),
            mpatches.Patch(edgecolor="#ff7f0e", facecolor="white", linewidth=2, label="borderline"),
        ],
        title="Validity",
        loc="upper right",
        bbox_to_anchor=(1.0, 0.76),
        framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=9,
    )

    apply_base_spines(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved chart: {out_path}")
    plt.show(block=False)
    plt.pause(3)
    plt.close(fig)
