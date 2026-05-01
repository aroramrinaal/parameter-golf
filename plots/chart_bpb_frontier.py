"""Chart 1: Validation BPB frontier over experiment discovery order."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plots.data import build_bpb_frontier, parse_delta
from plots.style import (
    apply_base_spines,
    phase_colours,
    edge_colour,
    delta_colour,
    setup_figure,
    BASELINE_BPB,
    OUTLIER_THRESHOLD,
)


def draw(rows, out_path):
    main_rows = [r for r in rows if r["val_bpb"] <= OUTLIER_THRESHOLD]
    outlier_rows = [r for r in rows if r["val_bpb"] > OUTLIER_THRESHOLD]

    if not main_rows:
        raise ValueError("No experiments left after outlier filtering!")

    x = [r["num"] for r in main_rows]
    y = [r["val_bpb"] for r in main_rows]
    frontier_x, frontier_y = build_bpb_frontier(main_rows)
    best = min(main_rows, key=lambda r: r["val_bpb"])

    phases = sorted({r["phase"] for r in main_rows})
    colours = phase_colours(phases)

    fig, ax = setup_figure(figsize=(16, 9))

    # Phase boundary lines
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

    # Scatter per phase
    for phase in phases:
        pr = [r for r in main_rows if r["phase"] == phase]
        ax.scatter(
            [r["num"] for r in pr],
            [r["val_bpb"] for r in pr],
            s=[max(60, min(350, r["size_mb"] * 18)) for r in pr],
            c=[colours[phase]] * len(pr),
            edgecolors=[edge_colour(r["valid"]) for r in pr],
            linewidths=1.5,
            alpha=0.85,
            zorder=3,
            label=phase,
        )

    # Delta annotations on every point
    for i, r in enumerate(main_rows):
        d = parse_delta(r["delta"])
        sign = "+" if d > 0 else ""
        text = f"{sign}{d:.4f}"
        # alternate above / below to reduce collision
        y_offset = 14 if i % 2 == 0 else -22
        ax.annotate(
            text,
            (r["num"], r["val_bpb"]),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            fontsize=7.5,
            color=delta_colour(d),
            fontweight="bold" if r["num"] in frontier_x else "normal",
            zorder=6,
        )

    # Frontier line
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

    # Frontier point numbers
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

    # Best highlight
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

    # Outlier notice
    if outlier_rows:
        note = "⚠ Outliers excluded (val_bpb > 2.0):\n" + "\n".join(
            f"  Exp {r['num']}: {r['val_bpb']:.3f} – {r['experiment'][:40]}"
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

    # Title / axes
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
    ax.set_xlabel("experiment metric observations in discovered artifact order", fontsize=11)
    ax.set_ylabel("validation bpb, lower is better", fontsize=11)

    ax.grid(True, axis="y", linestyle="-", alpha=0.18, color="#555555", zorder=0)
    ax.set_axisbelow(True)

    y_min, y_max = min(y), max(y)
    y_pad = max(0.04, (y_max - y_min) * 0.12)
    ax.set_ylim(max(1.18, y_min - y_pad), y_max + y_pad + 0.03)
    ax.set_xlim(min(x) - 0.8, max(x) + 2)
    ax.set_xticks(range(min(x), max(x) + 1))

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
