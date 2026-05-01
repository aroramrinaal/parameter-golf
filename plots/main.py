"""Orchestrator: run both charts from EXPERIMENTS.md."""
from pathlib import Path

from plots.data import parse_experiments_md
from plots.chart_bpb_frontier import draw as draw_frontier
from plots.chart_bpb_size import draw as draw_size


def main():
    md = Path(__file__).parent.parent / "EXPERIMENTS.md"
    rows = parse_experiments_md(md)

    out_frontier = Path(__file__).parent.parent / "experiments_bpb_frontier.png"
    draw_frontier(rows, out_frontier)

    out_size = Path(__file__).parent.parent / "experiments_bpb_vs_size.png"
    draw_size(rows, out_size)


if __name__ == "__main__":
    main()
