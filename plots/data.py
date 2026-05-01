"""Shared data parsing & frontier helpers for experiment charts."""
from pathlib import Path
import re
import csv
import io


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
            delta = row["delta vs baseline"]
            rows.append(
                {
                    "num": num,
                    "phase": row["phase"],
                    "experiment": row["experiment"],
                    "val_bpb": val_bpb,
                    "delta": delta,
                    "steps": steps,
                    "size_mb": size_mb,
                    "valid": valid,
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda r: r["num"])
    return rows


def parse_delta(delta_str):
    """Convert delta string like '-0.0326' or '+0.1834' or '—' to a float."""
    if delta_str in ("—", "-", "", "n/a"):
        return 0.0
    cleaned = delta_str.replace("+", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def build_bpb_frontier(rows):
    """Lower-is-better cumulative-minimum frontier by experiment order."""
    vals = [r["val_bpb"] for r in rows]
    cummin = []
    cur = float("inf")
    for v in vals:
        if v < cur:
            cur = v
        cummin.append(cur)

    fx, fy = [], []
    for i, r in enumerate(rows):
        if i == 0 or r["val_bpb"] == cummin[i]:
            fx.append(r["num"])
            fy.append(r["val_bpb"])
    return fx, fy


def build_size_pareto(rows):
    """
    Return Pareto-optimal rows for (size, bpb) where both lower is better.
    A point is dominated if another point has size <= it AND bpb <= it.
    Scan by size ascending; keep points that achieve a new minimum BPB.
    """
    sorted_by_size = sorted(rows, key=lambda r: (r["size_mb"], r["val_bpb"]))
    pareto = []
    min_bpb = float("inf")
    for r in sorted_by_size:
        if r["val_bpb"] < min_bpb:
            pareto.append(r)
            min_bpb = r["val_bpb"]
    # Return in discovery order for annotation convenience
    pareto.sort(key=lambda r: r["num"])
    return pareto
