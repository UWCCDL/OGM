"""
swow_reduce.py
--------------
Reduces a SWOW associative-strength TSV to a smaller file containing only
the most connected nodes, measured by total edge weight (sum of associative
strengths across all incoming and outgoing edges).

Only edges where BOTH endpoints are in the top-N node set are kept,
producing a clean induced subgraph.

Usage:
    python swow_reduce.py --data <input_file> --output <output_file> [--n 10000]

Example:
    python swow_reduce.py \
        --data strength.SWOW-EN.R123.20180827.csv \
        --output strength.SWOW-EN.R123.top10k.csv
"""

import argparse
import csv
import sys
from collections import defaultdict


def compute_total_weights(
    rows: list[list[str]],
    cue_idx: int,
    resp_idx: int,
    str_idx: int,
) -> dict[str, float]:
    """
    Sum associative strengths for every word across both its outgoing edges
    (as a cue) and incoming edges (as a response).
    """
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        try:
            strength = float(row[str_idx])
        except ValueError:
            continue
        if strength <= 0:
            continue
        cue      = row[cue_idx].strip().lower()
        response = row[resp_idx].strip().lower()
        if response in ("", "nan"):
            continue
        totals[cue]      += strength
        totals[response] += strength
    return dict(totals)


def reduce_graph(
    input_path: str,
    output_path: str,
    n_nodes: int,
    strength_col: str = "R123.Strength",
) -> None:
    # ------------------------------------------------------------------ load
    with open(input_path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        rows   = list(reader)

    try:
        cue_idx  = header.index("cue")
        resp_idx = header.index("response")
        str_idx  = header.index(strength_col)
    except ValueError:
        available = ", ".join(header)
        sys.exit(
            f"Column '{strength_col}' not found.\n"
            f"Available columns: {available}"
        )

    print(f"Loaded {len(rows):,} edges from '{input_path}'.")

    # ------------------------------------------- rank nodes by total weight
    totals = compute_total_weights(rows, cue_idx, resp_idx, str_idx)
    print(f"Found {len(totals):,} unique nodes.")

    top_nodes: set[str] = {
        word for word, _ in
        sorted(totals.items(), key=lambda x: x[1], reverse=True)[:n_nodes]
    }
    print(f"Selected top {len(top_nodes):,} nodes by total edge weight.")

    # ----------------------------- keep edges where both endpoints are in set
    kept = [
        row for row in rows
        if (
            len(row) > max(cue_idx, resp_idx, str_idx)
            and row[cue_idx].strip().lower()  in top_nodes
            and row[resp_idx].strip().lower() in top_nodes
        )
    ]
    print(f"Kept {len(kept):,} edges (both endpoints in top set).")

    # ----------------------------------------------------------------- write
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_NONE,
                            escapechar="\\")
        writer.writerow(header)
        writer.writerows(kept)

    print(f"Reduced file written to '{output_path}'.")

    # ---------------------------------------------------------------- report
    surviving_nodes = (
        {row[cue_idx].strip().lower()  for row in kept} |
        {row[resp_idx].strip().lower() for row in kept}
    )
    print(f"\nSummary")
    print(f"  Nodes in output : {len(surviving_nodes):,}")
    print(f"  Edges in output : {len(kept):,}")
    print(f"  Edges removed   : {len(rows) - len(kept):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reduce SWOW data to the top-N nodes by total edge weight."
    )
    parser.add_argument("--data",   required=True, help="Path to input SWOW TSV")
    parser.add_argument("--output", required=True, help="Path for reduced output TSV")
    parser.add_argument(
        "--n", type=int, default=10_000,
        help="Number of top nodes to keep (default: 10000)"
    )
    parser.add_argument(
        "--strength-col", default="R123.Strength",
        help="Strength column name (default: R123.Strength)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reduce_graph(
        input_path   = args.data,
        output_path  = args.output,
        n_nodes      = args.n,
        strength_col = args.strength_col,
    )


if __name__ == "__main__":
    main()
