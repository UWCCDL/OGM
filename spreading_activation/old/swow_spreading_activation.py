"""
swow_spreading_activation.py
----------------------------
Spreading activation traversal over the Small World of Words (SWOW) dataset.

Usage:
    python swow_spreading_activation.py --data <path_to_strength_file> \
                                        --cue <cue_word> \
                                        --n <retrieval_length> \
                                        [--R1 | --R123]  (default: R123)
                                        [--seed <int>]   (optional, for reproducibility)

Expected input file:
    The SWOW associative strength TSV, e.g.:
        strength.SWOW-EN.R1.csv   (first response only)
        strength.SWOW-EN.R123.csv (all three responses)

    The file has (at minimum) these columns:
        cue         - the stimulus word
        response    - the associate given
        R1 / R123   - associative strength (conditional probability of the
                      response given the cue, across participants)

    Load the file with the quoting fix recommended in the SWOW documentation:
        read_delim(..., quote='', escape_backslash=F, escape_double=F)
    The equivalent is handled below via csv.reader with no quoting.

Algorithm:
    Starting from the cue word, at each step:
      1. Look up all outgoing edges from the current node (cue -> responses).
      2. Use their associative strengths as a probability distribution.
      3. Sample the next node proportionally to those strengths.
      4. Record the visited node and move to it.
    Repeat for n steps.

    This is a biased random walk — the same node can be revisited.
"""

import argparse
import csv
import random
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_graph(filepath: str, strength_col: str = "R123") -> dict[str, dict[str, float]]:
    """
    Parse the SWOW associative-strength TSV into an adjacency dict.

    Returns:
        graph[cue][response] = associative_strength (float)
    """
    graph: dict[str, dict[str, float]] = defaultdict(dict)

    # SWOW files use tab separators and contain internal quotes that should
    # NOT be treated as quoting characters (per the official R loading note).
    with open(filepath, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)

        try:
            cue_idx = header.index("cue")
            resp_idx = header.index("response")
            str_idx = header.index(strength_col)
        except ValueError:
            available = ", ".join(header)
            sys.exit(
                f"Column '{strength_col}' not found.\n"
                f"Available columns: {available}\n"
                f"Use --R1 to switch to the R1 strength column."
            )

        for row in reader:
            if len(row) <= max(cue_idx, resp_idx, str_idx):
                continue  # skip malformed rows
            cue = row[cue_idx].strip().lower()
            response = row[resp_idx].strip().lower()
            try:
                strength = float(row[str_idx])
            except ValueError:
                continue  # skip rows with non-numeric strength
            if strength > 0 and response not in ("", "nan"):
                graph[cue][response] = strength

    print(f"Graph loaded: {len(graph):,} cue nodes, "
          f"{sum(len(v) for v in graph.values()):,} total edges.")
    return dict(graph)


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------

def spreading_activation_walk(
    graph: dict[str, dict[str, float]],
    cue: str,
    n: int,
    rng: random.Random,
) -> list[str]:
    """
    Traverse n steps from `cue` using associative-strength-weighted sampling.

    Dead-end handling: if the walk reaches a node with no outgoing edges,
    it backtracks through the path history until it finds a node that has
    at least one neighbor not already known to be a dead end, then continues
    from there. If backtracking exhausts the entire path, the walk stops early.

    Returns:
        A list of visited nodes of length n+1 (includes the starting cue).
    """
    cue = cue.strip().lower()
    if cue not in graph:
        close = [k for k in graph if k.startswith(cue[:3])][:5]
        hint = f"  Did you mean one of: {close}" if close else ""
        sys.exit(f"Cue word '{cue}' not found in graph.{hint}")

    path = [cue]
    # Track which neighbors have already been ruled out as dead ends,
    # keyed by the node they were reached from.
    dead_end_responses: dict[str, set[str]] = defaultdict(set)

    step = 0
    while step < n:
        current = path[-1]
        neighbors = graph.get(current, {})

        # Filter out neighbors already known to be dead ends
        viable = {r: w for r, w in neighbors.items()
                  if r not in dead_end_responses[current]}

        if not viable:
            # No viable neighbors — mark current as dead end from parent,
            # then backtrack without counting this as a step
            print(f"  '{current}' is a dead end — backtracking...")
            if len(path) == 1:
                print(f"  Could not recover from dead end at starting cue '{current}'. "
                      f"Walk ends early.")
                break
            parent = path[-2]
            dead_end_responses[parent].add(current)
            path.pop()
            continue

        # Lookahead: sample a candidate and check it has outgoing edges.
        # If not, mark it as a dead end and retry without consuming a step.
        responses = list(viable.keys())
        weights = [viable[r] for r in responses]
        (next_node,) = rng.choices(responses, weights=weights, k=1)

        if not graph.get(next_node):
            print(f"  '{next_node}' (sampled from '{current}') is a dead end — skipping...")
            dead_end_responses[current].add(next_node)
            continue

        path.append(next_node)
        step += 1

    return path


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_path(path: list[str], graph: dict[str, dict[str, float]]) -> None:
    print("\n" + "=" * 55)
    print(f"  Spreading activation walk  ({len(path)-1} steps)")
    print("=" * 55)
    for i, node in enumerate(path):
        if i == 0:
            print(f"  [START]  {node}")
        else:
            prev = path[i - 1]
            strength = graph.get(prev, {}).get(node)
            strength_str = f"  (p={strength:.4f})" if strength is not None else ""
            print(f"  step {i:>3}  {node}{strength_str}")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spreading activation random walk over the SWOW dataset."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to the SWOW associative strength TSV "
             "(e.g. strength.SWOW-EN.R123.csv)"
    )
    parser.add_argument(
        "--cue", required=True,
        help="Starting cue word for the walk"
    )
    parser.add_argument(
        "--n", required=True, type=int,
        help="Number of steps (retrieval length)"
    )
    strength_group = parser.add_mutually_exclusive_group()
    strength_group.add_argument(
        "--R1", dest="strength_col", action="store_const", const="R1.Strength",
        help="Use first-response associative strengths (R1.Strength column)"
    )
    strength_group.add_argument(
        "--R123", dest="strength_col", action="store_const", const="R123.Strength",
        help="Use all-response associative strengths (R123.Strength column, default)"
    )
    parser.set_defaults(strength_col="R123.Strength")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n < 1:
        sys.exit("--n must be a positive integer.")

    rng = random.Random(args.seed)

    print(f"\nLoading graph from: {args.data}")
    print(f"Strength column   : {args.strength_col}")
    graph = load_graph(args.data, strength_col=args.strength_col)

    print(f"\nStarting walk from '{args.cue}' for {args.n} step(s)...")
    path = spreading_activation_walk(graph, cue=args.cue, n=args.n, rng=rng)

    print_path(path, graph)


if __name__ == "__main__":
    main()
