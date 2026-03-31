"""
swow_activation.py
------------------
Single-step spreading activation logic for the Small World of Words (SWOW)
dataset.

Provides:
    load_graph()         — parse a SWOW associative-strength TSV into a
                           weighted adjacency dict
    activation_step()    — advance one step from the current node using
                           associative-strength-weighted sampling, with
                           dead-end lookahead
"""

import csv
import random
import sys
from collections import defaultdict


def load_graph(filepath: str, strength_col: str = "R123.Strength") -> dict[str, dict[str, float]]:
    """
    Parse a SWOW associative-strength TSV into a weighted adjacency dict.

    Parameters
    ----------
    filepath : str
        Path to the SWOW strength file
        (e.g. strength.SWOW-EN.R123.20180827.csv).
    strength_col : str
        Column name to use as edge weight. Defaults to "R123.Strength".
        Use "R1.Strength" for first-response-only weights.

    Returns
    -------
    dict[str, dict[str, float]]
        graph[cue][response] = associative_strength
    """
    graph: dict[str, dict[str, float]] = defaultdict(dict)

    # SWOW files are tab-separated and contain literal quote characters that
    # must NOT be treated as CSV quoting (per the official SWOW loading note).
    with open(filepath, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)

        try:
            cue_idx  = header.index("cue")
            resp_idx = header.index("response")
            str_idx  = header.index(strength_col)
        except ValueError:
            available = ", ".join(header)
            sys.exit(
                f"Column '{strength_col}' not found in file.\n"
                f"Available columns: {available}"
            )

        for row in reader:
            if len(row) <= max(cue_idx, resp_idx, str_idx):
                continue
            cue      = row[cue_idx].strip().lower()
            response = row[resp_idx].strip().lower()
            try:
                strength = float(row[str_idx])
            except ValueError:
                continue
            if strength > 0 and response not in ("", "nan"):
                graph[cue][response] = strength

    return dict(graph)


def activation_step(
    graph: dict[str, dict[str, float]],
    current: str,
    dead_end_responses: dict[str, set[str]],
    rng: random.Random,
) -> tuple[str | None, bool]:
    """
    Advance one spreading activation step from `current`.

    Samples the next node proportionally to associative strength, with a
    lookahead check: if the sampled node has no outgoing edges it is marked
    as a dead end for `current` and None is returned so the caller can retry
    the same step.  If `current` itself has no viable neighbors, None is also
    returned and `exhausted` is set to True so the caller can backtrack.

    Parameters
    ----------
    graph : dict[str, dict[str, float]]
        Weighted adjacency dict produced by load_graph().
    current : str
        The node to step away from.
    dead_end_responses : dict[str, set[str]]
        Mutable mapping of node -> set of neighbours already identified as
        dead ends.  Updated in-place when a dead end is discovered.
    rng : random.Random
        Random number generator (pass a seeded instance for reproducibility).

    Returns
    -------
    next_node : str | None
        The chosen next node, or None if no valid step was possible.
    exhausted : bool
        True when `current` has no viable neighbours at all (caller should
        backtrack); False when a dead-end candidate was simply skipped (caller
        should retry the step from the same node).
    """
    neighbors = graph.get(current, {})
    viable = {r: w for r, w in neighbors.items()
              if r not in dead_end_responses[current]}

    if not viable:
        # Current node is fully exhausted — signal the caller to backtrack
        return None, True

    responses = list(viable.keys())
    weights   = [viable[r] for r in responses]
    (next_node,) = rng.choices(responses, weights=weights, k=1)

    if not graph.get(next_node):
        # Lookahead: sampled node is a dead end — mark and signal retry
        dead_end_responses[current].add(next_node)
        return None, False

    return next_node, False
