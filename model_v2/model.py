"""OGM model v2: functional avoidance in cued autobiographical retrieval.

Retrieval is a walk through an associative memory network with an AMT-style
hierarchy (Memory): a cue activates general-event summary nodes, from which
the search can descend to specific events. At every step an RL agent weighs
the value of searching on against the payoff of reporting what is in mind
now, so each retrieval ends as a specific memory, a categoric (general)
memory, or an omission.

Agents sit on one spectrum (FeatureAgent): a successor-feature learner whose
content_share sets how its reward learning (total rate alpha_w) is split
between memory content and "retrieval in general" (a bias feature active
everywhere) - i.e. where the blame for an aversive memory lands.
content_share = 1 keeps value attached to content; content_share = 0 lets
only retrieval-in-general carry value, which behaves like the single-value
pole, SingleValueAgent. Value decay (decay_rate) lets learned avoidance fade.

Simulator runs retrievals, including a healthy warm_up before trauma onset.
experiments.py runs the experiments; analysis.ipynb visualises them.
"""
import random
import rustworkx as rx
import numpy as np
import math
from collections import Counter, defaultdict


class Memory:
    def __init__(self, n, rewards, state=None, thres=0,
                 n_features=50, features_per_node=5, num_groups=0, group_features=0,
                 max_assoc_strength=1.6, sigma=0.1, var=1, trauma=(None, 0),
                 summary_reward=0, summary_report=0.5, event_report=0.5):
        self.n_features = n_features
        self.features_per_node = features_per_node
        self.group_features = group_features
        self.max_assoc_strength = max_assoc_strength
        self.var = var
        self.summary_reward = summary_reward
        # Payoff of *stopping and reporting* what is in mind (the stop
        # option the agent weighs against searching on), distinct from the
        # rewards above, which are what the agent experiences on *visiting* a
        # memory. Nothing to report at the cue (omission) is worth 0.
        self.summary_report = summary_report
        self.event_report = event_report

        # Memory hierarchy (Conway's Self-Memory System): each group is a
        # general event. Its member nodes are specific events, and - when the
        # group has defining features - one extra "summary" node carries only
        # those shared features, standing for the categoric memory ("times at
        # the park"). node_level / node_group label every node; groups maps a
        # group id to its defining features and summary node index.
        self.node_level = {}
        self.node_group = {}
        self.groups = {}
        self.n_groups = num_groups
        self.cue = None
        self.cue_group = None

        # Group-defining features live in their own index range, starting
        # right after the general n_features pool, so they never collide
        # with the randomly-sampled individual features below. total_features
        # tracks how much of that combined space is allocated so far - it
        # grows every time a new group is created (here or in add_memories).
        self.total_features = n_features

        # Each node gets a random subset of the general feature vocabulary,
        # plus - if it belongs to one of num_groups groups - group_features
        # features exclusive to that group, which is what makes its
        # connections to the rest of the group especially strong.
        # feature_fan tracks, for every feature, how many nodes in the whole
        # network currently carry it - this is the ACT-R "fan" used below.
        self.node_features = {}
        self.feature_fan = Counter()
        self.feature_nodes = defaultdict(list)  # feature -> nodes carrying it
        group_size = n // num_groups if num_groups else 0
        group_blocks = [self._new_group_features() for _ in range(num_groups)]
        for node in range(n):
            if group_size and node < group_size * num_groups:
                gid = node // group_size
                group_feats = group_blocks[gid]
            else:
                gid = None
                group_feats = set()
            individual_count = max(features_per_node - len(group_feats), 0)
            feats = group_feats | set(random.sample(range(n_features), individual_count))
            self._set_features(node, feats)
            self.node_level[node] = "event"
            self.node_group[node] = gid

        # Summary nodes go after the n event nodes so event indices stay
        # 0..n-1 and reward dicts keyed by them still line up. A summary node
        # shares only its group's exclusive features, so _edge_weight links it
        # to its own group's events and nothing else.
        if group_features > 0:
            for gid, group_feats in enumerate(group_blocks):
                self._register_summary(n + gid, gid, group_feats)

        total = n + len(self.groups)
        graph = rx.PyGraph()
        graph.add_nodes_from([[1]] * total)
        edges = []
        for i in range(total):
            for j in range(i + 1, total):
                w = self._edge_weight(self.node_features[i], self.node_features[j])
                if w is not None:
                    edges.append((i, j, w))
        graph.add_edges_from(edges)
        # Normalise trauma[0] to a set so multiple nodes can be tracked.
        raw_nodes = trauma[0]
        if raw_nodes is None:
            trauma_nodes = set()
        elif isinstance(raw_nodes, int):
            trauma_nodes = {raw_nodes}
        else:
            trauma_nodes = set(raw_nodes)
        self.trauma = (trauma_nodes, trauma[1])
        self.graph = graph
        self.state = state
        self.thres = thres
        self.current_graph = graph.copy()
        self.sigma = sigma
        self.state = state
        # Copy so add_memories / summary rewards never mutate the caller's
        # dict (callers may reuse one rewards dict across environments).
        self.rewards = dict(rewards)
        for info in self.groups.values():
            self.rewards[info["summary"]] = summary_reward

    def _register_summary(self, idx, gid, group_feats):
        """Record node idx as the summary (categoric) node of group gid."""
        self._set_features(idx, set(group_feats))
        self.node_level[idx] = "summary"
        self.node_group[idx] = gid
        self.groups[gid] = {"features": set(group_feats), "summary": idx}

    def _set_features(self, idx, feats):
        """Give node idx its feature set, updating the fan counts and the
        feature -> nodes index used to find edge candidates."""
        self.node_features[idx] = feats
        self.feature_fan.update(feats)
        for f in feats:
            self.feature_nodes[f].append(idx)

    def _new_group_features(self):
        """Allocate a fresh block of group_features indices, exclusive to one
        group, from the growing group-feature pool beyond n_features."""
        start = self.total_features
        self.total_features += self.group_features
        return set(range(start, self.total_features))

    def _edge_weight(self, features_a, features_b):
        """ACT-R fan-effect strength of association between two feature sets.

        For each feature shared by both nodes, the associative strength is
        S_ji = max_assoc_strength - ln(fan(feature)), where fan(feature) is
        the number of nodes across the whole network currently carrying that
        feature (Anderson & Lebiere's spreading-activation equation). Strengths
        from all shared features are summed, so nodes sharing more - or
        rarer/more distinctive - features end up more strongly connected.
        Returns None if there is no shared feature or the summed strength
        (plus noise) is non-positive, meaning no edge is created.
        """
        shared = features_a & features_b
        if not shared:
            return None
        strength = sum(
            self.max_assoc_strength - math.log(self.feature_fan[f]) for f in shared
        ) + random.gauss(0, self.var)
        return strength if strength > 0 else None

    def add_memories(self, n, a=1, trauma=False):
        """Add n new memories to the graph as a single cohesive batch.

        The batch automatically forms its own group, exactly like the
        num_groups groups created in __init__: every node in it shares
        group_features features exclusive to this batch (drawn from the same
        growing group-feature pool), plus its own random individual features
        filling out features_per_node. Sharing those group features is what
        makes intra-batch edges strong - edge weights (both within the batch
        and to existing nodes) are computed the same way as in __init__, via
        the ACT-R fan-effect formula in ``_edge_weight``.

        Parameters
        ----------
        n : int
            Number of new memory nodes to add.
        a : float
            Initial activation payload for every new node. Defaults to 1.
        trauma : float or False
            When not False, the first node in the new batch is designated as a
            trauma memory.  The value is used as that node's reward (e.g.
            pass ``trauma=-10`` for a strongly aversive memory).  The node is
            added to the ``self.trauma[0]`` set so that ``spreading_activation``
            automatically applies the existing boost (``self.trauma[1]``) to it
            alongside all previously registered trauma nodes.
        """
        existing_indices = list(self.graph.node_indices())

        # All nodes receive the standard activation payload.
        new_indices = list(self.graph.add_nodes_from([[a]] * n))

        group_feats = self._new_group_features()
        gid = self.n_groups
        self.n_groups += 1
        individual_count = max(self.features_per_node - len(group_feats), 0)
        for idx in new_indices:
            feats = group_feats | set(
                random.sample(range(self.n_features), individual_count)
            )
            self._set_features(idx, feats)
            self.node_level[idx] = "event"
            self.node_group[idx] = gid

        # The batch's summary (categoric) node, as in __init__.
        batch = list(new_indices)
        if group_feats:
            summary_idx = self.graph.add_node([a])
            self._register_summary(summary_idx, gid, group_feats)
            batch.append(summary_idx)

        # Intra-batch edges.
        intra_edges = []
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                w = self._edge_weight(self.node_features[batch[i]],
                                       self.node_features[batch[j]])
                if w is not None:
                    intra_edges.append((batch[i], batch[j], w))
        self.graph.add_edges_from(intra_edges)

        # Cross edges between new batch and existing nodes. Only nodes sharing
        # a feature can be linked (_edge_weight returns None, without drawing
        # noise, otherwise), so candidates come from the feature -> nodes
        # index, in ascending order - same edges and random draws as looping
        # over every existing node, without the O(N) scan per new node.
        existing = set(existing_indices)
        cross_edges = []
        for ni in batch:
            candidates = set()
            for f in self.node_features[ni]:
                candidates.update(self.feature_nodes[f])
            for ei in sorted(candidates & existing):
                w = self._edge_weight(self.node_features[ni], self.node_features[ei])
                if w is not None:
                    cross_edges.append((ni, ei, w))
        self.graph.add_edges_from(cross_edges)

        # Update rewards: all new memories get a reward of 1 by default;
        # the summary node gets summary_reward.
        self.rewards.update({idx: 1 for idx in new_indices})
        if group_feats:
            self.rewards[summary_idx] = self.summary_reward

        # If a trauma memory was requested, override the first node's reward
        # and add it to the trauma node set.
        if trauma is not False:
            trauma_node_idx = new_indices[0]
            self.rewards[trauma_node_idx] = trauma
            self.trauma[0].add(trauma_node_idx)

    def visualize(self, title="Memory Network", ax=None, min_strength=0.0):
        """Visualize the current graph using a force-directed (spring) layout.

        Parameters
        ----------
        min_strength : float
            Nodes whose sum_t (sum of trace ** -0.5 over all traces) falls
            below this threshold are hidden. Higher values show only the
            most recently / frequently recalled memories. Defaults to 0
            (show all nodes).

        Visual encodings
        ----------------
        Node size   : sum_t = sum(trace ** -0.5) — larger when recalled more
                      recently or frequently, matching the activation formula.
        Node colour : steel blue for normal nodes, crimson for the trauma node.
        Node border : thick black ring marks the currently active state.
        Edge width  : proportional to edge weight — stronger connections are thicker.
        Edge alpha  : also proportional to weight, so weak cross-edges recede visually.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        graph = self.graph
        all_nodes = graph.node_indices()
        if not all_nodes:
            return

        # --- compute sum_t per node and apply strength filter ------------------
        def sum_t(n):
            traces = graph[n]
            if not traces:
                return 0.0
            return sum(trace ** -0.5 for trace in traces)

        nodes = [n for n in all_nodes if sum_t(n) >= min_strength]
        if not nodes:
            return

        # --- layout (on the full graph so positions are stable) ----------------
        pos = rx.graph_spring_layout(
            graph,
            weight_fn=lambda w: float(w),
            num_iter=200,
            seed=42,
        )  # {node_idx: (x, y)}

        # --- derived per-node quantities ---------------------------------------
        trauma_nodes = self.trauma[0]
        trauma_present = any(n in pos and n in nodes for n in trauma_nodes)

        node_xy    = np.array([pos[n] for n in nodes])
        node_sizes = np.array([max(sum_t(n), 1e-9) * 120 for n in nodes])
        summary_present = any(self.node_level.get(n) == "summary" for n in nodes)
        node_colors = [
            "crimson" if n in trauma_nodes
            else "goldenrod" if self.node_level.get(n) == "summary"
            else "steelblue"
            for n in nodes
        ]

        # --- derived per-edge quantities (only edges between visible nodes) ----
        visible = set(nodes)
        edge_list = [(u, v, w) for u, v, w in graph.weighted_edge_list()
                     if u in visible and v in visible]
        if edge_list:
            weights = np.array([float(w) for _, _, w in edge_list])
            w_max = weights.max() if weights.max() > 0 else 1.0
            edge_widths = 0.5 + 3.5 * (weights / w_max)
            edge_alphas = 0.15 + 0.75 * (weights / w_max)
        else:
            edge_widths = edge_alphas = []

        # --- draw --------------------------------------------------------------
        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Edges
        for (u, v, _), lw, alpha in zip(edge_list, edge_widths, edge_alphas):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1],
                    color="steelblue", linewidth=lw, alpha=alpha, zorder=1)

        # Nodes
        ax.scatter(
            node_xy[:, 0], node_xy[:, 1],
            s=node_sizes,
            c=node_colors,
            edgecolors="black",
            linewidths=[3.0 if n == self.state else 0.8 for n in nodes],
            zorder=3,
        )

        # --- legend ------------------------------------------------------------
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
                   markersize=9, markeredgecolor="black", label="memory"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
                   markersize=12, markeredgecolor="black", linewidth=2.5,
                   label="current state"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                   markersize=6, markeredgecolor="black",
                   label="size ∝ memory strength"),
        ]
        if trauma_present:
            legend_elements.insert(1,
                Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                       markersize=9, markeredgecolor="black", label="trauma node")
            )
        if summary_present:
            legend_elements.insert(1,
                Line2D([0], [0], marker="o", color="w", markerfacecolor="goldenrod",
                       markersize=9, markeredgecolor="black", label="summary (categoric)")
            )
        ax.legend(handles=legend_elements, loc="upper left",
                  fontsize=8, framealpha=0.85)

        if own_fig:
            plt.tight_layout()
            plt.show()

    def initialize_state(self, cue=None):
        """Start a retrieval from a cue, as in the Autobiographical Memory Test.

        The cue is a group's defining features. Retrieval starts *before* any
        memory (state = None); the first spreading_activation step goes from
        the cue to the summary nodes (generative retrieval reaches the general
        level first), and the agent can then descend to specific events.

        cue : None picks a random existing group; an int cues that group id.

        Networks without summary nodes (no groups, or group_features=0) fall
        back to the old behaviour: start at a uniformly random node.
        """
        if not self.groups:
            self.cue = None
            self.cue_group = None
            self.state = random.choice(list(self.graph.node_indices()))
            return
        if cue is None:
            cue = random.choice(list(self.groups))
        self.cue_group = cue
        self.cue = set(self.groups[cue]["features"])
        self.state = None

    def current_features(self):
        """Features of what is currently in mind: the cue before any memory
        has been reached, otherwise the current node's features."""
        return self.cue if self.state is None else self.node_features[self.state]

    def report_value(self):
        """Payoff of stopping now: 0 at the cue (omission), summary_report at
        a summary node (categoric answer), event_report at an event node
        (specific answer)."""
        if self.state is None:
            return 0.0
        if self.node_level.get(self.state) == "summary":
            return self.summary_report
        return self.event_report

    def _cue_strength(self, features):
        """Fan-effect strength from the cue to a node (same S_ji formula as
        _edge_weight, without the structural edge noise)."""
        shared = self.cue & features
        if not shared:
            return None
        strength = sum(self.max_assoc_strength - math.log(self.feature_fan[f]) for f in shared)
        return strength if strength > 0 else None

    def spreading_activation(self):
        graph = self.current_graph
        if self.state is None:
            # From the cue, activation reaches only summary nodes. Nothing has
            # been removed from current_graph yet, so all of them are present.
            adj = {}
            for info in self.groups.values():
                s = self._cue_strength(self.node_features[info["summary"]])
                if s is not None:
                    adj[info["summary"]] = s
        else:
            adj = graph.adj(self.state)
        neighbors = list(adj.keys())
        activations = []
        for j in neighbors:
            sum_t = sum(trace ** -0.5 for trace in graph[j])

            activation = adj[j] + math.log(sum_t) + random.gauss(0, self.sigma)
            if j in self.trauma[0]:
                activation += self.trauma[1]
            if activation > self.thres:
                activations.append(activation)
            else:
                activations.append(0)
        if sum(activations) == 0:
            return False
        next_state = random.choices(neighbors, weights=activations, k=1)[0]
        self.graph[next_state] = self.graph[next_state] + [1]
        if self.state is not None:
            graph.remove_node(self.state)
        return next_state

    def decay(self, time):
        for node in self.graph.node_indices():
            self.graph[node] = [trace + time for trace in self.graph[node]]


class FeatureAgent:
    """Successor-feature agent over the network's per-node feature sets.

    Generalises the group-based SR agent to overlapping features: a node's
    active features are its multi-hot state representation phi(s) (1 for
    each feature it carries, 0 otherwise), psi is learned per feature via
    linear TD(0), and a node's psi(s)/value are the sum over its active
    features' rows/weights - this collapses to the original one-hot,
    per-group formulas exactly when every node carries a single feature.

    Agent-only features (indices 0..N_AGENT-1; network feature k is index
    k + N_AGENT) - they live only in the agent's representation, never in
    Memory.node_features, so they don't touch fan or edge weights:

    * BIAS is on in every memory state (and the cue), like a regression
      intercept: its weight learns the reward common to all memories, so
      never-seen groups inherit a baseline value instead of 0. With *only*
      the bias, every state looks the same and this is a single-value agent;
      the real features add content-specific adjustments on top. Trauma
      blame is split between the bias (generalised) and the trauma group's
      features (specific). bias=False drops it; content=False drops the
      memory-content features instead, leaving the bias alone - see
      SingleValueAgent.
    * R_CAT / R_SPEC are terminal "reported a categoric / specific memory"
      pseudo-states. Their weights learn the report payoffs and psi carries
      them, so the value of searching includes the payoff retrieval is
      expected to end with.

    Two successor matrices: psi is the on-policy SR (target
    phi(s) + gamma * psi(s'), or phi(s) + gamma * phi(R) when retrieval ends
    with a report at s); psi_c is the successor representation *given the
    agent continues* (target gamma * psi(s'), learned only on transitions
    where it continued), so w @ psi_c(s) is the value of continuing, without
    the payoff of stopping at s mixed in.
    """

    BIAS, R_CAT, R_SPEC = 0, 1, 2
    N_AGENT = 3

    def __init__(self, network, alpha_psi=0.1, alpha_w=0.1, gamma=0.9, temp=0.1, decay_rate=0.0,
                 bias=True, content=True, alpha_bias=None, alpha_content=None,
                 content_share=None):
        if not (bias or content):
            raise ValueError("FeatureAgent needs bias and/or content features")
        self.network = network
        self.content = content
        # The spectrum parameter: split one total reward-learning rate
        # (alpha_w) between content features (content_share) and the bias
        # (1 - content_share). It sets alpha_bias / alpha_content, which can
        # also be given directly instead.
        if content_share is not None:
            if alpha_bias is not None or alpha_content is not None:
                raise ValueError("give content_share or alpha_bias/alpha_content, not both")
            if not 0 <= content_share <= 1:
                raise ValueError("content_share must be in [0, 1]")
            alpha_bias = (1 - content_share) * alpha_w
            alpha_content = content_share * alpha_w
        self.content_share = content_share
        # Reward-learning rate of the bias weight: how fast blame (or credit)
        # spreads to "retrieval in general" rather than staying with the
        # content features. None -> alpha_w (bias learns like any feature);
        # 0 -> the bias only helps predict what follows (psi), never carries
        # value itself.
        self.alpha_bias = alpha_w if alpha_bias is None else alpha_bias
        # Reward-learning rate of the memory-content features: how much
        # value (and trauma blame) can attach to *what* a memory is about.
        # None -> alpha_w. With alpha_bias = alpha_w, lowering it walks the
        # agent toward the SV pole; at 0 only the bias carries value, as in
        # SingleValueAgent, though psi still learns from content (so search
        # length and report odds can still differ between groups). The
        # report-payoff weights (R_CAT / R_SPEC) always learn at alpha_w.
        self.alpha_content = alpha_w if alpha_content is None else alpha_content
        self.gamma = gamma
        self.temp = temp
        self.alpha_psi = alpha_psi
        self.alpha_w = alpha_w
        self.decay_rate = decay_rate
        self.bias = bias
        # total_features (not n_features) since group_features, if used,
        # live in their own index range beyond the general n_features pool.
        n = self._n_needed()
        self.psi = np.zeros((n, n))    # psi_k: expected discounted future feature occupancy attributable to feature k
        self.psi_c = np.zeros((n, n))  # same, conditional on continuing the search
        self.w = np.zeros(n)           # reward weight per feature
        self._prev = None     # (indices, reward) of a state the agent continued from, awaiting its successor
        self._current = None  # (indices, reward) of the state the agent is in (reward None at the cue)

    def _n_needed(self):
        # Without content features only the agent-only indices are ever used.
        if not self.content:
            return self.N_AGENT
        return self.network.total_features + self.N_AGENT

    def _idx(self, features):
        idx = [f + self.N_AGENT for f in features] if self.content else []
        if self.bias:
            idx.append(self.BIAS)
        return idx

    def _learn_w(self, idx, target):
        """Delta-rule update of the reward weights of the active features idx
        toward target: each gets the same prediction error, scaled by its own
        learning rate (alpha_bias for the bias, alpha_w for the report
        payoffs, alpha_content for memory-content features)."""
        err = target - self.w[idx].sum()
        for i in idx:
            if i == self.BIAS:
                rate = self.alpha_bias
            elif i < self.N_AGENT:
                rate = self.alpha_w
            else:
                rate = self.alpha_content
            self.w[i] += rate * err

    def _phi(self, idx):
        v = np.zeros(len(self.w)); v[idx] = 1.0; return v

    def _psi_of(self, idx):
        return self.psi[idx].sum(axis=0)

    def _psi_c_of(self, idx):
        return self.psi_c[idx].sum(axis=0)

    GROW_CHUNK = 256

    def _grow_to(self, n):
        """Pad psi/psi_c/w with zeros if the network has allocated new
        features (e.g. add_memories creating another group) since __init__.
        Grows in chunks so the matrices aren't copied on every retrieval; the
        extra rows/columns stay zero and never affect a value."""
        if n <= len(self.w):
            return
        pad = max(n - len(self.w), self.GROW_CHUNK) if self.content else n - len(self.w)
        if pad > 0:
            self.psi = np.pad(self.psi, ((0, pad), (0, pad)))
            self.psi_c = np.pad(self.psi_c, ((0, pad), (0, pad)))
            self.w = np.pad(self.w, (0, pad))

    def policy(self, r, report=0.0):
        """Decide whether to keep searching (True) or stop here (False).

        Continue with probability sigmoid((value of continuing - report) /
        temp), where report is the payoff of stopping here and reporting what
        is in mind (Memory.report_value(): 0 at the cue, i.e. an omission). r
        is the reward of the memory in mind, or None at the cue."""
        self._grow_to(self._n_needed())
        at_cue = self.network.state is None
        i1 = self._idx(self.network.current_features())
        if self._prev is not None:
            ip, r_prev = self._prev
            nxt = self._psi_of(i1)
            self.psi[ip] += self.alpha_psi * (self._phi(ip) + self.gamma * nxt - self._psi_of(ip))
            self.psi_c[ip] += self.alpha_psi * (self.gamma * nxt - self._psi_c_of(ip))
            self._learn_w(ip, r_prev)

        # From the cue, continuing means arriving at the summary node, whose
        # features are the cue's, so its full on-policy value is what
        # continuing is worth. Elsewhere, use the continue-conditional SR.
        q = self.w @ (self._psi_of(i1) if at_cue else self._psi_c_of(i1))
        x = (q - report) / self.temp
        # For very negative x, e ** (-x) overflows; the sigmoid is ~0 there.
        recall = x >= -50 and 1 / (1 + np.e ** (-x)) > random.random()
        self._current = (i1, r)
        # r is None at the cue: no reward was experienced there, so the
        # cue -> first-memory transition isn't learned from.
        self._prev = (i1, r) if (recall and r is not None) else None
        return recall

    def p_continue(self, features, report, at_cue=False):
        """(value of continuing, probability policy() would continue) from a
        state with these features - the same formula as policy(), but with
        no learning, no randomness and no change to the agent, so it can be
        used to inspect a run without altering it. Features the agent has
        not allocated yet count as zero, as they would in policy()."""
        idx = [i for i in self._idx(features) if i < len(self.w)]
        q = float(self.w @ (self._psi_of(idx) if at_cue else self._psi_c_of(idx)))
        x = (q - report) / self.temp
        return q, (0.0 if x < -50 else float(1 / (1 + np.exp(-x))))

    def end_retrieval(self, payoff, outcome=None):
        """Learn from how the retrieval ended. For a reported memory (payoff
        not None; outcome "categoric" or "specific"), the state it ended on is
        terminal: its reward is learned, psi learns that it leads to the
        matching report pseudo-state, and that pseudo-state's weight learns
        the payoff. An omission (payoff None) teaches nothing. Also clears
        pending state so no transition is learned across retrievals."""
        cur = self._current
        self._prev = None
        self._current = None
        if payoff is None or cur is None or cur[1] is None:
            return
        i1, r = cur
        rep = self.R_CAT if outcome == "categoric" else self.R_SPEC
        target = self._phi(i1)
        target[rep] += self.gamma
        self.psi[i1] += self.alpha_psi * (target - self._psi_of(i1))
        self._learn_w(i1, r)
        self._learn_w([rep], payoff)

    def decay(self):
        """Relax the learned reward weights back toward zero (value decay),
        analogous to Memory.decay's aging of activation traces. Only w (the
        reward weight per feature, including the bias and report payoffs)
        decays - psi (the learned
        successor-feature dynamics) is left untouched, playing the same role
        the graph's edge structure plays for Memory.decay: structural
        knowledge that doesn't fade just because reward associations do."""
        self.w *= (1 - self.decay_rate)


class SingleValueAgent(FeatureAgent):
    """Single-value (SV) agent, nested inside FeatureAgent.

    A FeatureAgent whose state representation is the bias feature alone, so
    every memory (and the cue) looks the same and the agent holds one value
    of searching. It shares everything else with the successor-feature agent
    - TD rule, gamma, learning rates, report-payoff learning, decay - so the
    SV vs SF comparison differs only in whether value can attach to memory
    content. It is the pole of the content_share spectrum: content
    carries no value, only "retrieval in general" does. (The v1 Agent, with
    its different update rule, lives in model_v1/model.py.)
    """

    def __init__(self, network, **kwargs):
        kwargs.pop("content", None)
        kwargs.pop("bias", None)
        # Its single feature carries all learning, at the same rate as the
        # SF agent's content features (alpha_w) - alpha_bias doesn't apply.
        kwargs.pop("alpha_bias", None)
        kwargs.pop("alpha_content", None)
        kwargs.pop("content_share", None)
        super().__init__(network, bias=True, content=False, **kwargs)
        self.alpha_bias = self.alpha_w


class Simulator:
    def __init__(self, agent, network):
        self.agent = agent
        self.network = network
        self.states_visited = []
        self.warmup = None  # warm_up() moves its trials' logs here
        self._reset_logs()

    # Per-retrieval logs, all parallel to each other (one entry per retrieval).
    LOGS = ("record", "reward_totals", "trauma_encounters",
            "cue_groups", "responses", "outcomes", "stop_causes")

    def _reset_logs(self):
        self.record = []
        self.reward_totals = []
        self.trauma_encounters = []
        # Per-retrieval AMT-style coding (parallel to record):
        #   cue_groups - group id that cued the retrieval (None: random start)
        #   responses  - node in mind when retrieval ended (None: nothing reached)
        #   outcomes   - "specific" (event node), "categoric" (summary node),
        #                or "omission" (no memory reached)
        #   stop_causes - "agent_stop" (the RL policy chose to stop: functional
        #                avoidance), "dead_end" (no neighbour cleared the
        #                activation threshold), or "max_steps" (step cap)
        self.cue_groups = []
        self.responses = []
        self.outcomes = []
        self.stop_causes = []

    def warm_up(self, n, max_steps, **run_kwargs):
        """Healthy retrieval history before trauma onset.

        Runs n retrievals under the original stop rule (reporting=False: no
        report option, so stopping is worth 0 everywhere). A fresh agent,
        whose value starts at 0, then still searches past the summary level
        about half the time and learns what specific memories are worth -
        under the report option it would settle at the summary node and never
        learn (an OGM-like pattern with no trauma behind it).

        Build the network *without* a trauma memory, call this, then add the
        trauma (e.g. ``network.add_memories(20, trauma=-50)``) before the
        measured ``run``. run_kwargs go to ``run`` (time, delta, cue, ...).
        Warm-up trials are moved to ``self.warmup`` (a dict keyed by the LOGS
        names) so record / outcomes / etc. hold only post-onset retrievals.
        """
        self.run(n, max_steps, reporting=False, **run_kwargs)
        self.warmup = {name: getattr(self, name) for name in self.LOGS}
        self._reset_logs()

    def retrieve(self, max_steps, frame_index=0, output_dir=None, cue=None,
                 reporting=True):
        """One cued retrieval. The response is whatever node is in mind when
        the search ends, scored by its level in the hierarchy; see __init__
        for the outcome and stop-cause codes. reporting=False turns the report
        option off (stopping worth 0 everywhere, the original rule)."""
        import os
        import matplotlib.pyplot as plt

        trauma_encountered = False
        total_reward = 0
        self.network.initialize_state(cue)
        self.network.current_graph = self.network.graph.copy()

        # Identify trauma nodes using the same criterion as Memory.__init__
        trauma_nodes = {node for node in self.network.rewards if self.network.rewards[node] < -1}

        while True:
            state = self.network.state
            if state is None:
                r = None  # at the cue: no memory, no reward yet
            else:
                self.states_visited.append(state)
                if state in trauma_nodes:
                    trauma_encountered = True
                r = self.network.rewards[state]
                total_reward += r
            # Stopping ends the search on the current node - no further hop
            # (and so no trace strengthening of a node that was never visited).
            report = self.network.report_value() if reporting else 0.0
            if not self.agent.policy(r, report):
                cause = "agent_stop"
                break
            if len(self.states_visited) >= max_steps:
                cause = "max_steps"
                break
            s = self.network.spreading_activation()
            if s is False:
                cause = "dead_end"
                break
            self.network.state = s

        response = self.network.state
        if response is None:
            outcome = "omission"
        elif self.network.node_level.get(response) == "summary":
            outcome = "categoric"
        else:
            outcome = "specific"
        trial = {"cue_group": self.network.cue_group, "response": response,
                 "outcome": outcome, "stop_cause": cause}

        # Report payoff for whatever was reported (None for an omission), so
        # the agent learns what answering is worth. Delivered in warm-up too:
        # reporting=False only removes the stop option from the decision.
        payoff = None if response is None else self.network.report_value()
        trial["payoff"] = payoff
        if hasattr(self.agent, "end_retrieval"):
            self.agent.end_retrieval(payoff, outcome)

        if output_dir is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            self.network.visualize(
                title=f"Retrieval {len(self.record) + 1}",
                ax=ax,
                min_strength=1
            )
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"frame_{frame_index:04d}.png"),
                        dpi=100, bbox_inches="tight")
            plt.close(fig)
            frame_index += 1

        return trauma_encountered, frame_index, total_reward, trial

    def run(self, n, max_steps, decay=True, time=10, print_message=False,
            delta=None, add_freq=1, a=1, trauma=False, trauma_freq=1, visualize=False,
            output_dir="retrieval_frames", cue=None, reporting=True):
        """cue: None draws a random group as the cue on every retrieval; an
        int cues that group id every time (e.g. 0 for the trauma group).
        reporting: see retrieve; warm_up runs with it off."""
        import os

        if visualize:
            os.makedirs(output_dir, exist_ok=True)

        trauma_count = 0
        frame_index = 0

        for retrieval_num in range(n):
            encountered, frame_index, total_reward, trial = self.retrieve(
                max_steps,
                frame_index=frame_index,
                output_dir=output_dir if visualize else None,
                cue=cue,
                reporting=reporting,
            )
            self.cue_groups.append(trial["cue_group"])
            self.responses.append(trial["response"])
            self.outcomes.append(trial["outcome"])
            self.stop_causes.append(trial["stop_cause"])

            if decay:
                self.network.decay(time)
                if hasattr(self.agent, "decay"):
                    self.agent.decay()
            if delta is not None:
                for _ in range(add_freq):
                    self.network.add_memories(delta, a=a)
                # In addition to the regular delta batch, periodically add a single
                # new trauma memory (reward=`trauma`) so the network keeps accruing
                # aversive content over time rather than having just the one fixed
                # trauma node from __init__.
                if trauma is not False and (retrieval_num + 1) % trauma_freq == 0:
                    self.network.add_memories(1, a=a, trauma=trauma)

            if encountered:
                trauma_count += 1
            self.record.append(self.states_visited)
            self.reward_totals.append(total_reward)
            self.trauma_encounters.append(encountered)
            self.states_visited = []

        trauma_rate = trauma_count / n
        if visualize:
            print(f"Saved {frame_index} frame(s) to '{output_dir}/'")
        if print_message:
            print(f"Trauma encountered in {trauma_count}/{n} retrievals ({trauma_rate:.1%})")
            return trauma_rate