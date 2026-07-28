import random
import rustworkx as rx
import numpy as np
import math
from collections import Counter


class Memory:
    def __init__(self, n, rewards, state=None, thres=0,
                 n_features=50, features_per_node=5, num_groups=0, group_features=0,
                 max_assoc_strength=1.6, sigma=0.1, var=1, trauma=(None, 0)):
        self.n_features = n_features
        self.features_per_node = features_per_node
        self.group_features = group_features
        self.max_assoc_strength = max_assoc_strength
        self.var = var

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
        group_size = n // num_groups if num_groups else 0
        group_blocks = [self._new_group_features() for _ in range(num_groups)]
        for node in range(n):
            if group_size and node < group_size * num_groups:
                group_feats = group_blocks[node // group_size]
            else:
                group_feats = set()
            individual_count = max(features_per_node - len(group_feats), 0)
            feats = group_feats | set(random.sample(range(n_features), individual_count))
            self.node_features[node] = feats
            self.feature_fan.update(feats)

        graph = rx.PyGraph()
        graph.add_nodes_from([[1]] * n)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
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
        self.rewards = rewards

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
        new_indices = self.graph.add_nodes_from([[a]] * n)

        group_feats = self._new_group_features()
        individual_count = max(self.features_per_node - len(group_feats), 0)
        for idx in new_indices:
            feats = group_feats | set(
                random.sample(range(self.n_features), individual_count)
            )
            self.node_features[idx] = feats
            self.feature_fan.update(feats)

        # Intra-batch edges.
        intra_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                w = self._edge_weight(self.node_features[new_indices[i]],
                                       self.node_features[new_indices[j]])
                if w is not None:
                    intra_edges.append((new_indices[i], new_indices[j], w))
        self.graph.add_edges_from(intra_edges)

        # Cross edges between new batch and existing nodes.
        cross_edges = []
        for ni in new_indices:
            for ei in existing_indices:
                w = self._edge_weight(self.node_features[ni], self.node_features[ei])
                if w is not None:
                    cross_edges.append((ni, ei, w))
        self.graph.add_edges_from(cross_edges)

        # Update rewards: all new memories get a reward of 1 by default.
        self.rewards.update({idx: 1 for idx in new_indices})

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
        node_colors = ["crimson" if n in trauma_nodes else "steelblue" for n in nodes]

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
        ax.legend(handles=legend_elements, loc="upper left",
                  fontsize=8, framealpha=0.85)

        if own_fig:
            plt.tight_layout()
            plt.show()

    def initialize_state(self):
        self.state = random.choice(range(len(list(self.graph.nodes()))))

    def spreading_activation(self):
        graph = self.current_graph
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
        graph.remove_node(self.state)
        return next_state

    def decay(self, time):
        for node in self.graph.node_indices():
            self.graph[node] = [trace + time for trace in self.graph[node]]


class Agent:
    def __init__(self, alpha=0.1, gamma=0.9, temp=0.1, v_i=10, decay_rate=0.0):
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.value = v_i
        self.decay_rate = decay_rate

    def policy(self, r):
        if self.value / self.temp < -50:
            return False
        recall = 1 / (1 + np.e ** (-self.value / self.temp)) > random.random()
        if not recall:
            return False
        rpe = r - self.gamma * self.value
        self.value += self.alpha * rpe
        return True

    def decay(self):
        """Relax the accumulated value back toward zero, analogous to
        Memory.decay's aging of activation traces - lets a value that has
        collapsed toward the recall cutoff drift back up over time even
        without a successful recall to update it."""
        self.value *= (1 - self.decay_rate)


class FeatureAgent:
    """Successor-feature agent over the network's per-node feature sets.

    Generalises the group-based SR agent to overlapping features: a node's
    active features are its multi-hot state representation phi(s) (1 for
    each feature it carries, 0 otherwise), psi is learned per feature via
    linear TD(0), and a node's psi(s)/value are the sum over its active
    features' rows/weights - this collapses to the original one-hot,
    per-group formulas exactly when every node carries a single feature.
    """

    def __init__(self, network, alpha_psi=0.1, alpha_w=0.1, gamma=0.9, temp=0.1, decay_rate=0.0):
        self.network = network
        self.gamma = gamma
        self.temp = temp
        self.alpha_psi = alpha_psi
        self.alpha_w = alpha_w
        self.decay_rate = decay_rate
        # total_features (not n_features) since group_features, if used,
        # live in their own index range beyond the general n_features pool.
        n_features = network.total_features
        self.psi = np.zeros((n_features, n_features))  # psi_k: expected discounted future feature occupancy attributable to feature k
        self.w = np.zeros(n_features)                    # reward weight per feature
        self._prev_features = None
        self._prev_reward = None

    def _phi(self, features):
        v = np.zeros(len(self.w)); v[list(features)] = 1.0; return v

    def _psi_of(self, features):
        return self.psi[list(features)].sum(axis=0)

    def _value(self, features):
        return self.w @ self._psi_of(features)

    def _grow_to(self, n_features):
        """Pad psi/w with zeros if the network has allocated new features
        (e.g. add_memories creating another group) since __init__."""
        pad = n_features - len(self.w)
        if pad > 0:
            self.psi = np.pad(self.psi, ((0, pad), (0, pad)))
            self.w = np.pad(self.w, (0, pad))

    def policy(self, r):
        self._grow_to(self.network.total_features)
        f1 = self.network.node_features[self.network.state]
        if self._prev_features is not None:
            f_prev = list(self._prev_features)
            td = self._phi(f_prev) + self.gamma * self._psi_of(f1) - self._psi_of(f_prev)
            self.psi[f_prev] += self.alpha_psi * td               # learn dynamics (slow, stable)
            self.w[f_prev] += self.alpha_w * (self._prev_reward - self.w[f_prev].sum())  # learn reward (fast)

        v1 = self._value(f1)
        # Same guard as Agent.policy: for very negative v1/temp, e ** (-v1/temp)
        # overflows float64 - the sigmoid is ~0 there anyway, so skip straight to False.
        if v1 / self.temp < -50:
            recall = False
        else:
            recall = 1 / (1 + np.e ** (-v1 / self.temp)) > random.random()
        self._prev_features, self._prev_reward = (f1, r) if recall else (None, None)
        return recall

    def decay(self):
        """Relax the learned reward weights back toward zero, the FeatureAgent
        analogue of Agent.decay. Only w (the reward-weight-per-feature vector,
        the direct counterpart of Agent.value) decays - psi (the learned
        successor-feature dynamics) is left untouched, playing the same role
        the graph's edge structure plays for Memory.decay: structural
        knowledge that doesn't fade just because reward associations do."""
        self.w *= (1 - self.decay_rate)


class Simulator:
    def __init__(self, agent, network):
        self.states_visited = []
        self.record = []
        self.reward_totals = []
        self.agent = agent
        self.network = network

    def retrieve(self, max_steps, frame_index=0, output_dir=None):
        import os
        import matplotlib.pyplot as plt

        recall = True
        steps = 0
        trauma_encountered = False
        total_reward = 0
        self.network.initialize_state()
        self.network.current_graph = self.network.graph.copy()

        # Identify trauma nodes using the same criterion as Memory.__init__
        trauma_nodes = {node for node in self.network.rewards if self.network.rewards[node] < -1}

        while recall and steps < max_steps:
            self.states_visited.append(self.network.state)
            if self.network.state in trauma_nodes:
                trauma_encountered = True
            r = self.network.rewards[self.network.state]
            total_reward += r
            recall = self.agent.policy(r)
            s = self.network.spreading_activation()

            if s is False:
                break
            self.network.state = s
            steps += 1

        # Flush any pending TD update for agents that track per-node values.
        if hasattr(self.agent, "reset"):
            self.agent.reset()

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

        return trauma_encountered, frame_index, total_reward

    def run(self, n, max_steps, decay=True, time=10, print_message=False,
            delta=None, add_freq=1, a=1, trauma=False, visualize=False,
            output_dir="retrieval_frames"):
        import os

        if visualize:
            os.makedirs(output_dir, exist_ok=True)

        trauma_count = 0
        frame_index = 0

        for retrieval_num in range(n):
            encountered, frame_index, total_reward = self.retrieve(
                max_steps,
                frame_index=frame_index,
                output_dir=output_dir if visualize else None,
            )

            if decay:
                self.network.decay(time)
                if hasattr(self.agent, "decay"):
                    self.agent.decay()
            if delta is not None:
                # if (retrieval_num + 1) % add_freq == 0:
                #     self.network.add_memories(delta, a=a, trauma=trauma)
                for _ in range(add_freq):
                    self.network.add_memories(delta, a=a)

            if encountered:
                trauma_count += 1
            self.record.append(self.states_visited)
            self.reward_totals.append(total_reward)
            self.states_visited = []

        trauma_rate = trauma_count / n
        if visualize:
            print(f"Saved {frame_index} frame(s) to '{output_dir}/'")
        if print_message:
            print(f"Trauma encountered in {trauma_count}/{n} retrievals ({trauma_rate:.1%})")
            return trauma_rate