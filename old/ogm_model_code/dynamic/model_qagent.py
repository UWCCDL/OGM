import random
import rustworkx as rx
import numpy as np
import math


class Memory:
    def __init__(self, n, groups, rewards, state=None, thres=0, in_group=10,
                 sigma=0.1, mean_s=-2, var=1, trauma=(None, 0)):
        group_size = n // groups

        def weight(i, j):
            return in_group + random.gauss(0, var) if (i // group_size == j // group_size) else random.gauss(mean_s, var)
        graph = rx.PyGraph()
        graph.add_nodes_from([[1]] * n)
        graph.add_edges_from([
            (i, j, weight(i, j))
            for i in range(n)
            for j in range(i + 1, n)
        ])
        negative_edges = [
            (source, target) 
            for source, target, weight in graph.weighted_edge_list() if weight < 0
        ]
        graph.remove_edges_from(negative_edges)
        self.trauma = trauma
        self.graph = graph
        self.state = state
        self.thres = thres
        self.current_graph = graph.copy()
        self.sigma = sigma
        self.state = state
        self.rewards = rewards
        self.mean_s = mean_s
        self.var = var
        # Map group index → list of node indices belonging to that group.
        self.node_groups = {
            g: list(range(g * group_size, (g + 1) * group_size))
            for g in range(groups)
        }

    def add_memories(self, n, a=1):
        """Add n new memories to the graph as a single cohesive batch.

        The new nodes mirror the structure from __init__:
          - Payload initialised to [1].
          - Reward set to 1 for every new node.
          - Intra-batch edges: strong positive weight (10 + gauss(0, var)).
          - Cross edges to existing nodes: sparse weight (gauss(mean_s, var));
            negative weights are discarded, matching __init__ behaviour.
        """
        existing_indices = self.graph.node_indices()

        # Add n new nodes and capture their stable rustworkx indices.
        new_indices = self.graph.add_nodes_from([[a]] * n)

        # Intra-batch edges (strong, like same-group edges in __init__).
        intra_edges = [
            (new_indices[i], new_indices[j],
             10 + random.gauss(0, self.var))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        self.graph.add_edges_from(intra_edges)

        # Cross edges between new batch and existing nodes (sparse).
        cross_edges = [
            (ni, ei, random.gauss(self.mean_s, self.var))
            for ni in new_indices
            for ei in existing_indices
        ]
        # Discard negative cross edges, matching __init__ pruning logic.
        self.graph.add_edges_from(
            [(u, v, w) for u, v, w in cross_edges if w >= 0]
        )

        # Update rewards: all new memories get a reward of 1.
        self.rewards.update({idx: 1 for idx in new_indices})

        # Register the new batch as its own group (next available group key).
        new_group = max(self.node_groups) + 1
        self.node_groups[new_group] = list(new_indices)

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
        trauma_node = self.trauma[0]
        trauma_present = trauma_node is not None and trauma_node in pos and trauma_node in nodes

        node_xy    = np.array([pos[n] for n in nodes])
        node_sizes = np.array([max(sum_t(n), 1e-9) * 120 for n in nodes])
        node_colors = ["crimson" if n == trauma_node else "steelblue" for n in nodes]

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

    def group_activation(self):
        """Return the total activation of each group.

        Activation of a single node is defined as sum(trace ** -0.5) over
        all of its traces — the same quantity used in spreading_activation
        and visualize.

        Nodes are removed only from self.current_graph during a retrieval
        step; self.graph is never modified by remove_node, so all nodes are
        always present here and no missing-node guard is needed.

        Returns
        -------
        dict
            Mapping of group index → total activation (float).
        """
        result = {}
        for group, members in self.node_groups.items():
            total = 0.0
            for node in members:
                traces = self.graph[node]
                total += sum(trace ** -0.5 for trace in traces) if traces else 0.0
            result[group] = total
        return result

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
            if j == self.trauma[0]:
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
    def __init__(self, alpha=0.1, gamma=0.9, temp=0.1, v_i=10):
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.value = v_i
    
    def policy(self, r):
        if self.value / self.temp < -50:
            return False
        recall = 1 / (1 + np.e ** (-self.value / self.temp)) > random.random()
        if not recall:
            return False
        rpe = r - self.gamma * self.value
        self.value += self.alpha * rpe
        return True


class QAgent:
    """TD(0) agent with a per-node value table.

    Keeps a ``vtable`` dict that maps node indices to learned values,
    mirroring ``model_new``'s Agent.  Designed to work with the existing
    ``Simulator`` unchanged: ``policy(r)`` accepts a scalar reward and
    reads the current state from the ``network`` passed at construction.

    TD update timing
    ----------------
    ``Simulator.retrieve`` calls ``policy(r)`` *before* spreading
    activation moves to the next node, so the successor state s2 is not
    yet known inside ``policy``.  We therefore apply the TD update one
    step later: at the start of the *next* ``policy`` call we have both
    s1 (stored from the previous step) and s2 (the current network state),
    which lets us compute the full TD error  r + γ·V(s2) − V(s1).
    At the end of each retrieval episode the pending update is flushed
    in ``reset()``, treating the final state as terminal (no successor).
    ``Simulator.retrieve`` calls ``reset()`` after the retrieval loop.
    """

    def __init__(self, network, alpha=0.1, gamma=0.9, temp=0.1, v_i=1):
        self.network = network
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.v_i = v_i          # default initial value for unseen nodes
        self.vtable = {}
        # Bookkeeping for deferred TD update
        self._prev_state = None  # s1 from the previous step
        self._prev_reward = None  # r collected at s1

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _v(self, node):
        """Return (and lazily initialise) the value for *node*."""
        if node not in self.vtable:
            self.vtable[node] = self.v_i
        return self.vtable[node]

    def reset(self):
        """Flush any pending TD update at the end of an episode."""
        if self._prev_state is not None:
            # Terminal step: no successor, so bootstrap target is just the reward.
            v1 = self._v(self._prev_state)
            rpe = self._prev_reward - v1
            self.vtable[self._prev_state] += self.alpha * rpe
        self._prev_state = None
        self._prev_reward = None

    # ------------------------------------------------------------------
    # policy — same signature as Agent.policy so Simulator works as-is
    # ------------------------------------------------------------------

    def policy(self, r):
        """Decide whether to continue recall and apply the deferred TD update.

        Parameters
        ----------
        r : float
            Reward at the *current* network state (s1 for this step).

        Returns
        -------
        bool
            True  → continue recall (spreading activation will run next).
            False → stop recall.
        """
        s1 = self.network.state

        # Apply the TD update from the *previous* step now that s2 is known.
        if self._prev_state is not None:
            s2 = s1
            v_s1 = self._v(self._prev_state)
            v_s2 = self._v(s2)
            rpe = self._prev_reward + self.gamma * v_s2 - v_s1
            self.vtable[self._prev_state] += self.alpha * rpe

        # Decide whether to continue based on V(s1).
        v1 = self._v(s1)
        if v1 / self.temp < -50:
            self._prev_state = None
            self._prev_reward = None
            return False

        recall = 1 / (1 + np.e ** (-v1 / self.temp)) > random.random()
        if not recall:
            self._prev_state = None
            self._prev_reward = None
            return False

        # Store s1 and r so the TD update can be completed next step.
        self._prev_state = s1
        self._prev_reward = r
        return True


class Simulator:
    def __init__(self, agent, network):
        self.states_visited = []
        self.record = []
        self.agent = agent
        self.network = network
    
    def retrieve(self, max_steps):
        recall = True
        steps = 0
        trauma_encountered = False
        self.network.initialize_state()
        self.network.current_graph = self.network.graph.copy()

        # Identify trauma nodes using the same criterion as Memory.__init__
        trauma_nodes = {node for node in self.network.rewards if self.network.rewards[node] < -1}

        while recall and steps < max_steps:
            self.states_visited.append(self.network.state)
            if self.network.state in trauma_nodes:
                trauma_encountered = True
            r = self.network.rewards[self.network.state]
            recall = self.agent.policy(r)
            s = self.network.spreading_activation()
            if s is False:
                break
            self.network.state = s
            steps += 1

        # Flush any pending TD update for agents that track per-node values.
        if hasattr(self.agent, "reset"):
            self.agent.reset()

        return trauma_encountered

    def run(self, n, max_steps, decay=True, time=10, print_message=False, delta=None, a=1):
        trauma_count = 0
        for _ in range(n):
            encountered = self.retrieve(max_steps)
            if decay:
                self.network.decay(time)
            if delta is not None:
                self.network.add_memories(delta, a=a)
                
            if encountered:
                trauma_count += 1
            self.record.append(self.states_visited)
            self.states_visited = []

        trauma_rate = trauma_count / n
        if print_message:
            print(f"Trauma encountered in {trauma_count}/{n} retrievals ({trauma_rate:.1%})")
            return trauma_rate