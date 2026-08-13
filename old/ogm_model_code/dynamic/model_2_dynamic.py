import random
import rustworkx as rx
import numpy as np
import math


class Memory:
    def __init__(self, n, groups, rewards, state=None, thres=0, sigma=0.1,
                 mean_s=-2, var=1, trauma=(None, 0)):
        group_size = n // groups

        def weight(i, j):
            return 10 + random.gauss(0, var) if (i // group_size == j // group_size) else random.gauss(mean_s, var)
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

    def add_memories(self, n):
        existing_indices = self.graph.node_indices()

        # Add n new nodes and capture their stable rustworkx indices.
        new_indices = self.graph.add_nodes_from([[1]] * n)

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

    def visualize(self, title="Memory Network", ax=None):
        """Visualize the current graph using a force-directed (spring) layout.

        Visual encodings
        ----------------
        Node size   : number of recall visits (len of payload) — more visits → larger.
        Node colour : steel blue for normal nodes, crimson for the trauma node.
        Node border : thick black ring marks the currently active state.
        Edge width  : proportional to edge weight — stronger connections are thicker.
        Edge alpha  : also proportional to weight, so weak cross-edges recede visually.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        graph = self.graph
        nodes = graph.node_indices()
        if not nodes:
            return

        # --- layout ------------------------------------------------------------
        pos = rx.graph_spring_layout(
            graph,
            weight_fn=lambda w: float(w),
            num_iter=200,
            seed=42,
        )  # {node_idx: (x, y)}

        # --- derived per-node quantities ---------------------------------------
        trauma_node = self.trauma[0]
        trauma_present = trauma_node is not None and trauma_node in pos

        node_xy    = np.array([pos[n] for n in nodes])
        node_sizes = np.array([max(len(graph[n]), 1) * 120 for n in nodes])
        node_colors = ["crimson" if n == trauma_node else "steelblue" for n in nodes]

        # --- derived per-edge quantities ---------------------------------------
        edge_list = graph.weighted_edge_list()  # (u, v, w)
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
                   label="size ∝ recall count"),
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

        return trauma_encountered

    def run(self, n, max_steps, decay=True, time=10, print_message=False, delta=None):
        trauma_count = 0
        for _ in range(n):
            encountered = self.retrieve(max_steps)
            if decay:
                self.network.decay(time)
            if delta is not None:
                self.network.add_memories(delta)
                
            if encountered:
                trauma_count += 1
            self.record.append(self.states_visited)
            self.states_visited = []

        trauma_rate = trauma_count / n
        if print_message:
            print(f"Trauma encountered in {trauma_count}/{n} retrievals ({trauma_rate:.1%})")
            return trauma_rate
