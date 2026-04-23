import random
import rustworkx as rx
import numpy as np
import math


class Memory:
    def __init__(self, n, groups, state, rewards, thres=0, sigma=0.1):
        group_size = n // groups

        def weight(i, j):
            return 10 if (i // group_size == j // group_size) else 0

        graph = rx.PyGraph()
        graph.add_nodes_from([[1]] * n)
        graph.add_edges_from([
            (i, j, weight(i, j))
            for i in range(n)
            for j in range(i + 1, n)
        ])
        self.graph = graph
        self.state = state
        self.thres = thres
        self.current_graph = graph.copy()
        self.sigma = sigma
        self.state = state
        self.rewards = rewards

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
    def __init__(self, alpha=0.1, gamma=0.9, temp=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.vtable = {}
        self.reason = None
    
    def policy(self, network):
        s1 = network.state
        if s1 not in self.vtable:
            self.vtable[s1] = 1
        v1 = self.vtable[s1]
        if v1 / self.temp < -50:
            recall = False
        else:
            recall = 1 / (1 + np.e ** (-v1 / self.temp)) > random.random()
        if recall:
            s2 = network.spreading_activation()
            if s2 is False:
                self.reason = "No activation above threshold"
                return False
            network.state = s2
        else:
            self.reason = "Value is too low"
            return False
        if s2 not in self.vtable:
            self.vtable[s2] = 1
        v2 = self.vtable[s2]
        r = network.rewards[s2]
        rpe = r + self.gamma * v2 - v1
        self.vtable[s1] += self.alpha * rpe
        return True
    

class Simulator:
    def __init__(self, agent, network):
        self.states_visited = []
        self.record = []
        self.agent = agent
        self.network = network
    
    def retrieve(self, max_steps, decay, time):
        recall = True
        steps = 0
        self.network.current_graph = self.network.graph.copy()
        self.network.initialize_state()
        while recall and steps < max_steps:
            self.states_visited.append(self.network.state)
            recall = self.agent.policy(self.network)
            steps += 1
        if decay:
            self.network.decay(time)

    def run(self, n, max_steps, decay=True, time=10):
        for _ in range(n):
            self.retrieve(max_steps, decay, time)
            self.record.append(self.states_visited)
            self.states_visited = []