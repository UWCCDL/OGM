import random
import numpy as np


class Agent():
    def __init__(self, cue, alpha=0.1, gamma=0.9, temp=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.temp = temp
        self.vtable = {cue: 1}

    def policy(self, state):
        v = self.vtable[state]
        if v < -50:
            return 1 / (1 + np.e ** (50 / self.temp)) > random.random()
        return 1 / (1 + np.e ** (-v / self.temp)) > random.random()

    def v_learning(self, s1, r, s2):
        q1 = self.vtable[s1]
        if s2 not in self.vtable:
            self.vtable[s2] = 1
        q2 = self.vtable[s2]
        rpe = r + self.gamma * q2 - q1
        self.vtable[s1] += self.alpha * rpe
