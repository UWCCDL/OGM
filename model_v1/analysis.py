import json
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import model as model

ALPHA = 0.1
GAMMA = 0.99
TEMP = 0.1
VALUE_DECAY = 0.1
TRAUMA_VALENCE = -50

N_RETRIEVALS = 1000
MAX_STEPS = 50
N_REPS = 20
SWEEP_REPS = 10

decay_rate_values = [0, 0.01, 0.02]

default_rewards = {i: 1 for i in range(100)}
default_rewards[0] = TRAUMA_VALENCE

DATA_DIR = "data"


def make_environment():
    return model.Memory(100, default_rewards, num_groups=5, group_features=3, trauma=(0, 1), max_assoc_strength=5.0)


def single_run(agent_factory, n=N_RETRIEVALS, max_steps=MAX_STEPS):
    environment = make_environment()
    agent = agent_factory(environment)
    sim = model.Simulator(agent, environment)
    sim.run(n, max_steps, time=0.1, delta=20)
    lengths = [len(trial) for trial in sim.record]
    return np.array(lengths), np.array(sim.reward_totals), np.array(sim.trauma_encounters)


def collect_trajectories(agent_factory, n=N_RETRIEVALS, max_steps=MAX_STEPS, reps=N_REPS,
                          delta=20, trauma=False, trauma_freq=1):
    length_trajectories = []
    reward_trajectories = []
    for _ in tqdm(range(reps), desc="reps", leave=False):
        environment = make_environment()
        agent = agent_factory(environment)
        sim = model.Simulator(agent, environment)
        sim.run(n, max_steps, time=0.1, delta=delta, trauma=trauma, trauma_freq=trauma_freq)
        length_trajectories.append([len(trial) for trial in sim.record])
        reward_trajectories.append(sim.reward_totals)
    return np.array(length_trajectories), np.array(reward_trajectories)


def sweep_param(agent_builder, param_values, n=N_RETRIEVALS, max_steps=MAX_STEPS, reps=SWEEP_REPS,
                 trauma=False, trauma_freq=1):
    raw_avg_length = []
    for value in tqdm(param_values, desc="sweep"):
        lengths, _ = collect_trajectories(
            lambda env, value=value: agent_builder(env, value),
            n=n, max_steps=max_steps, reps=reps, trauma=trauma, trauma_freq=trauma_freq)
        raw_avg_length.append(lengths.mean(axis=1))
    return raw_avg_length


def run_single_run_section():
    feature_lengths, feature_rewards, feature_trauma = single_run(
        lambda env: model.FeatureAgent(env, alpha_psi=ALPHA, alpha_w=ALPHA, gamma=GAMMA, temp=TEMP, decay_rate=VALUE_DECAY))
    agent_lengths, agent_rewards, agent_trauma = single_run(
        lambda env: model.Agent(alpha=ALPHA, gamma=GAMMA, temp=TEMP, v_i=0, decay_rate=VALUE_DECAY))

    retrieval_num = np.arange(1, len(feature_lengths) + 1)
    rows = []
    for agent_type, lengths, rewards, trauma in [
        ("FeatureAgent", feature_lengths, feature_rewards, feature_trauma),
        ("Agent", agent_lengths, agent_rewards, agent_trauma),
    ]:
        for i in range(len(lengths)):
            rows.append({
                "retrieval_num": retrieval_num[i],
                "agent_type": agent_type,
                "length": lengths[i],
                "reward": rewards[i],
                "trauma_encounter": bool(trauma[i]),
            })
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "single_run.csv"), index=False)


def run_comparison_section():
    feature_lengths, feature_rewards = collect_trajectories(
        lambda env: model.FeatureAgent(env, alpha_psi=ALPHA, alpha_w=ALPHA, gamma=GAMMA, temp=TEMP, decay_rate=VALUE_DECAY))
    agent_lengths, agent_rewards = collect_trajectories(
        lambda env: model.Agent(alpha=ALPHA, gamma=GAMMA, temp=TEMP, v_i=0, decay_rate=VALUE_DECAY))

    rows = []
    for agent_type, lengths, rewards in [
        ("FeatureAgent", feature_lengths, feature_rewards),
        ("Agent", agent_lengths, agent_rewards),
    ]:
        reps, n = lengths.shape
        for rep in range(reps):
            for retrieval_num in range(n):
                rows.append({
                    "rep": rep,
                    "retrieval_num": retrieval_num + 1,
                    "agent_type": agent_type,
                    "length": lengths[rep, retrieval_num],
                    "reward": rewards[rep, retrieval_num],
                })
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "comparison_1.csv"), index=False)


def run_decay_rate_sweep_section():
    agent_raw = sweep_param(
        lambda env, dr: model.Agent(alpha=ALPHA, gamma=GAMMA, temp=TEMP, v_i=0, decay_rate=dr),
        decay_rate_values)
    feature_raw = sweep_param(
        lambda env, dr: model.FeatureAgent(env, alpha_psi=ALPHA, alpha_w=ALPHA, gamma=GAMMA, temp=TEMP, decay_rate=dr),
        decay_rate_values)

    rows = []
    for agent_type, raw in [("Agent", agent_raw), ("FeatureAgent", feature_raw)]:
        for dr, reps_arr in zip(decay_rate_values, raw):
            for rep, avg_length in enumerate(reps_arr):
                rows.append({
                    "agent_type": agent_type,
                    "decay_rate": dr,
                    "rep": rep,
                    "avg_retrieval_length": avg_length,
                })
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "decay_rate_sweep.csv"), index=False)


def save_params():
    rows = [
        {"name": "ALPHA", "value": ALPHA},
        {"name": "GAMMA", "value": GAMMA},
        {"name": "TEMP", "value": TEMP},
        {"name": "VALUE_DECAY", "value": VALUE_DECAY},
        {"name": "TRAUMA_VALENCE", "value": TRAUMA_VALENCE},
        {"name": "N_RETRIEVALS", "value": N_RETRIEVALS},
        {"name": "MAX_STEPS", "value": MAX_STEPS},
        {"name": "N_REPS", "value": N_REPS},
        {"name": "SWEEP_REPS", "value": SWEEP_REPS},
        {"name": "decay_rate_values", "value": json.dumps(decay_rate_values)},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(DATA_DIR, "params.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    save_params()
    run_single_run_section()
    run_comparison_section()
    run_decay_rate_sweep_section()
