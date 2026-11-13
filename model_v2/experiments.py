"""Run the model v2 experiments and save trial-level results for analysis.ipynb.

Protocol (every run)
--------------------
1. Build a memory network with no trauma memory.
2. Healthy warm-up: WARMUP retrievals under the original stop rule (no report
   option), so the agent learns what specific memories are worth.
3. Trauma onset: a new 20-memory group whose first memory has valence
   TRAUMA_VALENCE.
4. MEASURED retrievals. Each is cued by the trauma group with probability
   P_TRAUMA_CUE, otherwise by a random group (cue = "other").

Experiments
-----------
main         The two poles of the spectrum, crossed with value decay:
             SV (SingleValueAgent) vs SF (FeatureAgent, alpha_bias = 0)
             x decay_rate in DECAY_RATES.
sensitivity  Where between the poles SF stops localizing avoidance:
             SF over ALPHA_BIAS_VALUES (plus SV as the pole) at SENS_DECAY.
severity     Whether avoidance shows up as categoric rather than omitted
             memories: SEVERITY_AGENTS x trauma valence (SEVERITY_VALENCES) x
             payoff of a categoric answer (SEVERITY_SUMMARY_REPORTS), at
             SENS_DECAY, SEVERITY_SEEDS seeds.
spectrum     The OGM spectrum itself: SF over CONTENT_SHARES (content_share
             = how the total reward-learning rate alpha_w is split between
             memory content and the bias; 1 = content-specific blame, 0 = the
             pole, where only retrieval in general carries value), at
             moderate trauma (SPECTRUM_VALENCE),
             categoric payoff SPECTRUM_SUMMARY_REPORT and SENS_DECAY, with a
             long measured phase (SPECTRUM_MEASURED) to see where each agent
             settles. Main, sensitivity and severity predate the merged axis
             and are parameterised by alpha_bias with content learning at
             alpha_w (content_share = 1 is alpha_bias = 0 there too).

Conditions shared between experiments are run once. Every condition is
(agent, alpha_bias, decay_rate, trauma_valence, summary_report,
alpha_content); alpha_content is None (= alpha_w) unless lowered. A spectrum
point is stored as its (alpha_bias, alpha_content) pair, so it shares cached
runs with any earlier condition that had the same rates. Main and
sensitivity use the default TRAUMA_VALENCE and SUMMARY_REPORT. An experiment
may also override the measured-phase length; such runs carry it in their file
name and in the run_length column, so they never mix with 600-trial runs.

Output (in data/, or data_quick/ with --quick)
------
trials.csv   one row per retrieval, for every condition whose runs exist
             (so --only adds to it rather than replacing it): agent,
             alpha_bias, decay_rate, trauma_valence, summary_report, seed,
             phase (warmup / measured), trial, cue (trauma / other), outcome
             (specific / categoric / omission), stop_cause, length, reward,
             trauma_encounter, alpha_content (content-feature learning
             rate; 0 for SV), content_share (alpha_content / (alpha_bias +
             alpha_content); 0 for SV), on_spectrum (True for SV and for SF
             runs whose two rates sum to alpha_w), run_length (measured trials in that run),
             and - measured trials only - w_bias (the
             agent's "retrieval in general" weight; SV's single value) and
             w_trauma (mean weight of the trauma group's features; SF only).
params.json  the configuration below, so the notebook can label everything.
network/snapshots.json   (--snapshots) a few spectrum runs (SNAPSHOT_RUNS:
             content_share, seed) re-run with the same seed and inspected at
             SNAPSHOT_TIMES: the agent's probability of continuing at each
             node (and cue) of the trauma group, SNAPSHOT_OLD_GROUPS and the
             newest group, plus the paths of the next SNAPSHOT_WALKS
             retrievals. Each run is checked against its cached CSV.
runs/<config hash>/   one CSV per (condition, seed). Runs already there are
             skipped, so an interrupted run resumes. The hash covers only the
             settings that change a run (RUN_KEYS), so adding experiments
             reuses cached runs while a model/protocol change starts fresh.

Usage
-----
    python experiments.py                 # full run (long: run in background)
    python experiments.py --quick         # tiny smoke test -> data_quick/
    python experiments.py --workers 6 --seeds 5 --only main
    python experiments.py --only severity # just the severity sweep
    python experiments.py --only spectrum
    python experiments.py --snapshots     # network snapshots (after --only spectrum)
    python experiments.py --watch         # progress bar for a run already going
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import model

HERE = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    # memory network
    "n_memories": 100,
    "num_groups": 5,
    "group_features": 3,
    "max_assoc_strength": 5.0,
    "trauma_boost": 1,          # extra activation of trauma memories in spreading activation
    "summary_report": 0.5,      # payoff of reporting a categoric memory
    "event_report": 0.5,        # payoff of reporting a specific memory
    "trace_time": 0.1,          # Memory.decay time step per retrieval
    "delta": 20,                # new memories (one new group) added per retrieval
    "max_steps": 50,
    # agent
    "alpha_psi": 0.1,
    "alpha_w": 0.1,
    "gamma": 0.99,
    "temp": 0.1,
    # protocol
    "warmup": 150,
    "measured": 600,
    "trauma_valence": -50,
    "p_trauma_cue": 0.25,
    # experiments
    "decay_rates": [0, 0.01, 0.02],
    "alpha_bias_values": [0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.1],
    "sens_decay": 0.01,
    "n_seeds": 10,
    "severity_agents": [["SV", None], ["SF", 0], ["SF", 0.01]],
    "severity_valences": [-5, -10, -20, -50],
    "severity_summary_reports": [0.5, 1, 2, 4],
    "severity_seeds": 5,
    "content_shares": [1, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05, 0],
    "spectrum_valence": -10,
    "spectrum_summary_report": 1,
    "spectrum_measured": 1500,
    "spectrum_seeds": 10,
    # network snapshots (--snapshots): spectrum runs re-run with the same
    # seed, inspected at these measured-phase times (retrievals done)
    "snapshot_runs": [[0.2, 5], [0.1, 4], [0.1, 1], [0.05, 4]],   # [content_share, seed]
    "snapshot_times": [0, 300, 700, 1400],
    "snapshot_walks": 5,            # retrievals recorded after each snapshot time
    "snapshot_old_groups": [0, 75, 150],   # pre-trauma groups shown (ids; 0-4 initial, then warm-up)
}

QUICK = {"warmup": 10, "measured": 20, "n_seeds": 2, "severity_seeds": 1,
         "spectrum_measured": 40, "spectrum_seeds": 1,
         "snapshot_times": [0, 10, 20, 30], "snapshot_old_groups": [0, 3, 8]}

# Settings that change what a single run does (the cache key). Experiment
# design lists, seed counts and the per-condition trauma_valence /
# summary_report are not part of it.
RUN_KEYS = ["n_memories", "num_groups", "group_features", "max_assoc_strength",
            "trauma_boost", "event_report", "trace_time", "delta", "max_steps",
            "alpha_psi", "alpha_w", "gamma", "temp", "warmup", "measured", "p_trauma_cue"]
EXPERIMENTS = ["main", "sensitivity", "severity", "spectrum"]


def share_rates(share, alpha_w):
    """(alpha_bias, alpha_content) of a content_share, as stored in a
    condition: rounded so float noise doesn't change file names, and
    alpha_content None when it equals alpha_w (the default)."""
    ab = round((1 - share) * alpha_w, 12)
    ac = round(share * alpha_w, 12)
    return ab, (None if ac == alpha_w else ac)


def experiments(cfg):
    """{experiment: (conditions, n_seeds, measured)}. A condition is
    (agent, alpha_bias, decay_rate, trauma_valence, summary_report,
    alpha_content); alpha_bias is None for SV, which has no separate bias
    rate, and alpha_content is None for the default (alpha_w). measured is
    the length of the measured phase for that experiment's runs."""
    tv, sr, d0 = cfg["trauma_valence"], cfg["summary_report"], cfg["sens_decay"]
    main = [(a, ab, d, tv, sr, None) for d in cfg["decay_rates"] for a, ab in (("SV", None), ("SF", 0))]
    sens = [("SV", None, d0, tv, sr, None)] + [("SF", ab, d0, tv, sr, None) for ab in cfg["alpha_bias_values"]]
    sev = [(a, ab, d0, v, s, None) for a, ab in cfg["severity_agents"]
           for v in cfg["severity_valences"] for s in cfg["severity_summary_reports"]]
    spv, sps = cfg["spectrum_valence"], cfg["spectrum_summary_report"]
    spectrum = [("SF", *share_rates(c, cfg["alpha_w"]), d0, spv, sps) for c in cfg["content_shares"]]
    # Share 0 is the pole. SV is not included: it learns value like share 0
    # but cannot tell memories apart in psi either.
    spectrum = [(a, ab, d, v, s, ac) for a, ab, ac, d, v, s in spectrum]
    m = cfg["measured"]
    return {"main": (main, cfg["n_seeds"], m), "sensitivity": (sens, cfg["n_seeds"], m),
            "severity": (sev, cfg["severity_seeds"], m),
            "spectrum": (spectrum, cfg["spectrum_seeds"], cfg["spectrum_measured"])}


def run_one(job, observe=None):
    """One (condition, seed) run of the full protocol -> list of row dicts.

    observe(t, sim, env, agent, trauma_group), if given, is called before
    measured retrieval t + 1 (t = retrievals done so far) and once more at
    the end (t = measured). It must not draw random numbers or change the
    agent, so the run is identical with or without it."""
    (agent_name, alpha_bias, decay, trauma_valence, summary_report, alpha_content), seed, cfg = job
    random.seed(seed)
    np.random.seed(seed)

    env = model.Memory(cfg["n_memories"], {i: 1 for i in range(cfg["n_memories"])},
                       num_groups=cfg["num_groups"], group_features=cfg["group_features"],
                       max_assoc_strength=cfg["max_assoc_strength"],
                       trauma=(None, cfg["trauma_boost"]),
                       summary_report=summary_report, event_report=cfg["event_report"])
    kw = dict(alpha_psi=cfg["alpha_psi"], alpha_w=cfg["alpha_w"], gamma=cfg["gamma"],
              temp=cfg["temp"], decay_rate=decay)
    if agent_name == "SV":
        agent = model.SingleValueAgent(env, **kw)
    else:
        agent = model.FeatureAgent(env, alpha_bias=alpha_bias, alpha_content=alpha_content, **kw)
    sim = model.Simulator(agent, env)
    run_kw = dict(time=cfg["trace_time"], delta=cfg["delta"])

    base = {"agent": agent_name, "alpha_bias": alpha_bias, "decay_rate": decay,
            "trauma_valence": trauma_valence, "summary_report": summary_report, "seed": seed,
            "alpha_content": 0.0 if agent_name == "SV" else agent.alpha_content,
            "run_length": cfg["measured"]}
    rows = []

    sim.warm_up(cfg["warmup"], cfg["max_steps"], **run_kw)
    wu = sim.warmup
    for t in range(cfg["warmup"]):
        rows.append({**base, "phase": "warmup", "trial": t + 1, "cue": "other",
                     "outcome": wu["outcomes"][t], "stop_cause": wu["stop_causes"][t],
                     "length": len(wu["record"][t]), "reward": wu["reward_totals"][t],
                     "trauma_encounter": wu["trauma_encounters"][t],
                     "w_bias": np.nan, "w_trauma": np.nan})

    env.add_memories(cfg["delta"], trauma=trauma_valence)
    trauma_group = env.n_groups - 1
    trauma_idx = ([f + agent.N_AGENT for f in env.groups[trauma_group]["features"]]
                  if agent.content else None)

    for t in range(cfg["measured"]):
        if observe:
            observe(t, sim, env, agent, trauma_group)
        cue = trauma_group if random.random() < cfg["p_trauma_cue"] else None
        sim.run(1, cfg["max_steps"], cue=cue, **run_kw)
        rows.append({**base, "phase": "measured", "trial": t + 1,
                     "cue": "trauma" if sim.cue_groups[-1] == trauma_group else "other",
                     "outcome": sim.outcomes[-1], "stop_cause": sim.stop_causes[-1],
                     "length": len(sim.record[-1]), "reward": sim.reward_totals[-1],
                     "trauma_encounter": sim.trauma_encounters[-1],
                     "w_bias": agent.w[agent.BIAS],
                     "w_trauma": agent.w[trauma_idx].mean() if trauma_idx else np.nan})
    if observe:
        observe(cfg["measured"], sim, env, agent, trauma_group)
    return rows


def run_file(runs_dir, cond, seed, length=None):
    """Per-run cache file. length is given only when it differs from the
    default measured phase (so the default runs keep their original names)."""
    f = lambda x: "None" if x is None else f"{x:g}"   # 0.0 and 0 name the same run
    agent, ab, d, tv, sr, ac = (cond[0],) + tuple(f(x) for x in cond[1:])
    tag = (f"_ac{ac}" if cond[5] is not None else "") + (f"_n{length}" if length else "")
    return os.path.join(runs_dir, f"{agent}_ab{ab}_decay{d}_tv{tv}_sr{sr}{tag}_seed{seed}.csv")


def run_and_save(job):
    cond, seed, cfg, path = job
    t0 = time.time()
    pd.DataFrame(run_one((cond, seed, cfg))).to_csv(path, index=False)
    return path, time.time() - t0


def plan(args):
    """Config, output folders, the jobs to run for these arguments, and
    every job of every experiment (what trials.csv is assembled from)."""
    cfg = dict(CONFIG)
    if args.quick:
        cfg.update(QUICK)
    if args.seeds:
        cfg["n_seeds"] = cfg["severity_seeds"] = cfg["spectrum_seeds"] = args.seeds
    out_dir = os.path.join(HERE, "data_quick" if args.quick else "data")
    run_cfg = {k: cfg[k] for k in RUN_KEYS}
    digest = hashlib.sha1(json.dumps(run_cfg, sort_keys=True).encode()).hexdigest()[:10]
    runs_dir = os.path.join(out_dir, "runs", digest)

    def jobs_for(names):
        jobs = {}
        for name in names:
            conds, n_seeds, measured = experiments(cfg)[name]
            run_cfg = cfg if measured == cfg["measured"] else {**cfg, "measured": measured}
            length = None if measured == cfg["measured"] else measured
            for c in conds:
                for s in range(n_seeds):
                    path = run_file(runs_dir, c, s, length)
                    jobs.setdefault(path, (c, s, run_cfg, path))
        return list(jobs.values())

    jobs = jobs_for([args.only] if args.only else EXPERIMENTS)
    return cfg, out_dir, runs_dir, digest, jobs, jobs_for(EXPERIMENTS)


def _node_info(env, agent, n):
    """A node's place in the network and what the agent would do there:
    q = value of continuing, p = probability of continuing rather than
    reporting it (categoric at a summary node, specific at an event)."""
    level = env.node_level[n]
    report = env.summary_report if level == "summary" else env.event_report
    q, p = agent.p_continue(env.node_features[n], report)
    return {"id": n, "group": env.node_group[n], "level": level,
            "trauma": n in env.trauma[0], "q": q, "p": p}


def landscape(env, agent, gids):
    """The agent's stop/continue map over whole groups: for each group, the
    value and probability of continuing at its cue (stopping there is an
    omission) and at each of its nodes."""
    out = []
    for g in gids:
        feats = env.groups[g]["features"]
        q, p = agent.p_continue(feats, 0.0, at_cue=True)
        members = sorted(env.feature_nodes[min(feats)])    # group features are exclusive
        out.append({"group": g, "cue_q": q, "cue_p": p,
                    "nodes": [_node_info(env, agent, n) for n in members]})
    return out


def all_groups(env, agent):
    """Compact version of landscape() for every group in the network: value
    of continuing at the cue and the summary, and the mean and lowest over
    the group's events."""
    out = []
    for g, info in env.groups.items():
        events = [n for n in env.feature_nodes[min(info["features"])] if env.node_level[n] == "event"]
        eq = [agent.p_continue(env.node_features[n], env.event_report)[0] for n in events]
        out.append({"group": g, "cue_q": agent.p_continue(info["features"], 0.0, at_cue=True)[0],
                    "summary_q": agent.p_continue(info["features"], env.summary_report)[0],
                    "event_q_mean": float(np.mean(eq)), "event_q_min": float(np.min(eq))})
    return out


def snapshot_run(job):
    """Re-run one spectrum run (same seed, so the same run) and inspect it at
    cfg["snapshot_times"]: the stop/continue map over the trauma group, a few
    pre-trauma groups and the newest group (and, in brief, over every group:
    all_groups), plus the next snapshot_walks
    retrievals (path, outcome, edges among the visited nodes), with values
    taken just after those retrievals."""
    share, seed, cfg = job
    ab, ac = share_rates(share, cfg["alpha_w"])
    cond = ("SF", ab, cfg["sens_decay"], cfg["spectrum_valence"], cfg["spectrum_summary_report"], ac)
    run_cfg = {**cfg, "measured": cfg["spectrum_measured"]}
    times, k = sorted(cfg["snapshot_times"]), cfg["snapshot_walks"]
    shots = {}

    def observe(t, sim, env, agent, tg):
        if t in times:
            newest = max(g for g in env.groups if g != tg)
            shots[t] = {"t": t, "trauma_group": tg, "newest_group": newest, "n_groups": len(env.groups),
                        "w_bias": float(agent.w[agent.BIAS]), "w_cat": float(agent.w[agent.R_CAT]),
                        "w_spec": float(agent.w[agent.R_SPEC]),
                        "groups": landscape(env, agent, [tg] + list(cfg["snapshot_old_groups"]) + [newest]),
                        "all_groups": all_groups(env, agent),
                        "walks": []}
        if t - k in shots:
            for i in range(t - k, t):
                path = sim.record[i]
                visited = set(path)
                edges = [(u, v, float(w)) for u in visited for v, w in env.graph.adj(u).items()
                         if v in visited and u < v]
                cue_q, cue_p = agent.p_continue(env.groups[sim.cue_groups[i]]["features"], 0.0, at_cue=True)
                shots[t - k]["walks"].append({
                    "trial": i + 1, "cue_group": sim.cue_groups[i],
                    "cue": "trauma" if sim.cue_groups[i] == tg else "other",
                    "outcome": sim.outcomes[i], "stop_cause": sim.stop_causes[i],
                    "cue_q": cue_q, "cue_p": cue_p,
                    "nodes": [_node_info(env, agent, n) for n in path], "edges": edges})

    rows = run_one((cond, seed, run_cfg), observe=observe)
    outcomes = [r["outcome"] for r in rows if r["phase"] == "measured"]
    return {"content_share": share, "seed": seed, "cond": cond,
            "snapshots": [shots[t] for t in times if t in shots], "outcomes": outcomes}


def snapshots(cfg, out_dir, runs_dir, workers):
    """Run the snapshot runs and write <out_dir>/network/snapshots.json,
    checking each against its cached spectrum run."""
    jobs = [(share, seed, cfg) for share, seed in cfg["snapshot_runs"]]
    print(f"{len(jobs)} snapshot runs ({min(workers, len(jobs))} workers)", flush=True)
    with Pool(min(workers, len(jobs))) as pool:
        results = list(tqdm(pool.imap(snapshot_run, jobs), total=len(jobs), unit="run", desc="snapshots"))
    for r in results:
        cached = run_file(runs_dir, r["cond"], r["seed"], cfg["spectrum_measured"])
        if os.path.exists(cached):
            c = pd.read_csv(cached)
            c = c.loc[c["phase"] == "measured", "outcome"].tolist()
            same = np.mean([a == b for a, b in zip(c, r["outcomes"])]) if len(c) == len(r["outcomes"]) else np.nan
            print(f"share {r['content_share']:g} seed {r['seed']}: {same:.1%} of outcomes match the cached run")
        else:
            print(f"share {r['content_share']:g} seed {r['seed']}: no cached run to check against")
        r["cond"] = list(r["cond"])
    os.makedirs(os.path.join(out_dir, "network"), exist_ok=True)
    path = os.path.join(out_dir, "network", "snapshots.json")
    with open(path, "w") as f:
        json.dump({"walks_per_snapshot": cfg["snapshot_walks"], "runs": results}, f)
    print(f"wrote {path}")


def watch(jobs, interval=5):
    """Progress bar for a run started elsewhere (e.g. in the background):
    counts finished per-run CSVs until all jobs are done. Ctrl-C stops
    watching without affecting the run."""
    done = lambda: sum(os.path.exists(j[3]) for j in jobs)
    with tqdm(total=len(jobs), initial=done(), unit="run", desc="experiments", smoothing=0) as bar:
        while bar.n < len(jobs):
            time.sleep(interval)
            bar.update(done() - bar.n)
    print("all runs finished - trials.csv is written once the experiment process wraps up")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="tiny smoke test, writes data_quick/")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seeds", type=int, help="override n_seeds")
    ap.add_argument("--only", choices=EXPERIMENTS)
    ap.add_argument("--snapshots", action="store_true",
                    help="re-run SNAPSHOT_RUNS and write network/snapshots.json")
    ap.add_argument("--watch", action="store_true",
                    help="show a progress bar for a run already in progress (same other flags)")
    args = ap.parse_args()

    cfg, out_dir, runs_dir, digest, jobs, all_jobs = plan(args)
    if args.watch:
        return watch(jobs)
    if args.snapshots:
        return snapshots(cfg, out_dir, runs_dir, args.workers)
    os.makedirs(runs_dir, exist_ok=True)

    todo = [j for j in jobs if not os.path.exists(j[3])]
    print(f"{len(jobs)} runs ({len(jobs) - len(todo)} cached, {len(todo)} to run, "
          f"{args.workers} workers)", flush=True)

    with Pool(args.workers) as pool, \
            tqdm(total=len(jobs), initial=len(jobs) - len(todo), unit="run",
                 desc="experiments", smoothing=0) as bar:
        for path, secs in pool.imap_unordered(run_and_save, todo):
            bar.set_postfix_str(f"last: {os.path.basename(path)[:-4]} ({secs:.0f}s)")
            bar.update(1)

    have = [j[3] for j in all_jobs if os.path.exists(j[3])]
    trials = pd.concat([pd.read_csv(p) for p in have], ignore_index=True)
    # runs cached before run_length existed are default-length runs
    trials["run_length"] = trials.get("run_length", np.nan)
    trials["run_length"] = trials["run_length"].fillna(cfg["measured"]).astype(int)
    # runs cached before alpha_content existed: SF learned content at alpha_w,
    # SV has no content features
    trials["alpha_content"] = trials.get("alpha_content", np.nan)
    trials["alpha_content"] = trials["alpha_content"].fillna(
        trials["agent"].map({"SV": 0.0}).fillna(cfg["alpha_w"]))
    # position on the merged spectrum; only meaningful where on_spectrum
    total = trials["alpha_bias"].fillna(0) + trials["alpha_content"]
    trials["content_share"] = np.where(trials["agent"] == "SV", 0.0,
                                       (trials["alpha_content"] / total.where(total > 0)).round(10))
    trials["on_spectrum"] = (trials["agent"] == "SV") | np.isclose(total, cfg["alpha_w"])
    trials.to_csv(os.path.join(out_dir, "trials.csv"), index=False)
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({**cfg, "config_hash": digest}, f, indent=2)
    print(f"wrote {len(trials)} rows to {os.path.join(out_dir, 'trials.csv')}")


if __name__ == "__main__":
    sys.exit(main())
