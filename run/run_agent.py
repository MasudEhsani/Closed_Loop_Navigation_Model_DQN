"""
Run one DQN agent through the trial schedule (counterpart of network/network_runner.py).

    python run/run_agent.py --seed 7000                     # replay mode from config/dqn_params.json
    python run/run_agent.py --seed 7000 --replay none       # same seed, no learning during rest
    python run/run_agent.py --seed 7000 --replay reverse --tag test

Output folder: data/agent<seed>_<replay_mode>[_<tag>]/ with files in the spiking model's format
(trials_params.dat, locs_time.dat, dopamine_l2-0.gdf, action-0.gdf, actionID_dir.dat) so that
meta_analysis/trial_collective.py of the spiking project can read it, plus DQN-specific files
(summary.dat, replay_log.dat, checkpoints/trial_XX.npz with network weights and the Q-map).
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "run"))

from env import GridWorld, Vision, HEADING_VEC, ACTIONS   # noqa: E402
from agent import DQNAgent, REPLAY_MODES                  # noqa: E402
from trial_generator import build_trials                  # noqa: E402

GDF_HEADER = "# NEST version: 3.4 (format only; written by the DQN model)\n" \
             "# RecordingBackendASCII version: 2\nsender\ttime_ms\n"
ACTION_IDS = {h: h + 1 for h in range(4)}       # heading N/E/S/W -> "neuron" id 1..4


def load(path):
    with open(path) as f:
        return json.load(f)


def epsilon(cfg, k):
    e = cfg["exploration"]
    return max(e["eps_end"], e["eps_start"] * e["eps_decay_per_trial"] ** k)


def run(seed, replay=None, tag=None, data_dir=None, sim_path=None, env_path=None, dqn_path=None,
        save_qmaps=True, quiet=False):
    sim = load(sim_path or os.path.join(ROOT, "config", "sim_params.json"))
    envp = load(env_path or os.path.join(ROOT, "config", "env_params.json"))
    cfg = load(dqn_path or os.path.join(ROOT, "config", "dqn_params.json"))
    if replay is not None:
        if replay not in REPLAY_MODES:
            raise SystemExit(f"--replay must be one of {REPLAY_MODES}")
        cfg["rest"]["replay_mode"] = replay
    mode = cfg["rest"]["replay_mode"]
    sim = copy.deepcopy(sim)
    sim["master_rng_seed"] = seed

    name = f"agent{seed}_{mode}" + (f"_{tag}" if tag else "")
    out = os.path.join(data_dir or os.path.join(ROOT, "data"), name)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    sim["data_path"] = out

    trials = build_trials(sim)
    trials.to_csv(os.path.join(out, "trials_params.dat"), sep="\t", index=False)
    sim["simtime"] = float(trials["max_tr_dur"].sum())
    sim["max_num_trs"] = len(trials)
    for fname, d in (("sim_params.json", sim), ("env_params.json", envp), ("dqn_params.json", cfg)):
        with open(os.path.join(out, fname), "w") as f:
            json.dump(d, f, indent=2)

    world = GridWorld(envp)
    vision = Vision(envp, world)
    agent = DQNAgent(vision, len(ACTIONS), cfg, seed)
    step_ms = float(envp["step_ms"])
    L = cfg["learning"]

    pd.DataFrame({"id": [ACTION_IDS[h] for h in range(4)],
                  "x": HEADING_VEC[:, 0].astype(float), "y": HEADING_VEC[:, 1].astype(float)}
                 ).to_csv(os.path.join(out, "actionID_dir.dat"), sep="\t", index=False)

    locs, dop, spikes, replay_rows, summary = [], [], [], [], []
    t0, n_explore = 0.0, 0
    t_start = time.time()

    for _, tr in trials.iterrows():
        trial, typ, dur = int(tr["trial_num"]), tr["type"], float(tr["max_tr_dur"])
        row = {"trial": trial, "type": typ, "success": 0, "steps_to_goal": np.nan,
               "time_to_goal_ms": np.nan, "eps": np.nan, "mean_abs_td": np.nan, "rest_updates": 0}

        if typ == "explore":
            eps = epsilon(cfg, n_explore)
            n_explore += 1
            world.set_goal(tr["goal_x"], tr["goal_y"], tr["goal_size1"])
            s = world.reset(tr["start_x"], tr["start_y"])
            agent.start_trial()
            x, y = world.cell_center(s[0], s[1])
            locs.append((trial, t0, x, y, s[2]))
            tds = []
            for k in range(int(round(dur / step_ms))):
                a = agent.act(s, eps)
                s2, goal, info = world.step(a)
                r = L["goal_reward"] if goal else -L["step_cost"]
                agent.observe(s, a, r, s2, goal)
                tds.append(agent.learn_online())
                t = t0 + (k + 1) * step_ms
                x, y = world.cell_center(s2[0], s2[1])
                locs.append((trial, t, x, y, s2[2]))
                spikes.append((ACTION_IDS[s2[2]], t))
                s = s2
                if goal:
                    dop.append((1, t))
                    row.update(success=1, steps_to_goal=k + 1, time_to_goal_ms=(k + 1) * step_ms)
                    if cfg["online"].get("learn_after_goal", True):
                        agent.learn_after_goal(int(round(dur / step_ms)) - (k + 1))
                    break
            agent.end_trial(trial, bool(row["success"]))
            row.update(eps=eps, mean_abs_td=float(np.mean(tds)) if tds else np.nan)

        elif typ == "rest":
            n_upd = int(round(dur / step_ms * cfg["rest"]["updates_per_step"]))
            rows = agent.rest(n_upd)
            dt = dur / max(n_upd, 1)
            for (k, trs, idx, n_rew, td) in rows:
                trs = trs if isinstance(trs, list) else [trs]
                replay_rows.append((trial, k, t0 + (k - 1) * dt, mode,
                                    ",".join(map(str, sorted(set(trs)))),
                                    ",".join(map(str, idx)), n_rew, td))
            row.update(rest_updates=len(rows),
                       mean_abs_td=float(np.mean([r[4] for r in rows])) if rows else np.nan)

        ck = {"trial": trial, **{f"param_{k}": v for k, v in agent.online.state_dict().items()}}
        if save_qmaps:
            ck["qmap"] = agent.q_map(world)
            ck["goal_mask"] = world.goal_mask
        np.savez_compressed(os.path.join(out, "checkpoints", f"trial_{trial:02d}.npz"), **ck)

        summary.append(row)
        if not quiet:
            extra = (f"success={row['success']} steps={row['steps_to_goal']} eps={row['eps']:.2f}"
                     if typ == "explore" else f"replay={mode} updates={row['rest_updates']}")
            print(f"  {name}  trial {trial:2d} {typ:8s} {extra}")
        t0 += dur

    # ---- write files ----------------------------------------------------------
    pd.DataFrame(locs, columns=["trial", "time", "x", "y", "heading"]).to_csv(
        os.path.join(out, "locs_time.dat"), sep="\t", index=False, float_format="%.5f")
    for fname, rows in (("dopamine_l2-0.gdf", dop), ("action-0.gdf", spikes)):
        with open(os.path.join(out, fname), "w") as f:
            f.write(GDF_HEADER)
            for sender, t in rows:
                f.write(f"{sender}\t{t:.3f}\n")
    pd.DataFrame(replay_rows, columns=["trial", "update", "time_ms", "mode", "source_trials",
                                       "step_indices", "n_rewarded", "abs_td"]).to_csv(
        os.path.join(out, "replay_log.dat"), sep="\t", index=False)
    pd.DataFrame(summary).to_csv(os.path.join(out, "summary.dat"), sep="\t", index=False)
    if not quiet:
        print(f"  {name}: done in {time.time() - t_start:.1f} s -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--replay", choices=REPLAY_MODES, help="override rest.replay_mode")
    ap.add_argument("--tag", help="extra suffix for the agent folder name")
    ap.add_argument("--data-dir", help="output folder (default: data/)")
    ap.add_argument("--sim", help="sim_params.json (default: config/)")
    ap.add_argument("--env", help="env_params.json (default: config/)")
    ap.add_argument("--dqn", help="dqn_params.json (default: config/)")
    ap.add_argument("--no-qmaps", action="store_true", help="do not store Q-maps in checkpoints")
    ap.add_argument("--quiet", action="store_true", help="no per-trial printout")
    a = ap.parse_args()
    run(a.seed, a.replay, a.tag, a.data_dir, a.sim, a.env, a.dqn, save_qmaps=not a.no_qmaps,
        quiet=a.quiet)
