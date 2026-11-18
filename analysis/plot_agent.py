"""
Figures for one DQN agent (counterpart of analysis/analysis_runner.py in the spiking project).

    python analysis/plot_agent.py data/agent7000_reverse

Writes to data/fig-<agent folder name>/:
  trajectories.png   path of every exploration trial (green = reached the goal)
  learning.png       success and steps-to-goal per trial, rest trials marked
  value_maps.png     value map max_a Q(cell, heading, a) (max over heading) with greedy
                     arrows, before and after every rest (counterpart of the vector fields)
"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from env import GridWorld, HEADING_VEC  # noqa: E402


def load(agent_dir):
    envp = json.load(open(os.path.join(agent_dir, "env_params.json")))
    trials = pd.read_csv(os.path.join(agent_dir, "trials_params.dat"), sep="\t")
    locs = pd.read_csv(os.path.join(agent_dir, "locs_time.dat"), sep="\t")
    summ = pd.read_csv(os.path.join(agent_dir, "summary.dat"), sep="\t")
    return envp, trials, locs, summ


def draw_arena(ax, world, goal_mask=None):
    ax.set_xlim(world.xmin, world.xmax)
    ax.set_ylim(world.ymin, world.ymax)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if goal_mask is not None:
        ix, iy = np.nonzero(goal_mask)
        for i, j in zip(ix, iy):
            x, y = world.cell_center(i, j)
            ax.add_patch(plt.Rectangle((x - world.cell / 2, y - world.cell / 2), world.cell, world.cell,
                                       color="gold", alpha=0.5, lw=0))


def plot_trajectories(agent_dir, out, world, trials, locs, summ):
    ex = trials[trials["type"] == "explore"]["trial_num"].tolist()
    n = len(ex)
    cols = min(10, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.6 * rows), squeeze=False)
    tr0 = trials[trials["type"] == "explore"].iloc[0]
    world.set_goal(tr0["goal_x"], tr0["goal_y"], tr0["goal_size1"])
    succ = summ.set_index("trial")["success"]
    for ax, t in zip(axs.flat, ex):
        draw_arena(ax, world, world.goal_mask)
        L = locs[locs["trial"] == t]
        ax.plot(L["x"], L["y"], "-", lw=0.8, color="tab:green" if succ.get(t, 0) else "tab:gray")
        ax.plot(L["x"].iloc[0], L["y"].iloc[0], "o", ms=2.5, color="k")
        ax.set_title(f"tr {t}", fontsize=7)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "trajectories.png"), dpi=150)
    plt.close(fig)


def plot_learning(out, trials, summ):
    s = summ.copy()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    ex = s[s["type"] == "explore"]
    a1.plot(ex["trial"], ex["success"].rolling(5, min_periods=1).mean(), "-o", ms=2)
    a1.set_ylabel("success (5-trial mean)")
    a1.set_ylim(-0.05, 1.05)
    a2.plot(ex["trial"], ex["steps_to_goal"], "o", ms=3)
    a2.set_ylabel("steps to goal")
    a2.set_xlabel("trial")
    for t in s[s["type"] == "rest"]["trial"]:
        for a in (a1, a2):
            a.axvline(t, color="gray", alpha=0.4, lw=4)
    a1.set_title("gray bars = rest trials")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learning.png"), dpi=150)
    plt.close(fig)


def plot_value_maps(agent_dir, out, world, trials):
    files = sorted(glob.glob(os.path.join(agent_dir, "checkpoints", "trial_*.npz")))
    if not files:
        return
    ck = {int(os.path.basename(f)[6:8]): f for f in files}
    rest = trials[trials["type"] == "rest"]["trial_num"].tolist()
    picks = []
    for r in rest:
        if r - 1 in ck:
            picks.append((f"before rest (tr {r - 1})", ck[r - 1]))
        if r in ck:
            picks.append((f"after rest (tr {r})", ck[r]))
    if not picks:
        picks = [(f"trial {k}", f) for k, f in list(ck.items())[-1:]]
    picks = [p for p in picks if "qmap" in np.load(p[1]).files]
    if not picks:
        return
    cols = min(len(picks), 4)
    rows = int(np.ceil(len(picks) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)
    vmax = max(float(np.load(f)["qmap"].max()) for _, f in picks)
    for ax, (title, f) in zip(axs.flat, picks):
        d = np.load(f)
        Q = d["qmap"]                                    # (nx, ny, 4, 3)
        V = Q.max(axis=(2, 3))
        ax.imshow(V.T, origin="lower", extent=(world.xmin, world.xmax, world.ymin, world.ymax),
                  cmap="viridis", vmin=0, vmax=vmax)
        # greedy arrow per cell: best (heading, action) -> direction of the resulting move
        flat = Q.reshape(world.nx, world.ny, -1).argmax(-1)
        h, a = flat // Q.shape[3], flat % Q.shape[3]
        h_after = np.where(a == 1, (h - 1) % 4, np.where(a == 2, (h + 1) % 4, h))
        U, W = HEADING_VEC[h_after, 0], HEADING_VEC[h_after, 1]
        cx = world.xmin + (np.arange(world.nx) + 0.5) * world.cell
        cy = world.ymin + (np.arange(world.ny) + 0.5) * world.cell
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        ax.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], W[::2, ::2], color="w", scale=30, width=0.006)
        gm = d["goal_mask"]
        ix, iy = np.nonzero(gm)
        ax.plot([world.cell_center(i, j)[0] for i, j in zip(ix, iy)],
                [world.cell_center(i, j)[1] for i, j in zip(ix, iy)], "s", color="r", ms=3, alpha=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axs.flat)[len(picks):]:
        ax.axis("off")
    fig.suptitle("value map max Q and greedy direction")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "value_maps.png"), dpi=150)
    plt.close(fig)


def main(agent_dir):
    agent_dir = os.path.abspath(agent_dir)
    envp, trials, locs, summ = load(agent_dir)
    world = GridWorld(envp)
    out = os.path.join(os.path.dirname(agent_dir), "fig-" + os.path.basename(agent_dir).replace("agent", "", 1))
    os.makedirs(out, exist_ok=True)
    plot_trajectories(agent_dir, out, world, trials, locs, summ)
    plot_learning(out, trials, summ)
    plot_value_maps(agent_dir, out, world, trials)
    print(f"figures in {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        main(d)
