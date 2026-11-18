"""
Compare replay modes across agents (counterpart of meta_analysis/trial_collective.py).

    python analysis/compare_modes.py                    # all agents in data/
    python analysis/compare_modes.py data/ --seeds 7000 7019

Groups agent folders agent<seed>_<mode>[...] by mode. Only seeds present in every mode
are used, so the comparison is always between matched agents (same seed = identical
behaviour before the first rest). Writes data/fig-compare/:
  success_per_trial.png   mean success per exploration trial, rest trials marked
  rest_effect.png         success in the trials just before vs just after every rest
  rest_effect.dat         the numbers behind rest_effect.png

Tip: the agent folders use the spiking model's file format, so the spiking project's
meta_analysis/trial_collective.py can also read them, e.g. with
VARIANTS = {"No replay": ["7000_none"], "Reverse": ["7000_reverse"]}.
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ORDER = ["none", "random", "forward", "reverse", "shuffled"]


def collect(data_dir, seeds=None):
    runs = {}
    for d in glob.glob(os.path.join(data_dir, "agent*_*")):
        m = re.match(r"agent(\d+)_([a-z]+)(?:_.*)?$", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "summary.dat")):
            continue
        seed, mode = int(m.group(1)), m.group(2)
        if seeds and not (seeds[0] <= seed <= seeds[1]):
            continue
        runs.setdefault(mode, {})[seed] = pd.read_csv(os.path.join(d, "summary.dat"), sep="\t")
    common = set.intersection(*(set(v) for v in runs.values())) if runs else set()
    return {m: [runs[m][s] for s in sorted(common)] for m in runs}, sorted(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", nargs="?", default=os.path.join(ROOT, "data"))
    ap.add_argument("--seeds", nargs=2, type=int)
    ap.add_argument("--window", type=int, default=5, help="trials before/after each rest")
    a = ap.parse_args()

    runs, seeds = collect(a.data_dir, a.seeds)
    if not seeds:
        raise SystemExit("no seeds found in every mode")
    modes = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    out = os.path.join(a.data_dir, "fig-compare")
    os.makedirs(out, exist_ok=True)
    print(f"{len(seeds)} matched seeds, modes: {modes}")

    ref = runs[modes[0]][0]
    rest_trials = ref[ref["type"] == "rest"]["trial"].tolist()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    rows = []
    for m in modes:
        S = np.array([d["success"].to_numpy(float) for d in runs[m]])
        ex = (ref["type"] == "explore").to_numpy()
        t = ref["trial"].to_numpy()[ex]
        mean = pd.Series(S[:, ex].mean(0)).rolling(5, min_periods=1).mean()
        ax.plot(t, mean, label=m)
        for r in rest_trials:
            before = [i for i in range(len(ref)) if ex[i] and r - a.window <= ref["trial"][i] < r]
            after = [i for i in range(len(ref)) if ex[i] and r < ref["trial"][i] <= r + a.window]
            if before and after:
                b, af = S[:, before].mean(1), S[:, after].mean(1)
                rows.append({"mode": m, "rest_trial": r, "before": b.mean(), "after": af.mean(),
                             "change": (af - b).mean(), "change_sem": (af - b).std() / np.sqrt(len(b))})
    for r in rest_trials:
        ax.axvline(r, color="gray", alpha=0.3, lw=3)
    ax.set_xlabel("trial")
    ax.set_ylabel("success (5-trial mean)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=len(modes))
    ax.set_title(f"{len(seeds)} matched seeds; gray = rest trials")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success_per_trial.png"), dpi=150)
    plt.close(fig)

    eff = pd.DataFrame(rows)
    eff.to_csv(os.path.join(out, "rest_effect.dat"), sep="\t", index=False)
    if len(eff):
        g = eff.groupby("mode").agg(change=("change", "mean"), sem=("change_sem", "mean")).reindex(modes)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(g.index, g["change"], yerr=g["sem"], capsize=4, color="tab:blue")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(f"success after - before rest\n({a.window} trials each side)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "rest_effect.png"), dpi=150)
        plt.close(fig)
        print(g.round(3).to_string())
    print(f"figures in {out}")


if __name__ == "__main__":
    main()
