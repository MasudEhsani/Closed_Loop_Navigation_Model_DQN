"""
Show what the agent sees: the four views (front, left, back, right) at given positions.

    python analysis/plot_views.py                       # a few default positions
    python analysis/plot_views.py -0.8 -0.8 N  0.5 0.5 E # x y heading triples

Each view is drawn as a strip of wall colour; the bar height shows nearness (1 - distance),
so near walls look tall. Saves analysis/views.png.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
from env import GridWorld, Vision, HEADINGS, VIEW_NAMES  # noqa: E402


def main(args):
    envp = json.load(open(os.path.join(ROOT, "config", "env_params.json")))
    world = GridWorld(envp)
    vis = Vision(envp, world)
    if args:
        pos = [(float(args[i]), float(args[i + 1]), args[i + 2]) for i in range(0, len(args), 3)]
    else:
        pos = [(-0.8, -0.8, "N"), (-0.8, -0.8, "E"), (0.5, 0.5, "N"), (1.1, 0.0, "W")]
    fig, axs = plt.subplots(len(pos), 4, figsize=(10, 1.6 * len(pos)), squeeze=False)
    for r, (x, y, h) in enumerate(pos):
        ix, iy = world.pos_to_cell(x, y)
        obs = vis.render(ix, iy, HEADINGS.index(h))
        for v in range(4):
            ax = axs[r, v]
            rgb, d = obs[v, :3].T, obs[v, 3]
            img = np.ones((40, obs.shape[2], 3))
            for c in range(obs.shape[2]):
                hgt = int(round((1 - d[c]) * 38)) + 2
                lo = (40 - hgt) // 2
                img[lo:lo + hgt, c] = rgb[c]
            ax.imshow(img, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(VIEW_NAMES[v])
            if v == 0:
                ax.set_ylabel(f"({x:+.1f},{y:+.1f})\nfacing {h}", fontsize=8)
    fig.tight_layout()
    out = os.path.join(ROOT, "analysis", "views.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
