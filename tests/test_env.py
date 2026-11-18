"""Sanity checks of the grid world and the vision.  Run:  python tests/test_env.py"""
import json
import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from env import GridWorld, Vision  # noqa: E402


def main():
    envp = json.load(open(os.path.join(ROOT, "config", "env_params.json")))
    w = GridWorld(envp)
    v = Vision(envp, w)
    assert (w.nx, w.ny) == (24, 24)

    # moves and walls
    w.reset(-1.15, -1.15, heading=2)              # bottom-left corner, facing south
    _, _, info = w.step(0)
    assert info["bumped"] and w.state() == (0, 0, 2), "moving into a wall must not move"
    w.step(2)                                      # turn right -> west, bump
    assert w.state() == (0, 0, 3)
    w.step(2)                                      # turn right -> north, move
    assert w.state() == (0, 1, 0)

    # goal
    w.set_goal(0.5, 0.5, 0.2)
    assert w.goal_mask.sum() > 0 and w.goal_mask[w.pos_to_cell(0.5, 0.5)]

    # vision: in the middle facing north, the front view sees the north wall (red),
    # the back view the south wall (blue); distances front and back are equal
    ix, iy = w.pos_to_cell(0.05, 0.05)
    obs = v.render(ix, iy, 0)
    front, back = obs[0], obs[2]
    mid = obs.shape[2] // 2
    assert front[0, mid] > front[2, mid], "front should be red-ish (north wall)"
    assert back[2, mid] > back[0, mid], "back should be blue (south wall)"
    assert abs(front[3, mid] - back[3, mid]) < 0.05
    # turning: facing east, the left view is the north wall
    obs_e = v.render(ix, iy, 1)
    assert np.allclose(obs_e[1], obs[0]), "left view facing east == front view facing north"
    print("environment and vision checks passed")


if __name__ == "__main__":
    main()
