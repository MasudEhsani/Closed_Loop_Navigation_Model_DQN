"""
Discrete open field: the same square arena as the spiking model, cut into square cells.

State   : cell (ix, iy) and heading h in {0: N, 1: E, 2: S, 3: W}
Actions : 0 forward, 1 turn left, 2 turn right
          With turn_moves=True a turn is followed by one step forward, so every
          action moves the agent one cell (like the spiking agent, which keeps moving).
          A move into a wall leaves the agent in its cell (the turn still happens).
Goal    : every cell whose centre lies within goal_size1 of (goal_x, goal_y).
"""

import numpy as np

HEADINGS = ["N", "E", "S", "W"]
HEADING_VEC = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])   # cell steps per heading
ACTIONS = ["forward", "left", "right"]


class GridWorld:
    def __init__(self, env_params):
        a = env_params["arena"]
        self.xmin, self.xmax, self.ymin, self.ymax = a["xmin"], a["xmax"], a["ymin"], a["ymax"]
        self.cell = float(env_params["cell_size"])
        self.nx = int(round((self.xmax - self.xmin) / self.cell))
        self.ny = int(round((self.ymax - self.ymin) / self.cell))
        self.turn_moves = bool(env_params.get("turn_moves", True))
        self.initial_heading = HEADINGS.index(env_params.get("initial_heading", "N"))
        self.goal_mask = np.zeros((self.nx, self.ny), bool)
        self.ix = self.iy = 0
        self.h = self.initial_heading

    # ---- geometry -------------------------------------------------------------
    def cell_center(self, ix, iy):
        return (self.xmin + (ix + 0.5) * self.cell, self.ymin + (iy + 0.5) * self.cell)

    def pos_to_cell(self, x, y):
        ix = int(np.clip(np.floor((x - self.xmin) / self.cell), 0, self.nx - 1))
        iy = int(np.clip(np.floor((y - self.ymin) / self.cell), 0, self.ny - 1))
        return ix, iy

    def set_goal(self, goal_x, goal_y, goal_size):
        cx = self.xmin + (np.arange(self.nx) + 0.5) * self.cell
        cy = self.ymin + (np.arange(self.ny) + 0.5) * self.cell
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        self.goal_mask = np.hypot(X - goal_x, Y - goal_y) <= goal_size + 1e-9
        if not self.goal_mask.any():
            self.goal_mask[self.pos_to_cell(goal_x, goal_y)] = True

    # ---- dynamics ---------------------------------------------------------------
    def reset(self, start_x, start_y, heading=None):
        self.ix, self.iy = self.pos_to_cell(start_x, start_y)
        self.h = self.initial_heading if heading is None else int(heading)
        return self.state()

    def state(self):
        return (self.ix, self.iy, self.h)

    def in_goal(self):
        return bool(self.goal_mask[self.ix, self.iy])

    def step(self, action):
        """Returns (state, reached_goal, info)."""
        if action == 1:
            self.h = (self.h - 1) % 4
        elif action == 2:
            self.h = (self.h + 1) % 4
        moved = bumped = False
        if action == 0 or self.turn_moves:
            nx_, ny_ = self.ix + HEADING_VEC[self.h][0], self.iy + HEADING_VEC[self.h][1]
            if 0 <= nx_ < self.nx and 0 <= ny_ < self.ny:
                self.ix, self.iy = nx_, ny_
                moved = True
            else:
                bumped = True
        return self.state(), self.in_goal(), {"moved": moved, "bumped": bumped}
