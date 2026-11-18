"""
Egocentric vision: four views (front, left, back, right relative to the heading),
each fov_deg wide, so together they cover 360 degrees.

Each view is cast as `columns_per_view` rays from the centre of the agent's cell. A ray
returns the colour of the wall texture where it hits, and the distance to that wall.

Observation shape: (4 views, 4 channels, columns)
  channels = R, G, B (0..1) and distance / max_distance (0..1)
  columns run from the left edge to the right edge of each view, as the agent sees it.

Because the world is deterministic, the observation of every (cell, heading) is
rendered once and cached.
"""

import numpy as np

WALLS = ["north", "east", "south", "west"]
VIEW_NAMES = ["front", "left", "back", "right"]


def heading_angle(h):
    """Heading index (0 N, 1 E, 2 S, 3 W) -> angle in degrees, counter-clockwise from +x."""
    return 90.0 - 90.0 * h


class Vision:
    def __init__(self, env_params, world):
        self.world = world
        v = env_params["vision"]
        self.fov = float(v["fov_deg"])
        self.ncol = int(v["columns_per_view"])
        self.dmax = float(v["max_distance"])
        self.walls = env_params["walls"]
        self.shape = (4, 4, self.ncol)
        self._cache = {}

    # ---- textures -------------------------------------------------------------
    def texture(self, wall, u):
        """Colour of `wall` at position u in [0, 1] along it (west->east or south->north)."""
        w = self.walls[wall]
        base = np.array(w["color"], float)
        p = w.get("pattern", "plain")
        if p == "stripes":
            on = (np.floor(u * (self.world.xmax - self.world.xmin) / w["period"] * 2) % 2) == 0
            return base if on else 0.45 * base
        if p == "block":
            if any(abs(u - c) <= w["width"] / 2 for c in w["centers"]):
                return np.array([0.95, 0.95, 0.95])
            return base
        if p == "dots":
            if any(abs(u - c) <= w["width"] / 2 for c in w["centers"]):
                return np.array([0.05, 0.05, 0.05])
            return base
        return base

    # ---- ray casting ----------------------------------------------------------
    def cast(self, x, y, angle_deg):
        """Ray from (x, y): returns (wall name, u along wall, distance)."""
        W = self.world
        a = np.deg2rad(angle_deg)
        dx, dy = np.cos(a), np.sin(a)
        hits = []
        if dx > 1e-12:
            hits.append(((W.xmax - x) / dx, "east"))
        if dx < -1e-12:
            hits.append(((W.xmin - x) / dx, "west"))
        if dy > 1e-12:
            hits.append(((W.ymax - y) / dy, "north"))
        if dy < -1e-12:
            hits.append(((W.ymin - y) / dy, "south"))
        t, wall = min(hits)
        hx, hy = x + t * dx, y + t * dy
        if wall in ("north", "south"):
            u = (hx - W.xmin) / (W.xmax - W.xmin)
        else:
            u = (hy - W.ymin) / (W.ymax - W.ymin)
        return wall, float(np.clip(u, 0, 1)), float(t)

    def column_angles(self, h, view):
        """Ray angles of one view, from its left edge to its right edge."""
        centre = heading_angle(h) + 90.0 * view          # front, left (+90), back, right (+270)
        half = self.fov / 2
        # column centres, left edge = centre + half (counter-clockwise is to the left)
        offs = half - (np.arange(self.ncol) + 0.5) * self.fov / self.ncol
        return centre + offs

    def render(self, ix, iy, h):
        key = (ix, iy, h)
        if key in self._cache:
            return self._cache[key]
        x, y = self.world.cell_center(ix, iy)
        obs = np.zeros(self.shape, np.float32)
        for v in range(4):
            for c, ang in enumerate(self.column_angles(h, v)):
                wall, u, d = self.cast(x, y, ang)
                obs[v, :3, c] = self.texture(wall, u)
                obs[v, 3, c] = min(d / self.dmax, 1.0)
        self._cache[key] = obs
        return obs

    def observe(self, state):
        return self.render(*state)
