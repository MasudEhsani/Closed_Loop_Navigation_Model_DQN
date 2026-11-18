"""
Q-network in plain numpy (no PyTorch needed), with hand-written backpropagation.

    observation (B, 4 views, 4 channels, L columns)
        |  same encoder for every view (shared weights)
        v
    Conv1d(4 -> c1, k1) -> ReLU -> Conv1d(c1 -> c2, k2) -> ReLU -> AvgPool(pool)
        |  c2 * L/pool features per view
        v
    concat 4 views -> Linear -> ReLU -> Linear -> Q(s, a) for each action

The CNN is the first half of the Q-network: the TD error is backpropagated through
the head into the convolution filters, so the visual features are learned from reward.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------- layers
def conv1d_forward(x, W, b):
    """x (N, Cin, L), W (Cout, Cin, k), 'same' padding, stride 1."""
    k = W.shape[2]
    pad = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    patches = sliding_window_view(xp, k, axis=2)           # (N, Cin, L, k)
    out = np.einsum("nclk,ock->nol", patches, W, optimize=True) + b[None, :, None]
    return out, patches


def conv1d_backward(dout, patches, W, L):
    k = W.shape[2]
    pad = k // 2
    dW = np.einsum("nol,nclk->ock", dout, patches, optimize=True)
    db = dout.sum(axis=(0, 2))
    dpatch = np.einsum("nol,ock->nclk", dout, W, optimize=True)
    dxp = np.zeros((dout.shape[0], W.shape[1], L + 2 * pad))
    for j in range(k):
        dxp[:, :, j:j + L] += dpatch[:, :, :, j]
    return dxp[:, :, pad:pad + L], dW, db


# ---------------------------------------------------------------------------- network
class QNetwork:
    def __init__(self, obs_shape, n_actions, cfg, rng):
        V, C, L = obs_shape
        c1, k1 = cfg["conv1_channels"], cfg["conv1_kernel"]
        c2, k2 = cfg["conv2_channels"], cfg["conv2_kernel"]
        self.pool = cfg["pool"]
        assert L % self.pool == 0, "columns_per_view must be divisible by pool"
        H = cfg["hidden"]
        self.V, self.L = V, L
        feat = V * c2 * (L // self.pool)

        def he(shape, fan_in):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float64)

        self.p = {
            "W1": he((c1, C, k1), C * k1), "b1": np.zeros(c1),
            "W2": he((c2, c1, k2), c1 * k2), "b2": np.zeros(c2),
            "W3": he((feat, H), feat), "b3": np.zeros(H),
            "W4": he((H, n_actions), H) * 0.1, "b4": np.zeros(n_actions),
        }

    # -- parameters
    def copy_from(self, other):
        for k in self.p:
            self.p[k][...] = other.p[k]

    def state_dict(self):
        return {k: v.copy() for k, v in self.p.items()}

    def load_state_dict(self, d):
        for k in self.p:
            self.p[k][...] = d[k]

    # -- forward / backward
    def forward(self, x, keep=False):
        p = self.p
        B = x.shape[0]
        xv = x.reshape(B * self.V, x.shape[2], self.L)
        z1, P1 = conv1d_forward(xv, p["W1"], p["b1"])
        h1 = np.maximum(z1, 0)
        z2, P2 = conv1d_forward(h1, p["W2"], p["b2"])
        h2 = np.maximum(z2, 0)
        N, C2, L = h2.shape
        pooled = h2.reshape(N, C2, L // self.pool, self.pool).mean(-1)
        f = pooled.reshape(B, -1)
        z3 = f @ p["W3"] + p["b3"]
        h3 = np.maximum(z3, 0)
        q = h3 @ p["W4"] + p["b4"]
        if not keep:
            return q
        cache = dict(x_shape=x.shape, P1=P1, z1=z1, h1=h1, P2=P2, z2=z2, h2_shape=h2.shape,
                     f=f, z3=z3, h3=h3)
        return q, cache

    def backward(self, dq, c):
        p = self.p
        g = {}
        g["W4"] = c["h3"].T @ dq
        g["b4"] = dq.sum(0)
        dh3 = dq @ p["W4"].T
        dz3 = dh3 * (c["z3"] > 0)
        g["W3"] = c["f"].T @ dz3
        g["b3"] = dz3.sum(0)
        df = dz3 @ p["W3"].T
        N, C2, L = c["h2_shape"]
        dpooled = df.reshape(N, C2, L // self.pool)
        dh2 = np.repeat(dpooled, self.pool, axis=2) / self.pool
        dz2 = dh2 * (c["z2"] > 0)
        dh1, g["W2"], g["b2"] = conv1d_backward(dz2, c["P2"], p["W2"], L)
        dz1 = dh1 * (c["z1"] > 0)
        _, g["W1"], g["b1"] = conv1d_backward(dz1, c["P1"], p["W1"], L)
        return g


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=None):
        self.p, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, b1, b2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        if self.clip:
            norm = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            if norm > self.clip:
                grads = {k: g * (self.clip / norm) for k, g in grads.items()}
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)
