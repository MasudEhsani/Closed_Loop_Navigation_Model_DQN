"""Finite-difference check of the hand-written backpropagation.  Run:  python tests/test_gradients.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent.networks import QNetwork  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    cfg = {"conv1_channels": 3, "conv1_kernel": 5, "conv2_channels": 4, "conv2_kernel": 3,
           "pool": 4, "hidden": 6}
    net = QNetwork((4, 4, 8), 3, cfg, rng)
    x = rng.random((5, 4, 4, 8))
    w = rng.standard_normal((5, 3))                    # loss = sum(w * q)

    q, cache = net.forward(x, keep=True)
    grads = net.backward(w, cache)

    worst = 0.0
    for name, P in net.p.items():
        for _ in range(6):
            i = tuple(rng.integers(s) for s in P.shape)
            old = P[i]
            P[i] = old + 1e-6
            lp = (w * net.forward(x)).sum()
            P[i] = old - 1e-6
            lm = (w * net.forward(x)).sum()
            P[i] = old
            num = (lp - lm) / 2e-6
            err = abs(num - grads[name][i]) / max(1e-6, abs(num) + abs(grads[name][i]))
            worst = max(worst, err)
    print(f"largest relative gradient error: {worst:.2e}")
    assert worst < 1e-4, "backpropagation does not match finite differences"
    print("gradient check passed")


if __name__ == "__main__":
    main()
