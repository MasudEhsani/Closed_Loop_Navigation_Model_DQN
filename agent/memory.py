"""
Two separate memories. Keeping them apart is what makes the rest/replay comparison meaningful.

WorkingWindow   used ONLY for online learning during behaviour. It holds the last
                `window` steps of the current trial (cleared at trial start by default),
                like an eligibility trace: a step can only be learned from while it is
                recent. It is not a long-term store, unlike the large random-sampled
                replay buffer of standard DQN.

EpisodicMemory  stores every exploration trial as an ordered sequence of steps.
                Used ONLY during rest. Rest replay never adds new experience; it only
                re-uses these stored sequences.

A transition is stored as states, not images: (state, action, reward, next_state, done)
with state = (ix, iy, heading). Images are re-rendered by the (deterministic) vision.
"""

from collections import deque
import numpy as np


class WorkingWindow:
    """Online-learning memory.
    sampling="recent"          batch = the last `size` steps (default; no long-term memory)
    sampling="uniform_buffer"  standard DQN: keep up to `size` steps across trials and draw
                               random batches of `batch` from them (continuous replay during
                               behaviour; a reference condition, NOT the spiking counterpart)"""
    def __init__(self, size, sampling="recent", batch=32, rng=None):
        self.buf = deque(maxlen=int(size))
        self.sampling, self.batch_size, self.rng = sampling, int(batch), rng

    def clear(self):
        self.buf.clear()

    def add(self, tr):
        self.buf.append(tr)

    def batch(self):
        if self.sampling == "uniform_buffer":
            pick = self.rng.integers(len(self.buf), size=min(self.batch_size, len(self.buf)))
            return [self.buf[i] for i in pick]
        return list(self.buf)

    def __len__(self):
        return len(self.buf)


class EpisodicMemory:
    def __init__(self):
        self.episodes = []          # dicts: trial, rewarded, transitions (list, in time order)

    def add_episode(self, trial, transitions, rewarded):
        if transitions:
            self.episodes.append({"trial": trial, "rewarded": bool(rewarded),
                                  "transitions": list(transitions)})

    def eligible(self, rewarded_only):
        return [e for e in self.episodes if e["rewarded"] or not rewarded_only]

    # ---- replay streams -------------------------------------------------------
    def episode_order(self, eps, order, rng):
        """Endless iterator over episodes."""
        if order == "recent_first":
            while True:
                for e in reversed(eps):
                    yield e
        else:  # "random"
            while True:
                yield eps[rng.integers(len(eps))]

    def sequence_batches(self, mode, batch, rewarded_only, order, rng):
        """Endless iterator of batches for mode 'forward', 'reverse' or 'shuffled'.
        A batch holds consecutive steps of one episode and never crosses episodes.
        'shuffled' replays the same episodes as the sequence modes, with the steps of
        each episode in random order (same content, no order)."""
        eps = self.eligible(rewarded_only)
        if not eps:
            return
        for e in self.episode_order(eps, order, rng):
            idx = np.arange(len(e["transitions"]))
            if mode == "reverse":
                idx = idx[::-1]
            elif mode == "shuffled":
                idx = rng.permutation(idx)
            for s in range(0, len(idx), batch):
                sel = idx[s:s + batch]
                yield e["trial"], sel, [e["transitions"][i] for i in sel]

    def random_batches(self, batch, rewarded_only, rng):
        """Endless iterator of batches drawn uniformly from all stored steps."""
        eps = self.eligible(rewarded_only)
        pool = [(e["trial"], i, t) for e in eps for i, t in enumerate(e["transitions"])]
        if not pool:
            return
        while True:
            pick = rng.integers(len(pool), size=batch)
            items = [pool[i] for i in pick]
            yield [it[0] for it in items], [it[1] for it in items], [it[2] for it in items]
