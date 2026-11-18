"""
DQN agent with separate online learning (behaviour) and rest replay (offline).

Online (exploration trials):
    act epsilon-greedily, store the step in the working window and in the current episode,
    then update the Q-network on the steps in the working window, bootstrapping from the
    target network (synced every `online.target_sync_steps` steps).

Rest (rest trials): no movement, no new experience. `rest_updates` gradient updates on
    batches taken from episodic memory, in the order set by `rest.replay_mode`:
      none      no updates (the rest trial is a pause)
      random    steps drawn uniformly from memory (plain DQN-style replay)
      forward   whole episodes replayed start -> goal
      reverse   whole episodes replayed goal -> start
      shuffled  same episodes as forward/reverse, steps in random order (order control)
    During rest the target network is synced every `rest.target_sync_updates` updates
    (default 1 = bootstrap from the network as it was just before this update), so value
    can travel along a replayed sequence. The same setting is used for every mode, and
    every mode except 'none' performs the same number of updates of the same batch size,
    so modes differ only in WHAT is replayed and in WHICH ORDER.

Random number streams are separate (init / acting / replay), so agents with the same seed
behave identically until the first rest, whatever the replay mode.
"""

import numpy as np
from .networks import QNetwork, Adam
from .memory import WorkingWindow, EpisodicMemory

REPLAY_MODES = ("none", "random", "forward", "reverse", "shuffled")


class DQNAgent:
    def __init__(self, vision, n_actions, cfg, seed):
        self.vision = vision
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng_init = np.random.default_rng([seed, 1])
        self.rng_act = np.random.default_rng([seed, 2])
        self.rng_replay = np.random.default_rng([seed, 3])

        self.online = QNetwork(vision.shape, n_actions, cfg["network"], self.rng_init)
        self.target = QNetwork(vision.shape, n_actions, cfg["network"], self.rng_init)
        self.target.copy_from(self.online)
        L = cfg["learning"]
        self.gamma, self.delta = L["gamma"], L["huber_delta"]
        self.opt = Adam(self.online.p, lr=L["lr"], clip=L["grad_clip"])

        O = cfg["online"]
        self.rng_buffer = np.random.default_rng([seed, 4])
        self.window = WorkingWindow(O["window"], O.get("sampling", "recent"),
                                    O.get("batch", 32), self.rng_buffer)
        self.memory = EpisodicMemory()
        self.current = []
        self.online_steps = 0
        self.rest_updates_total = 0

        mode = cfg["rest"]["replay_mode"]
        if mode not in REPLAY_MODES:
            raise ValueError(f"replay_mode must be one of {REPLAY_MODES}, got {mode!r}")

    # ------------------------------------------------------------------ acting
    def q_values(self, states):
        obs = np.stack([self.vision.observe(s) for s in states])
        return self.online.forward(obs)

    def act(self, state, eps):
        """Epsilon-greedy. The random action is 'forward' with probability
        exploration.random_forward_prob, otherwise left or right with equal chance, which
        gives a persistent random walk (like the smooth runs of the spiking agent) rather
        than jitter. Three random numbers are drawn every step, so the acting stream does
        not depend on the outcome."""
        u, u_fwd, u_turn = self.rng_act.random(3)
        if u < eps:
            pf = self.cfg["exploration"].get("random_forward_prob", 1.0 / self.n_actions)
            if u_fwd < pf:
                return 0
            return 1 if u_turn < 0.5 else 2
        return int(np.argmax(self.q_values([state])[0]))

    # ------------------------------------------------------------------ learning core
    def _update(self, transitions):
        S = np.stack([self.vision.observe(t[0]) for t in transitions])
        A = np.array([t[1] for t in transitions])
        R = np.array([t[2] for t in transitions], float)
        S2 = np.stack([self.vision.observe(t[3]) for t in transitions])
        D = np.array([t[4] for t in transitions], float)

        q, cache = self.online.forward(S, keep=True)
        q2 = self.target.forward(S2)
        if self.cfg["learning"].get("double_dqn", True):
            # Double DQN: the online network picks the next action, the target network
            # values it. Reduces the over-estimation of plain max-bootstrapping.
            a2 = self.online.forward(S2).argmax(axis=1)
            v2 = q2[np.arange(len(a2)), a2]
        else:
            v2 = q2.max(axis=1)
        y = R + self.gamma * (1.0 - D) * v2
        td = q[np.arange(len(A)), A] - y
        g = np.clip(td, -self.delta, self.delta)          # gradient of the Huber loss
        dq = np.zeros_like(q)
        dq[np.arange(len(A)), A] = g / len(A)
        self.opt.step(self.online.backward(dq, cache))
        return float(np.mean(np.abs(td)))

    # ------------------------------------------------------------------ online (behaviour)
    def start_trial(self):
        self.current = []
        if self.cfg["online"].get("clear_window_each_trial", True):
            self.window.clear()

    def observe(self, s, a, r, s2, done):
        tr = (s, int(a), float(r), s2, bool(done))
        self.window.add(tr)
        self.current.append(tr)

    def learn_online(self):
        td = None
        for _ in range(self.cfg["online"]["updates_per_step"]):
            td = self._update(self.window.batch())
            self.online_steps += 1
            if self.online_steps % self.cfg["online"]["target_sync_steps"] == 0:
                self.target.copy_from(self.online)
        return td

    def learn_after_goal(self, n_steps):
        """Counterpart of the spiking runner, which keeps simulating (with plasticity on)
        for the rest of the trial after the dopamine pulse: the agent stands still and
        keeps updating on its working window (this trial's recent steps only) for the
        remaining steps of the trial. No long-term memory is used."""
        for _ in range(max(0, n_steps)):
            self.learn_online()

    def end_trial(self, trial, rewarded):
        self.memory.add_episode(trial, self.current, rewarded)
        self.current = []

    # ------------------------------------------------------------------ rest (offline)
    def rest(self, n_updates):
        """Run one rest trial. Returns log rows (update, trials, indices, n_rewarded, |td|)."""
        R = self.cfg["rest"]
        mode = R["replay_mode"]
        rows = []
        if mode == "none" or n_updates <= 0:
            return rows
        if mode == "random":
            stream = self.memory.random_batches(R["batch"], R["rewarded_only"], self.rng_replay)
        else:
            stream = self.memory.sequence_batches(mode, R["batch"], R["rewarded_only"],
                                                  R["episode_order"], self.rng_replay)
        stream = iter(stream)
        for k in range(n_updates):
            try:
                trials, idx, batch = next(stream)
            except StopIteration:                       # nothing stored yet
                break
            if R.get("zero_rewards_control", False):
                batch = [(s, a, 0.0, s2, d) for (s, a, r, s2, d) in batch]
            td = self._update(batch)
            self.rest_updates_total += 1
            if (k + 1) % R["target_sync_updates"] == 0:
                self.target.copy_from(self.online)
            n_rew = sum(1 for t in batch if t[2] > 0)
            rows.append((k + 1, trials, list(map(int, idx)), n_rew, td))
        self.target.copy_from(self.online)
        return rows

    # ------------------------------------------------------------------ inspection
    def q_map(self, world):
        """Q-values of every (cell, heading): array (nx, ny, 4, n_actions)."""
        states = [(ix, iy, h) for ix in range(world.nx) for iy in range(world.ny) for h in range(4)]
        out = []
        for s in range(0, len(states), 512):
            out.append(self.q_values(states[s:s + 512]))
        return np.concatenate(out).reshape(world.nx, world.ny, 4, self.n_actions)
