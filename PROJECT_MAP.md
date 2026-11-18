# Project map: DQN counterpart of the spiking navigation model

## 1. Goal

A DQN agent that solves the same task with the same trial structure as the spiking model
and writes the same output files, so both can be analysed side by side. Only the learner
changes; the arena, the schedule and the measurements stay the same.

Not included, by decision: Forced trials, weight scaling, dopamine gating of rest.

## 2. Spiking model and its DQN counterpart

| Spiking model | DQN counterpart |
|---|---|
| Arena −1.2 … 1.2, continuous position | Grid of 0.1 cells, 24 × 24 |
| Start, goal, `goal_size1` from `sim_params` | Same file; goal = all cells whose centre is within `goal_size1` of the goal (12 cells) |
| Place and border cells (fixed position code) | Vision of textured walls → CNN encoder (learned position code) |
| Layer 2 recurrent sheet | Hidden layers of the Q-network |
| Action ring (40 directions) | Q-values of 3 actions: forward, turn left + step, turn right + step (heading N/E/S/W) |
| One 20 ms closed-loop chunk | One step. Measured: the spiking agent moves ≈ 0.1 per 20 ms while exploring, i.e. one cell |
| `max_tr_dur` | Steps = `max_tr_dur` / 20 (1500 ms → 75 steps) |
| Dopamine at the goal | Reward +1 at the goal; the trial's behaviour ends there |
| Plasticity continues after the dopamine pulse until the trial ends | `online.learn_after_goal`: online updates on this trial's working window continue for the rest of the trial |
| Explore / rest trials | Same trial types and `trials_params.dat` |

## 3. Replay: three different things

1. **Replay during behaviour (standard DQN).** Standard DQN keeps a large buffer of all past
   steps and trains on random batches from it at every step. The spiking model has nothing
   like that; it learns online. Using it would mean the "no rest" agent already replays a
   lot. → Here online learning uses only a **working window**: the last 20 steps of the
   current trial, cleared at each trial start (a stand-in for eligibility traces).
   Standard DQN is available as a labelled reference (`online.sampling = "uniform_buffer"`).
2. **Rest replay (the counterpart of the rest trials).** No movement, no new experience;
   updates come only from an **episodic memory** of whole past trials. Modes:
   none / random / forward / reverse / shuffled.
3. **Generated replay** (the spiking network produces its own sequences, possibly never
   experienced). Not built; a possible later stage (learned model + simulated paths).

Controls built in:
- every replay mode does the same number of updates with the same batch size;
- `shuffled` replays exactly the episodes the sequence modes replay, without their order;
- `zero_rewards_control` and `rewarded_only` separate the effect of reward content;
- separate random-number streams (init / acting / replay / buffer): same seed → identical
  behaviour until the first rest, in every mode.

Rest details: `rest.updates_per_step` updates per 20 ms of rest (3000 ms × 4 = 600 updates of
8 steps). The target network is synced after every rest update (`target_sync_updates = 1`),
so value can travel along a replayed sequence; the same for all modes.

## 4. Environment

- 24 × 24 cells; heading N/E/S/W; a move into a wall leaves the agent in place.
- Walls: north red stripes, east green with a white block, south plain blue, west yellow
  with two black dots — so the agent can tell walls apart and where along a wall it looks.
- Vision every step: 4 views (front, left, back, right) of 90°, 32 ray-cast columns each,
  channels R, G, B, distance. Observation shape (4, 4, 32).

## 5. Agent

- Q-network: Conv1d(4→8, k5) → ReLU → Conv1d(8→16, k3) → ReLU → AvgPool(8), shared by the 4
  views → concat (256) → 64 → 3 Q-values. The TD error trains CNN and head together.
- Double DQN targets, Huber loss, Adam (lr 1e-3), target network.
- ε-greedy; the random action is "forward" with probability 0.75 (persistent random walk,
  like the smooth runs of the spiking agent; uniform random actions reach the goal in only
  ~1.5 % of trials, forward-biased ones in ~10 %).

## 6. First results (4 seeds per mode, preliminary)

Schedule: 8 × (10 exploration trials + 1 rest of 3000 ms). Mean success per block of 10
exploration trials (the rest comes after each block):

| Mode | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| none | .12 | .02 | .05 | .05 | .02 | .00 | .10 | .00 |
| random | .12 | .12 | .08 | .10 | .18 | .30 | .25 | .30 |
| forward | .12 | .10 | .12 | .12 | .32 | .18 | .20 | .15 |
| reverse | .12 | .10 | .08 | .15 | .20 | .12 | .15 | .18 |
| shuffled | .12 | .08 | .08 | .10 | .12 | .25 | .25 | .30 |

What this shows so far:
- **Online learning alone does not learn this task** (even over 150 trials). Standard DQN
  with replay during behaviour learns it fully (100 % success after ~40 trials). In a
  DQN, most of the learning is done by replay — unlike the spiking network.
- **Rest replay helps**: every replay mode beats `none`.
- **Order**: random and shuffled ≥ forward and reverse. Shuffled replays the same steps as
  the sequence modes, so the difference is the order. A likely reason: consecutive steps
  are strongly correlated, and correlated batches make gradient learning in a neural
  network less effective. In a table-based learner reverse order would be best; in a DQN
  that advantage can be hidden by this correlation effect. Keep this in mind when
  comparing sequence replay between the two models.
- Values rise around the goal after each rest and fade during the following exploration
  (see `value_maps.png`): online updates from unrewarded trials overwrite what rest added.

These are 4 seeds; run 20 per mode (`bash run/run_sim.sh 7000 7019`) before drawing conclusions.

## 7. Next steps

1. 20 seeds per mode; `analysis/compare_modes.py`.
2. Sequence replay without the correlation problem: replay sequences in batches that mix
   steps from several episodes at the same position in the sequence.
3. Random start positions (counterpart of the spiking 9000 series).
4. Generated replay (section 3, point 3).
