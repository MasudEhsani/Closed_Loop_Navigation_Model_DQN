# Closed_Navigation_Modell_DQN

A Deep Q-Network (DQN) counterpart of the spiking closed-loop navigation model
(`~/Closed_Loop_Navigation_Model`). Same arena, start, goal and trial schedule; the agent
moves on a discrete grid and sees textured walls in four directions. The question it is
built for: **what does replay during rest add, and does the order of replay matter?**

Plain numpy, matplotlib, pandas: no PyTorch, no NEST. The network and its backpropagation
are written by hand and checked against numerical gradients (`tests/test_gradients.py`).

See [`PROJECT_MAP.md`](PROJECT_MAP.md) for the design, the mapping to the spiking model,
and the reasoning about replay.

## Quick start

```bash
cd ~/Closed_Navigation_Modell_DQN
conda env create -f environment.yml && conda activate dqn-nav   # or any env with numpy, pandas, matplotlib

python tests/test_env.py          # environment and vision checks
python tests/test_gradients.py    # backpropagation check
python analysis/plot_views.py     # what the agent sees -> analysis/views.png

python run/run_agent.py --seed 7000 --replay reverse            # one agent (~40 s)
python analysis/plot_agent.py data/agent7000_reverse             # its figures

bash run/run_sim.sh 7000 7019                                     # 20 seeds x 5 replay modes
python analysis/compare_modes.py                                  # compare the modes
```

## Folder layout

| Path | Content |
|---|---|
| `config/sim_params.json` | Trial schedule, same format as the spiking model (explore / rest blocks, goal, start, durations in ms) |
| `config/env_params.json` | Arena, cell size (0.1), ms per step (20), wall textures, vision |
| `config/dqn_params.json` | Network, learning, online memory, exploration, rest replay |
| `env/gridworld.py` | Discrete arena: cells, heading, moves, walls, goal |
| `env/vision.py` | Ray casting: four 90° views of the wall textures plus distance |
| `agent/networks.py` | CNN encoder (shared over the 4 views) + MLP head = Q-network; Adam |
| `agent/memory.py` | Working window (online learning) and episodic memory (rest only) |
| `agent/dqn_agent.py` | Acting, online learning, rest replay modes |
| `run/trial_generator.py` | Trial table from `sim_params.json` (same logic as the spiking model) |
| `run/run_agent.py` | Runs one agent through the schedule and writes its data |
| `run/run_sim.sh` | Many seeds x replay modes |
| `analysis/plot_agent.py` | Trajectories, learning curve, value maps before/after each rest |
| `analysis/plot_views.py` | The four views at chosen positions |
| `analysis/compare_modes.py` | Replay modes compared on matched seeds; rest effect |
| `tests/` | Environment and gradient checks |
| `data/` | One folder per agent: `agent<seed>_<mode>/`, figures in `fig-<seed>_<mode>/` |

## Output of one agent (`data/agent<seed>_<mode>/`)

Files in the **spiking model's format**, so its `meta_analysis/trial_collective.py` can read them:

| File | Content |
|---|---|
| `trials_params.dat` | Trial table (same columns as the spiking model) |
| `locs_time.dat` | `trial time x y heading`, one row per 20 ms step (cell centres) |
| `dopamine_l2-0.gdf` | Time of every goal arrival (the reward; "sender" is always 1) |
| `action-0.gdf`, `actionID_dir.dat` | One "spike" per step from the direction moved (4 ids: N, E, S, W) |

DQN-specific files:

| File | Content |
|---|---|
| `summary.dat` | Per trial: success, steps and time to goal, ε, mean \|TD error\|, number of rest updates |
| `replay_log.dat` | Every rest update: which trial and which steps were replayed, rewards in the batch, \|TD error\| |
| `checkpoints/trial_XX.npz` | Network weights after every trial, and the Q-map (all cells x headings x actions) |
| `sim_params.json`, `env_params.json`, `dqn_params.json` | The exact settings used |

## Replay modes (`rest.replay_mode`, or `--replay`)

| Mode | During rest |
|---|---|
| `none` | Nothing is learned (the rest trial is a pause) |
| `random` | Single steps drawn at random from all stored trials |
| `forward` | Stored trials replayed start → goal |
| `reverse` | Stored trials replayed goal → start |
| `shuffled` | The same trials as forward/reverse, steps in random order (controls for order) |

All modes except `none` do the same number of updates with the same batch size, and agents
with the same seed behave identically until the first rest. Further controls:
`rest.rewarded_only` (replay only trials that reached the goal), `rest.zero_rewards_control`
(replay with rewards removed), `rest.episode_order` (`recent_first` or `random`).

`online.sampling = "uniform_buffer"` turns the agent into standard DQN (random replay during
behaviour as well). It is a reference condition, **not** the counterpart of the spiking model.
See PROJECT_MAP.md, section 3.
