# data/

Filled by `run/run_agent.py` and `run/run_sim.sh`:

```
data/
├── agent<seed>_<mode>/     one agent (see the top-level README for the files)
├── fig-<seed>_<mode>/      its figures (analysis/plot_agent.py)
└── fig-compare/            replay modes compared (analysis/compare_modes.py)
```

To read these agents with the spiking project's `meta_analysis/trial_collective.py`, point it
at this folder (`python trial_collective.py ~/Closed_Navigation_Modell_DQN/data`) and name
agents as strings, e.g. `VARIANTS = {"No replay": ["7000_none"], "Reverse": ["7000_reverse"]}`.
