"""
Builds the trial table from config/sim_params.json, using the same logic and the same
columns as network/trial_generator.py of the spiking model:

    for every rest phase:  trials_per_explore x explore,  then trials_per_rest x rest

Run on its own to print the table:   python run/trial_generator.py [path/to/sim_params.json]
"""

import json
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM = os.path.join(HERE, "..", "config", "sim_params.json")


def build_trials(sim):
    tp = dict(sim["trial_params"])
    tp["max_tr_dur"] = list(tp["max_tr_dur"])
    tp["max_tr_dur"][0] = sim["explore"]["max_tr_dur"]
    tp["max_tr_dur"][1] = sim["rest"]["max_tr_dur"]

    indexer = []
    for _ in range(int(sim["rest"]["num_of_rest_phases"])):
        if sim["explore"]["num_of_explore_phases"] > 0:
            indexer += [0] * int(sim["explore"]["trials_per_explore"])
        indexer += [1] * int(sim["rest"]["trials_per_rest"])

    table = {"trial_num": range(1, len(indexer) + 1)}
    for key, values in tp.items():
        table[key] = [values[i] for i in indexer]
    return pd.DataFrame(table)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SIM
    with open(path) as f:
        print(build_trials(json.load(f)).to_string(index=False))
