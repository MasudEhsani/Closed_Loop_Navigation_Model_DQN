#!/bin/bash
# Run DQN agents for seeds FIRST ... LAST, one folder per seed and replay mode.
#
#   ./run/run_sim.sh FIRST LAST [MODE ...]
#   ./run/run_sim.sh 7000 7019                       # all five modes
#   ./run/run_sim.sh 7000 7019 none reverse          # only these
#
# Modes: none random forward reverse shuffled. Agents with the same seed behave identically
# until the first rest, so modes are compared on matched agents.
# Set PLOTS=false to skip the per-agent figures.

if [ $# -lt 2 ]; then
    echo "usage: $0 FIRST LAST [MODE ...]" >&2
    exit 1
fi
FIRST=$1; LAST=$2; shift 2
MODES=("$@")
[ ${#MODES[@]} -eq 0 ] && MODES=(none random forward reverse shuffled)
PLOTS=${PLOTS:-true}

cd "$(dirname "$0")/.." || exit 1

for (( seed=FIRST; seed<=LAST; seed++ )); do
    for mode in "${MODES[@]}"; do
        echo "=== seed $seed, replay $mode"
        python3 run/run_agent.py --seed "$seed" --replay "$mode" || { echo "failed: $seed $mode" >&2; continue; }
        if [ "$PLOTS" = "true" ]; then
            python3 analysis/plot_agent.py "data/agent${seed}_${mode}" || echo "plots failed: $seed $mode" >&2
        fi
    done
done
