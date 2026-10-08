#!/usr/bin/env bash
# Generate samples for every sampler config, then score them and draw figures.
# Each config is one single-threaded process at low priority; at most $JOBS run at once,
# so evaluation shares the machine politely with other work.
#
#   scripts/run_eval.sh mnist [N] [JOBS]
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${1:-mnist}
N=${2:-100}
JOBS=${3:-2}
W=2  # guidance weight for the step-count sweep

export OMP_NUM_THREADS=1
run() { nice -n 10 uv run python -m ddpm.evaluate "$@"; }

# DDPM (1000 steps) dominates the cost, so it is split into $JOBS shards that run in parallel;
# shards reproduce the unsharded samples exactly and are merged by `report`.
SHARD_BATCH=$(( (N + JOBS - 1) / JOBS ))
configs=()
for i in $(seq 0 $((JOBS - 1))); do
  configs+=("--method ddpm --steps 1000 --guidance $W --batch $SHARD_BATCH --shard $i --num-shards $JOBS")
done
configs+=(
  "--method ddim --steps 100 --guidance $W"
  "--method ddim --steps 50 --guidance $W"
  "--method ddim --steps 20 --guidance $W"
  "--method ddim --steps 10 --guidance $W"
  "--method ddim --steps 50 --guidance 0"
  "--method ddim --steps 50 --guidance 1"
  "--method ddim --steps 50 --guidance 3"
  "--method ddim --steps 50 --guidance 5"
)

mkdir -p results/runs/"$DATASET"
for cfg in "${configs[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  # shellcheck disable=SC2086
  run generate --dataset "$DATASET" --n "$N" --batch "$N" $cfg > /dev/null &  # later --batch wins
done
wait

run report --dataset "$DATASET" --sweep-guidance "$W"
run figures --dataset "$DATASET" --sweep-guidance "$W"
run timing --dataset "$DATASET" --guidance "$W" --batch 10
