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
# CKPT_DIR / RESULTS_DIR select a different model and output location (e.g. an ablation).
CKPT_DIR=${CKPT_DIR:-checkpoints}
RESULTS_DIR=${RESULTS_DIR:-results}
run() { nice -n 10 uv run python -m ddpm.evaluate "$@" --ckpt-dir "$CKPT_DIR" --results-dir "$RESULTS_DIR"; }

# DDPM (1000 steps) dominates the cost, so it is split into $JOBS shards that run in parallel;
# shards reproduce the unsharded samples exactly and are merged by `report`.
# Generation batch is capped so large N stays light on memory (seeds depend only on batch offset).
BATCH=$(( N < 250 ? N : 250 ))
SHARD_BATCH=$(( (N + JOBS - 1) / JOBS ))
SHARD_BATCH=$(( SHARD_BATCH < 250 ? SHARD_BATCH : 250 ))
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
  # eta ablation: stochastic DDIM (eta = 1 is DDPM-like noise injection on the subsequence)
  "--method ddim --steps 50 --guidance $W --eta 0.5"
  "--method ddim --steps 50 --guidance $W --eta 1"
  "--method ddim --steps 10 --guidance $W --eta 1"
)

# Start clean: leftover samples from an earlier run (e.g. a different shard count) would be merged in.
rm -rf "$RESULTS_DIR"/runs/"$DATASET"
mkdir -p "$RESULTS_DIR"/runs/"$DATASET"
for cfg in "${configs[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  # shellcheck disable=SC2086
  run generate --dataset "$DATASET" --n "$N" --batch "$BATCH" $cfg > /dev/null &  # a later --batch wins
done
wait

run report --dataset "$DATASET" --sweep-guidance "$W"
run figures --dataset "$DATASET" --sweep-guidance "$W"
run timing --dataset "$DATASET" --guidance "$W" --batch 10
