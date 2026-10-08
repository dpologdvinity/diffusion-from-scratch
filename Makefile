# Long-running targets run at low priority with single-threaded workers so they
# share the CPU with other work. Override variables on the command line, e.g.
#   make train DATASET=fashion MINUTES=180 WORKERS=2
DATASET ?= mnist
MINUTES ?= 60
WORKERS ?= 2
N       ?= 100
JOBS    ?= 2

export OMP_NUM_THREADS = 1

.PHONY: test lint toy train resume eval web serve

test:
	uv run pytest -q

lint:
	uvx ruff check ddpm tests scripts

toy:
	nice -n 10 uv run python -m ddpm.toy

train:
	nice -n 10 uv run torchrun --standalone --nproc-per-node $(WORKERS) -m ddpm.train \
	  --dataset $(DATASET) --threads 1 --batch-size 26 --lr 4e-4 --warmup 200 \
	  --steps 100000 --minutes $(MINUTES) --log-every 50 --ckpt-every 250 --preview-every 1000

resume:
	nice -n 10 uv run torchrun --standalone --nproc-per-node $(WORKERS) -m ddpm.train \
	  --dataset $(DATASET) --threads 1 --batch-size 26 --lr 4e-4 --warmup 200 \
	  --steps 100000 --minutes $(MINUTES) --log-every 50 --ckpt-every 250 --preview-every 1000 --resume

eval:
	scripts/run_eval.sh $(DATASET) $(N) $(JOBS)

web:
	nice -n 10 uv run python -m ddpm.export_web

serve:
	uv run python -m ddpm.serve
