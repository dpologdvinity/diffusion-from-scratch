# Diffusion from scratch

A class-conditional **DDPM** with **classifier-free guidance** and **DDIM** sampling, written from scratch in PyTorch (no `diffusers` or other diffusion libraries) and trained entirely on a laptop CPU.

<!-- RESULTS -->

## What's implemented

| Piece | Where | Notes |
|---|---|---|
| Noise schedule + closed-form forward process | [`ddpm/schedule.py`](ddpm/schedule.py) | linear (default) and cosine β schedules, T = 1000 |
| Noise-prediction UNet | [`ddpm/unet.py`](ddpm/unet.py) | 28→14→7, ResBlocks + GroupNorm, attention at 7×7, sinusoidal time + learned class embeddings, 0.42M params |
| Training | [`ddpm/train.py`](ddpm/train.py) | ε-prediction MSE, 10% label dropout for CFG, EMA weights, data-parallel CPU training via `torchrun` |
| DDPM sampler | [`ddpm/sampling.py`](ddpm/sampling.py) | ancestral sampling from the true posterior, 1000 steps |
| DDIM sampler | [`ddpm/sampling.py`](ddpm/sampling.py) | any step count, η ∈ [0, 1]; η = 1 with consecutive steps reduces to DDPM (tested) |
| Classifier-free guidance | [`ddpm/sampling.py`](ddpm/sampling.py) | conditional and null-label passes batched into one forward call |
| Evaluation | [`ddpm/evaluate.py`](ddpm/evaluate.py), [`ddpm/classifier.py`](ddpm/classifier.py), [`ddpm/metrics.py`](ddpm/metrics.py) | class accuracy, FID and KID in a dataset-specific classifier's feature space, wall-clock timing |
| 2D toy experiments | [`ddpm/toy.py`](ddpm/toy.py) | 8 Gaussians, two moons, spiral with an MLP denoiser |

[`NOTES.md`](NOTES.md) walks through the math (forward process, ε-prediction, DDIM's non-Markovian reverse process, why CFG works) and maps each equation to the code.

## 2D toy distributions

The same schedule, samplers, and guidance code with a small MLP denoiser. Colors are classes; each model is class-conditional.

![toy samplers](results/toy_samplers.png)

Sliced Wasserstein distance to held-out data (lower is better; "real" is the floor from two independent draws of real data):

| dataset | DDPM 1000 | DDIM 50 | DDIM 10 | DDIM 5 | real | DDPM 1000 time | DDIM 50 time |
|---|---|---|---|---|---|---|---|
| 8gaussians | 0.029 | 0.031 | 0.033 | 0.043 | 0.017 | 34.2s | 1.4s |
| moons | 0.041 | 0.044 | 0.092 | 0.170 | 0.020 | 40.1s | 0.9s |
| spiral | 0.038 | 0.079 | 0.139 | 0.233 | 0.015 | 32.6s | 0.7s |

DDIM holds quality well down to ~10 steps on simple geometry (8 Gaussians) but degrades sooner on the spiral, where the trajectory curves more and large deterministic steps cut corners.

Classifier-free guidance on 8 Gaussians, asking for one mode. w = 0 is the unconditional model; w > 1 over-sharpens and pushes samples past the mode, the usual diversity-for-fidelity trade:

![toy guidance](results/toy_guidance.png)

## Reproduce

```bash
uv sync                                   # CPU-only PyTorch
uv run pytest                             # sampler math, schedule, CFG, UNet, metrics

uv run python -m ddpm.toy                 # 2D experiments (~5 min)

# MNIST: data-parallel over single-threaded CPU workers (batch size is per worker)
OMP_NUM_THREADS=1 uv run torchrun --standalone --nproc-per-node 5 -m ddpm.train \
    --dataset mnist --threads 1 --batch-size 26 --lr 4e-4 --warmup 200 --minutes 60

uv run python -m ddpm.classifier --dataset mnist   # evaluation classifier
scripts/run_eval.sh mnist 200 3                    # samples, metrics, figures, timing
```

## Evaluation caveats

- **FID here is not comparable to published FIDs.** Inception-v3 features suit 28×28 grayscale digits poorly, so FID and KID are computed on the 64-d penultimate features of a small CNN trained on MNIST. Scores are meaningful only relative to each other and to the "real images" floor at the same sample size.
- **Small sample sizes.** Samples were generated on a shared laptop CPU, so each config uses N = 200. FID is biased upward at small N (hence the equal-N real-image floor); KID is unbiased and is the more reliable of the two here.
- **Wall-clock timings** were measured on a machine running other jobs. Step counts are the load-independent measure; the timing table confirms the ratio holds in practice.

## References

Ho et al. 2020 (DDPM) · Song et al. 2021 (DDIM) · Ho & Salimans 2022 (classifier-free guidance) · Nichol & Dhariwal 2021 (cosine schedule) · Bińkowski et al. 2018 (KID)
