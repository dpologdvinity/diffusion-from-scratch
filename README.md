# Diffusion from scratch

[![tests](https://github.com/dpologdvinity/diffusion-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/dpologdvinity/diffusion-from-scratch/actions/workflows/tests.yml)

A class-conditional **DDPM** with **classifier-free guidance** and **DDIM** sampling, written from scratch in PyTorch (no `diffusers` or other diffusion libraries) and trained entirely on a laptop CPU.

## Try it

- **In the browser:** [interactive demo](https://dpologdvinity.github.io/diffusion-from-scratch/). Drag the guidance and sampler sliders over precomputed samples (every setting starts from the same noise) and watch the digits denoise.
- **Locally, with live sampling:** clone the repo, then `uv sync && make serve` and open http://127.0.0.1:8000. Pick any digit, seed, guidance weight, sampler, step count, and η; the model runs on one CPU thread. The trained weights ship with the repo (`models/mnist.pt`, 1.7 MB, EMA weights only), so no training is needed; a checkpoint you train yourself (`make train`) takes precedence. API docs are at `/api/docs`.

## Results (MNIST)

Class-conditional samples, DDIM 50 steps, guidance w = 2. Each row is the requested class:

![MNIST samples](results/mnist_samples.png)

**DDIM matches 1000-step DDPM at 20× fewer steps.** At guidance w = 2, DDIM with 50 steps reaches the same class accuracy as DDPM with 1000 (99.9%) at a lower FID (19.0 vs 28.0; real images score 3.1), while sampling 18–21× faster in wall-clock time. Even 10 steps keeps 99.9% accuracy.

| sampler | steps | class acc ↑ | FID ↓ | KID×1000 ↓ | wall-clock speedup |
|---|---|---|---|---|---|
| DDPM | 1000 | 99.9% | 28.0 | 132.1 | 1× |
| DDIM | 100 | 99.8% | 18.7 | 78.8 | — |
| DDIM | 50 | 99.9% | 19.0 | 80.7 | 18–21× |
| DDIM | 20 | 99.9% | 19.4 | 81.4 | 51× |
| DDIM | 10 | 99.9% | 18.6 | 72.2 | 113× |
| real images | — | 99.0% | 3.1 | 0.4 | — |

N = 1,000 samples per config. Speedups come from a sequential benchmark on one CPU thread (batch 10); DDIM 50 was timed before and after DDPM to bracket changes in background load, and the step ratio is exactly 20×.

**Why DDIM beats DDPM here: it's the noise, not the step count.** Turning DDIM's stochasticity back on (η) at a fixed 50 steps walks FID back up to DDPM's level; at 10 steps, full stochasticity hurts badly, because each large step injects noise the model has too few remaining steps to remove:

| sampler (w = 2) | class acc ↑ | FID ↓ |
|---|---|---|
| DDIM 50, η = 0 | 99.9% | 19.0 |
| DDIM 50, η = 0.5 | 99.7% | 22.6 |
| DDIM 50, η = 1 | 100.0% | 30.1 |
| DDPM 1000 | 99.9% | 28.0 |
| DDIM 10, η = 0 | 99.9% | 18.6 |
| DDIM 10, η = 1 | 100.0% | 36.4 |

Same starting noise, same label, different samplers. Deterministic DDIM (η = 0) gives nearly the same image at every step count; DDPM's stochastic steps take a different path and, at this guidance weight, produce visibly thinner strokes:

![sampler comparison](results/mnist_sampler_comparison.png)

![steps trade-off](results/mnist_steps_tradeoff.png)

**Classifier-free guidance trades diversity for fidelity.** w = 1 (the plain conditional model) gives by far the best FID (8.5) at 92.7% class accuracy; w = 2 buys 99.9% accuracy at FID 19.0; past that, samples stay on-class but FID climbs as strokes thicken and variety collapses. w = 0 is the unconditional model (chance accuracy), which saw only 10% of training batches:

| guidance w (DDIM 50) | class acc ↑ | FID ↓ |
|---|---|---|
| 0 | 10.7% | 76.4 |
| 1 | 92.7% | 8.5 |
| 2 | 99.9% | 19.0 |
| 3 | 100.0% | 45.4 |
| 5 | 100.0% | 79.0 |

Longer training shifted this trade-off: after 1,812 steps the plain conditional model reached only 75% accuracy and needed w = 2 to look right; after 15,351 steps it reaches 93%, so the FID-optimal guidance weight dropped from 2 to 1.

![guidance comparison](results/mnist_guidance_comparison.png)

Denoising trajectory (DDIM 50, x_t from pure noise to the final sample) and the training loss of the 0.42M-parameter UNet (15,351 steps, about 16 epochs, 3 hours in total on a shared laptop CPU):

![denoising](results/mnist_denoising.png)

![loss](results/mnist_loss.png)

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
| Interactive demo | [`web/`](web), [`ddpm/export_web.py`](ddpm/export_web.py), [`ddpm/serve.py`](ddpm/serve.py) | one vanilla-JS page: precomputed samples on GitHub Pages, live sampling through a local FastAPI server ([design](design/frontend.md)) |

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
uv sync                         # CPU-only PyTorch
make test                       # sampler math, schedule, CFG, UNet, metrics
make toy                        # 2D experiments (~5 min)
make train MINUTES=60           # MNIST, data-parallel over single-threaded CPU workers
make eval N=100                 # samples for every sampler config -> metrics, figures, timing
make web                        # regenerate the static demo in docs/ (GitHub Pages)
make serve                      # local demo with live sampling at http://127.0.0.1:8000
```

Variables: `DATASET=mnist|fashion`, `WORKERS`, `MINUTES`, `N`, `JOBS`; `make resume` continues from the last checkpoint. Long jobs run under `nice` with `OMP_NUM_THREADS=1`: on a contended CPU, single-threaded DDP workers sync once per step and scale far better than intra-op threads.

Larger evaluations can split the expensive DDPM run across processes; shards reproduce the unsharded samples exactly and are merged automatically:

```bash
for i in 0 1 2 3; do uv run python -m ddpm.evaluate generate --method ddpm --n 1000 --batch 50 --shard $i --num-shards 4 & done; wait
```

## Roadmap

- [x] DDPM, DDIM, classifier-free guidance, UNet, CPU data-parallel training
- [x] 2D toy experiments and MNIST evaluation (class accuracy, FID/KID, guidance sweep, timing)
- [x] Longer MNIST training run (15,351 steps, about 16 epochs)
- [ ] FashionMNIST model and evaluation
- [x] Larger evaluation (N = 1,000 per config) for tighter FID/KID estimates
- [x] DDIM η ablation (η ∈ {0, 0.5, 1})
- [ ] Cosine vs linear noise schedule (needs a full retrain)

## Evaluation caveats

- **FID here is not comparable to published FIDs.** Inception-v3 features suit 28×28 grayscale digits poorly, so FID and KID are computed on the 64-d penultimate features of a small CNN trained on the same dataset. Scores are meaningful only relative to each other and to the "real images" floor at the same sample size.
- **Sample size.** Each config uses N = 1,000 samples (the 1000-step DDPM run alone is 2M network evaluations with guidance, about 2.5 hours on one CPU core). FID is biased at finite N, so every config, including the real-image floor, uses the same N. An earlier N = 100 run was too small: on FashionMNIST it scored real images worse than generated ones.
- **Wall-clock timings** were measured on a machine running other jobs. Step counts are the load-independent measure; the timing table confirms the ratio holds in practice.

## References

Ho et al. 2020 (DDPM) · Song et al. 2021 (DDIM) · Ho & Salimans 2022 (classifier-free guidance) · Nichol & Dhariwal 2021 (cosine schedule) · Bińkowski et al. 2018 (KID)
