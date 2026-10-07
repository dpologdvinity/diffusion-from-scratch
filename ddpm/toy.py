"""Class-conditional diffusion on 2D toy distributions.

The same schedule, samplers, and guidance code as the image model, with an MLP denoiser.
Small enough to train in about a minute on CPU, and easy to inspect visually.

    uv run python -m ddpm.toy
"""

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling import sample
from .schedule import NoiseSchedule
from .unet import timestep_embedding


def eight_gaussians(n: int, g: torch.Generator):
    y = torch.randint(0, 8, (n,), generator=g)
    angle = y.float() * 2 * math.pi / 8
    centers = torch.stack([angle.cos(), angle.sin()], dim=1)
    return centers + 0.05 * torch.randn(n, 2, generator=g), y


def two_moons(n: int, g: torch.Generator):
    y = torch.randint(0, 2, (n,), generator=g)
    s = torch.rand(n, generator=g) * math.pi
    x = torch.stack([s.cos(), s.sin()], dim=1)
    lower = torch.stack([1 - s.cos(), 0.5 - s.sin()], dim=1)
    x = torch.where(y[:, None] == 1, lower, x)
    x = (x - torch.tensor([0.5, 0.25])) / 1.2
    return x + 0.04 * torch.randn(n, 2, generator=g), y


def spiral(n: int, g: torch.Generator):
    y = torch.randint(0, 2, (n,), generator=g)
    r = torch.rand(n, generator=g).sqrt()
    theta = r * 3 * math.pi + y.float() * math.pi
    x = torch.stack([r * theta.cos(), r * theta.sin()], dim=1)
    return x + 0.03 * torch.randn(n, 2, generator=g), y


TOYS = {"8gaussians": (eight_gaussians, 8), "moons": (two_moons, 2), "spiral": (spiral, 2)}


class MLPDenoiser(nn.Module):
    def __init__(self, num_classes: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.hidden = hidden
        self.inp = nn.Linear(2, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.class_emb = nn.Embedding(num_classes + 1, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)])
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, x, t, y):
        h = self.inp(x) + self.time_mlp(timestep_embedding(t, self.hidden)) + self.class_emb(y)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


def sliced_wasserstein(a: torch.Tensor, b: torch.Tensor, n_proj: int = 256, seed: int = 0) -> float:
    """Average 1D Wasserstein-1 distance over random projections (equal sample sizes)."""
    g = torch.Generator().manual_seed(seed)
    dirs = F.normalize(torch.randn(n_proj, a.shape[1], generator=g), dim=1)
    pa, _ = (a @ dirs.T).sort(dim=0)
    pb, _ = (b @ dirs.T).sort(dim=0)
    return (pa - pb).abs().mean().item()


def nn_label_accuracy(x: torch.Tensor, y: torch.Tensor, ref_x: torch.Tensor, ref_y: torch.Tensor) -> float:
    """Fraction of samples whose nearest real point has the requested class."""
    nearest = torch.cdist(x, ref_x).argmin(dim=1)
    return (ref_y[nearest] == y).float().mean().item()


def train_toy(name: str, steps: int, sched: NoiseSchedule, seed: int = 0):
    fn, num_classes = TOYS[name]
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = MLPDenoiser(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        x0, y = fn(512, g)
        y = torch.where(torch.rand(y.shape, generator=g) < 0.1, torch.full_like(y, num_classes), y)
        t = torch.randint(0, sched.T, (512,), generator=g)
        noise = torch.randn(x0.shape, generator=g)
        loss = F.mse_loss(model(sched.q_sample(x0, t, noise), t, y), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return model, losses


def scatter(ax, x, y, num_classes, title):
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="tab10" if num_classes > 2 else "coolwarm", s=2, alpha=0.6, vmin=0, vmax=max(num_classes - 1, 1))
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sched = NoiseSchedule(1000)
    samplers = [("DDPM 1000", "ddpm", 1000), ("DDIM 50", "ddim", 50), ("DDIM 10", "ddim", 10), ("DDIM 5", "ddim", 5)]
    metrics, models = {}, {}
    fig, axes = plt.subplots(len(TOYS), 2 + len(samplers), figsize=(2.2 * (2 + len(samplers)), 2.3 * len(TOYS)))

    for row, name in enumerate(TOYS):
        fn, num_classes = TOYS[name]
        t0 = time.perf_counter()
        model, losses = train_toy(name, args.steps, sched)
        models[name] = model
        print(f"{name}: trained {args.steps} steps in {time.perf_counter() - t0:.0f}s, final loss {sum(losses[-200:]) / 200:.4f}")

        ref_x, ref_y = fn(args.n, torch.Generator().manual_seed(123))
        y = torch.arange(num_classes).repeat_interleave(math.ceil(args.n / num_classes))[: args.n]
        scatter(axes[row, 0], ref_x, ref_y, num_classes, f"{name}: data")
        noisy = sched.q_sample(ref_x, torch.full((args.n,), 300), torch.randn_like(ref_x))
        scatter(axes[row, 1], noisy, ref_y, num_classes, "forward noise, t=300")

        metrics[name] = {}
        for col, (label, method, steps) in enumerate(samplers, start=2):
            g = torch.Generator().manual_seed(1)
            t0 = time.perf_counter()
            x = sample(model, sched, (args.n, 2), y, method=method, steps=steps, guidance=args.guidance, clip=False, generator=g)
            secs = time.perf_counter() - t0
            m = {
                "nfe": steps,
                "seconds": round(secs, 2),
                "swd": round(sliced_wasserstein(x, fn(args.n, torch.Generator().manual_seed(7))[0]), 4),
                "class_acc": round(nn_label_accuracy(x, y, ref_x, ref_y), 4),
            }
            metrics[name][label] = m
            scatter(axes[row, col], x, y, num_classes, f"{label}  SWD={m['swd']:.3f}")
            print(f"  {label:10s} {m}")

        # Sliced-Wasserstein floor: two independent draws of real data.
        metrics[name]["real vs real"] = {"swd": round(sliced_wasserstein(ref_x, fn(args.n, torch.Generator().manual_seed(7))[0]), 4)}

    fig.tight_layout()
    fig.savefig(args.out / "toy_samplers.png", dpi=130)
    plt.close(fig)

    # Classifier-free guidance on 8 gaussians: ask for one mode, vary the guidance weight.
    sched_model = models["8gaussians"]
    ws = [0.0, 1.0, 3.0, 7.0]
    fig, axes = plt.subplots(1, len(ws), figsize=(2.4 * len(ws), 2.6))
    ref_x, ref_y = eight_gaussians(args.n, torch.Generator().manual_seed(123))
    metrics["8gaussians_guidance"] = {}
    for ax, w in zip(axes, ws):
        y = torch.zeros(500, dtype=torch.long)  # request mode 0 (the point at angle 0)
        x = sample(sched_model, sched, (500, 2), y, method="ddim", steps=50, guidance=w, clip=False, generator=torch.Generator().manual_seed(2))
        acc = nn_label_accuracy(x, y, ref_x, ref_y)
        metrics["8gaussians_guidance"][f"w={w}"] = {"class_acc": round(acc, 4)}
        ax.scatter(ref_x[:, 0], ref_x[:, 1], c="lightgray", s=1)
        ax.scatter(x[:, 0], x[:, 1], c="tab:red", s=3, alpha=0.6)
        ax.set_title(f"guidance w={w:g}\non-class {acc:.0%}", fontsize=9)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("8 Gaussians, class 0 requested (DDIM 50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "toy_guidance.png", dpi=130)
    plt.close(fig)

    (args.out / "toy_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}/toy_samplers.png, toy_guidance.png, toy_metrics.json")


if __name__ == "__main__":
    main()
