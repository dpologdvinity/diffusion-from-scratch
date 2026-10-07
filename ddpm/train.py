"""Train a class-conditional DDPM with classifier-free guidance on MNIST / FashionMNIST.

    uv run python -m ddpm.train --dataset mnist --steps 20000

Training objective (Ho et al. 2020, "simple" loss):

    t ~ Uniform{0..T-1},  eps ~ N(0, I),  x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps
    L = || eps - eps_theta(x_t, t, y) ||^2

For classifier-free guidance, the label is replaced by the null label with probability
p_uncond, so one network learns both the conditional and unconditional score.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from . import data
from .sampling import sample
from .schedule import NoiseSchedule
from .unet import UNet


def save_checkpoint(path: Path, model, ema, opt, step: int, args, history: list):
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "unet_config": model.config,
            "schedule": {"T": args.T, "kind": args.schedule},
            "args": vars(args),
            "history": history,
        },
        tmp,
    )
    tmp.replace(path)


def load_ema_model(path: str | Path) -> tuple[UNet, NoiseSchedule, dict]:
    """Load the EMA weights of a trained checkpoint for sampling."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = UNet(**ckpt["unet_config"])
    model.load_state_dict(ckpt["ema"])
    model.eval()
    return model, NoiseSchedule(**ckpt["schedule"]), ckpt


@torch.no_grad()
def update_ema(ema, model, decay: float):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


@torch.no_grad()
def save_preview(ema, sched, path: Path, num_classes: int, guidance: float):
    """A num_classes x 8 grid: each row is one class, sampled with 50 DDIM steps."""
    ema.eval()
    y = torch.arange(num_classes).repeat_interleave(8)
    g = torch.Generator().manual_seed(0)
    x = sample(ema, sched, (len(y), 1, 28, 28), y, method="ddim", steps=50, guidance=guidance, generator=g)
    save_image(data.to_unit(x), path, nrow=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(data.DATASETS), default="mnist")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--minutes", type=float, default=0, help="stop after this much wall-clock time (0 = no limit)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--p-uncond", type=float, default=0.1, help="label dropout probability for CFG")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--ch-mults", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--num-res-blocks", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=1000)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    x_all, y_all = data.load(args.dataset, train=True)
    num_classes = int(y_all.max()) + 1
    sched = NoiseSchedule(args.T, args.schedule)

    model = UNet(
        base_ch=args.base_ch, ch_mults=tuple(args.ch_mults), num_res_blocks=args.num_res_blocks,
        num_classes=num_classes, dropout=args.dropout,
    )
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out / f"{args.dataset}.pt"
    log_path = args.out / f"{args.dataset}_log.jsonl"
    step, history = 0, []
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        step, history = ckpt["step"], ckpt["history"]
        print(f"resumed from step {step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.dataset}: {len(x_all)} images, {num_classes} classes, UNet {n_params / 1e6:.2f}M params, threads={torch.get_num_threads()}")

    model.train()
    t0, running = time.time(), 0.0
    deadline = time.time() + args.minutes * 60 if args.minutes else float("inf")
    while step < args.steps and time.time() < deadline:
        idx = torch.randint(0, len(x_all), (args.batch_size,))
        x0, y = x_all[idx], y_all[idx]
        y = torch.where(torch.rand(y.shape) < args.p_uncond, torch.full_like(y, num_classes), y)

        t = torch.randint(0, sched.T, (args.batch_size,))
        noise = torch.randn_like(x0)
        loss = F.mse_loss(model(sched.q_sample(x0, t, noise), t, y), noise)

        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # Ramp EMA decay up early so the average isn't dominated by random init.
        update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
        step += 1
        running += loss.item()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            rec = {"step": step, "loss": running / args.log_every, "lr": lr, "sec": round(elapsed, 1)}
            history.append(rec)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:6d}  loss {rec['loss']:.4f}  {elapsed / 60:6.1f} min", flush=True)
            running = 0.0
        if step % args.ckpt_every == 0:
            save_checkpoint(ckpt_path, model, ema, opt, step, args, history)
        if args.preview_every and step % args.preview_every == 0:
            save_preview(ema, sched, args.out / f"{args.dataset}_preview_{step:06d}.png", num_classes, guidance=3.0)

    save_checkpoint(ckpt_path, model, ema, opt, step, args, history)
    save_preview(ema, sched, args.out / f"{args.dataset}_preview_{step:06d}.png", num_classes, guidance=3.0)
    print(f"done: {step} steps in {(time.time() - t0) / 60:.1f} min -> {ckpt_path}")


if __name__ == "__main__":
    main()
