"""Train a class-conditional DDPM with classifier-free guidance on MNIST / FashionMNIST.

    uv run python -m ddpm.train --dataset mnist --minutes 120
    # data-parallel over N single-threaded CPU workers (batch size is per worker):
    uv run torchrun --nproc-per-node 6 -m ddpm.train --dataset mnist --minutes 120

Training objective (Ho et al. 2020, "simple" loss):

    t ~ Uniform{0..T-1},  eps ~ N(0, I),  x_t = sqrt(ab_t) x_0 + sqrt(1 - ab_t) eps
    L = || eps - eps_theta(x_t, t, y) ||^2

For classifier-free guidance, the label is replaced by the null label with probability
p_uncond, so one network learns both the conditional and unconditional score.
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
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


def export_ema(src: str | Path, dst: str | Path):
    """Write an inference-only checkpoint (EMA weights, config, schedule) small enough to commit."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: ckpt[k] for k in ("ema", "unet_config", "schedule", "step")}, dst)


def resolve_device(name: str) -> torch.device:
    """'auto' picks CUDA when available, otherwise CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def update_ema(ema, model, decay: float):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


@torch.no_grad()
def save_preview(ema, sched, path: Path, num_classes: int, guidance: float):
    """One sample per class (fixed noise) with 20 DDIM steps, kept small because it runs on CPU mid-training."""
    ema.eval()
    y = torch.arange(num_classes)
    g = torch.Generator().manual_seed(0)
    x = sample(ema, sched, (len(y), 1, 28, 28), y, method="ddim", steps=20, guidance=guidance, generator=g)
    save_image(data.to_unit(x), path, nrow=num_classes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(data.DATASETS), default="mnist")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--minutes", type=float, default=0, help="stop after this much wall-clock time (0 = no limit)")
    p.add_argument("--batch-size", type=int, default=64, help="per worker")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--p-uncond", type=float, default=0.1, help="label dropout probability for CFG")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--base-ch", type=int, default=16)
    p.add_argument("--ch-mults", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--num-res-blocks", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads per worker (0 = default)")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or auto (single-process only for GPUs)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--preview-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Under torchrun each process is one data-parallel worker; gradients are averaged with gloo.
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if world > 1:
        dist.init_process_group("gloo")
    main_proc = rank == 0
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)  # identical init on every worker (DDP also broadcasts rank 0's weights)

    device = resolve_device(args.device)
    x_all, y_all = data.load(args.dataset, train=True)
    x_all, y_all = x_all.to(device), y_all.to(device)
    num_classes = int(y_all.max()) + 1
    sched = NoiseSchedule(args.T, args.schedule)

    model = UNet(
        base_ch=args.base_ch, ch_mults=tuple(args.ch_mults), num_res_blocks=args.num_res_blocks,
        num_classes=num_classes, dropout=args.dropout,
    ).to(device)
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
        if main_proc:
            print(f"resumed from step {step}")

    net = DistributedDataParallel(model) if world > 1 else model
    torch.manual_seed(args.seed + 1000 * rank + step)  # different batches and noise per worker

    n_params = sum(p.numel() for p in model.parameters())
    if main_proc:
        print(
            f"{args.dataset}: {len(x_all)} images, {num_classes} classes, UNet {n_params / 1e6:.2f}M params, "
            f"{world} worker(s) x {torch.get_num_threads()} thread(s) on {device}, global batch {args.batch_size * world}",
            flush=True,
        )

    def should_stop() -> bool:
        # Workers must agree, or one would exit while the others wait in all-reduce.
        stop = torch.tensor([float(step >= args.steps or time.monotonic() >= deadline)])
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        return bool(stop.item())

    model.train()
    t0, running, start_step = time.monotonic(), 0.0, step
    deadline = t0 + args.minutes * 60 if args.minutes else float("inf")
    while not should_stop():
        idx = torch.randint(0, len(x_all), (args.batch_size,), device=device)
        x0, y = x_all[idx], y_all[idx]
        y = torch.where(torch.rand(y.shape, device=device) < args.p_uncond, torch.full_like(y, num_classes), y)

        t = torch.randint(0, sched.T, (args.batch_size,), device=device)
        noise = torch.randn_like(x0)
        loss = F.mse_loss(net(sched.q_sample(x0, t, noise), t, y), noise)

        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        running += loss.item()
        if not main_proc:
            continue

        # Ramp EMA decay up early so the average isn't dominated by random init.
        update_ema(ema, model, min(args.ema_decay, (1 + step) / (10 + step)))
        if step % args.log_every == 0:
            elapsed = time.monotonic() - t0
            rec = {"step": step, "loss": running / args.log_every, "lr": lr, "sec": round(elapsed, 1)}
            history.append(rec)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {step:6d}  loss {rec['loss']:.4f}  {elapsed / 60:6.1f} min  {elapsed / (step - start_step):.2f} s/step", flush=True)
            running = 0.0
        if step % args.ckpt_every == 0:
            save_checkpoint(ckpt_path, model, ema, opt, step, args, history)
        if args.preview_every and step % args.preview_every == 0:
            save_preview(ema, sched, args.out / f"{args.dataset}_preview_{step:06d}.png", num_classes, guidance=3.0)
            model.train()

    if main_proc:
        save_checkpoint(ckpt_path, model, ema, opt, step, args, history)
        save_preview(ema, sched, args.out / f"{args.dataset}_preview_{step:06d}.png", num_classes, guidance=3.0)
        print(f"done: {step} steps in {(time.monotonic() - t0) / 60:.1f} min -> {ckpt_path}", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
