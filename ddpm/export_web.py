"""Precompute samples for the static web page (GitHub Pages serves the output from docs/).

    uv run python -m ddpm.export_web            # reads checkpoints/mnist.pt, writes docs/

Every setting uses the same seed, so all digits start from the same noise x_T and differences
between settings come from the setting, not from randomness.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from . import data
from .render import png_row, sample_frames
from .train import load_ema_model

DIGITS = torch.arange(10)


def export(model, sched, out_dir: Path, web_dir: Path | None, guidances=(0, 1, 2, 3, 5), steps=(10, 20, 50, 100),
           traj_frames: int = 11, traj_guidance: float = 2.0, traj_steps: int = 50, ddpm_guidance: float = 2.0,
           seed: int = 0, model_info: dict | None = None, labels: list[str] | None = None) -> dict:
    """Write sprite rows (one image per class, 280x28 each) and the manifest into out_dir.

    web_dir, if given, is copied in too (the page lives at the site root, next to the MNIST assets).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if web_dir is not None:
        for f in web_dir.iterdir():
            if f.is_file():
                shutil.copy(f, out_dir / f.name)
    sprite_dir = out_dir / "sprites"
    sprite_dir.mkdir(exist_ok=True)

    def write(name: str, x: torch.Tensor) -> str:
        (sprite_dir / name).write_bytes(png_row(x))
        return f"sprites/{name}"

    grid = {}
    for g in guidances:
        for s in steps:
            final, _ = sample_frames(model, sched, DIGITS, guidance=g, sampler="ddim", steps=s, eta=0.0, seed=seed, frames=2)
            grid[f"w{g:g}_s{s}"] = write(f"ddim_w{g:g}_s{s}.png", final)
            print(f"  grid w={g:g} steps={s}", flush=True)

    final, _ = sample_frames(model, sched, DIGITS, guidance=ddpm_guidance, sampler="ddpm", steps=sched.T, eta=1.0,
                             seed=seed, frames=2)
    ddpm = {"guidance": ddpm_guidance, "steps": sched.T, "sprite": write(f"ddpm_w{ddpm_guidance:g}.png", final)}
    print(f"  ddpm {sched.T} steps", flush=True)

    _, frames = sample_frames(model, sched, DIGITS, guidance=traj_guidance, sampler="ddim", steps=traj_steps, eta=0.0,
                              seed=seed, frames=traj_frames)
    trajectory = {"guidance": traj_guidance, "steps": traj_steps,
                  "frames": [write(f"traj_{i:02d}.png", f) for i, f in enumerate(frames)]}

    manifest = {
        "model": model_info or {},
        "labels": labels or [str(d) for d in DIGITS.tolist()],
        "digits": DIGITS.tolist(),
        "guidance": list(guidances),
        "steps": list(steps),
        "grid": grid,
        "ddpm": ddpm,
        "trajectory": trajectory,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(data.DATASETS), default="mnist")
    p.add_argument("--ckpt", default=None, help="default: checkpoints/<dataset>.pt")
    p.add_argument("--out", type=Path, default=None, help="default: docs/ for mnist, docs/<dataset>/ otherwise")
    p.add_argument("--web", type=Path, default=Path("web"))
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)
    args.ckpt = args.ckpt or f"checkpoints/{args.dataset}.pt"
    args.out = args.out or (Path("docs") if args.dataset == "mnist" else Path("docs") / args.dataset)
    if not Path(args.ckpt).exists():
        raise SystemExit(f"checkpoint not found: {args.ckpt}. Train one with `make train` first.")
    torch.set_num_threads(args.threads)
    model, sched, ckpt = load_ema_model(args.ckpt)
    info = {"train_steps": ckpt["step"], "params": sum(p.numel() for p in model.parameters())}
    # The page itself is copied only with the root (MNIST) assets.
    export(model, sched, args.out, args.web if args.dataset == "mnist" else None, model_info=info,
           labels=data.CLASS_NAMES[args.dataset])
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
