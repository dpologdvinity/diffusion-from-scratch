"""Evaluate a trained model: sample quality vs sampler steps and vs guidance weight.

    # 1. generate samples for one sampler config (run several of these in parallel)
    uv run python -m ddpm.evaluate generate --dataset mnist --method ddim --steps 50 --guidance 2
    # 2. score every generated config against the test set, write tables and plots
    uv run python -m ddpm.evaluate report --dataset mnist
    # 3. sequential wall-clock benchmark of the samplers
    uv run python -m ddpm.evaluate timing --dataset mnist
    # 4. sample grids, denoising trajectory, loss curve
    uv run python -m ddpm.evaluate figures --dataset mnist

All configs of a dataset use the same seed and labels, so the deterministic DDIM runs
start from identical noise x_T and can be compared image by image.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from . import data
from .classifier import load_or_train, predict
from .metrics import frechet_distance, kernel_distance
from .sampling import sample
from .train import load_ema_model

RESULTS = Path("results")


def run_name(method: str, steps: int, guidance: float, eta: float = 0.0) -> str:
    name = f"{method}{steps}_w{guidance:g}"
    return f"{name}_eta{eta:g}" if method == "ddim" and eta else name


def runs_dir(dataset: str) -> Path:
    return RESULTS / "runs" / dataset


def balanced_labels(n: int, num_classes: int = 10) -> torch.Tensor:
    return torch.arange(num_classes).repeat(n // num_classes + 1)[:n]


def load_runs(dataset: str) -> dict[str, dict]:
    """All generated configs, with shards (`name.partIofK.pt`) concatenated back into one run."""
    parts: dict[str, list] = {}
    for path in sorted(runs_dir(dataset).glob("*.pt")):
        parts.setdefault(path.stem.split(".part")[0], []).append(torch.load(path))
    runs = {}
    for name, rs in parts.items():
        runs[name] = {**rs[0], "x": torch.cat([r["x"] for r in rs]), "y": torch.cat([r["y"] for r in rs]),
                      "seconds": sum(r["seconds"] for r in rs)}
    return runs


def cmd_generate(args):
    model, sched, _ = load_ema_model(Path(args.ckpt_dir) / f"{args.dataset}.pt")
    steps = sched.T if args.method == "ddpm" else args.steps
    name = run_name(args.method, steps, args.guidance, args.eta)
    y_all = balanced_labels(args.n)
    # Batches are dealt round-robin to shards; seeds depend only on the batch offset, so a
    # sharded run produces exactly the same samples as an unsharded one.
    starts = [i for b, i in enumerate(range(0, args.n, args.batch)) if b % args.num_shards == args.shard]
    xs, ys, secs = [], [], 0.0
    for i in starts:
        y = y_all[i : i + args.batch]
        g = torch.Generator().manual_seed(args.seed + i)
        t0 = time.perf_counter()
        xs.append(sample(model, sched, (len(y), 1, 28, 28), y, method=args.method, steps=steps, eta=args.eta,
                         guidance=args.guidance, generator=g))
        ys.append(y)
        secs += time.perf_counter() - t0
        print(f"  {args.dataset} {name}: batch at {i} done ({secs:.0f}s)", flush=True)
    suffix = f".part{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    path = runs_dir(args.dataset) / f"{name}{suffix}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"x": torch.cat(xs).half(), "y": torch.cat(ys), "method": args.method, "steps": steps, "guidance": args.guidance,
         "eta": args.eta, "seconds": secs},
        path,
    )
    print(f"saved {path}")


def cmd_report(args):
    clf = load_or_train(args.dataset)
    x_test, y_test = data.load(args.dataset, train=False)
    f_test, p_test = predict(clf, x_test)
    clf_acc = (p_test == y_test).float().mean().item()
    # KID's cubic kernel assumes O(1) dot products; rescale features by the real-data std.
    feat_scale = f_test.std()
    kid_ref = f_test[torch.randperm(len(f_test), generator=torch.Generator().manual_seed(0))[:2000]] / feat_scale

    def kid(f):
        return 1000 * kernel_distance(f / feat_scale, kid_ref)

    rows = []
    for name, r in load_runs(args.dataset).items():
        x = r["x"].float()
        f, p = predict(clf, x)
        rows.append({
            "run": name, "method": r["method"], "steps": r["steps"], "guidance": r["guidance"], "eta": r.get("eta", 0.0), "n": len(x),
            "class_acc": (p == r["y"]).float().mean().item(),
            "fid": frechet_distance(f, f_test),
            "kid_x1000": kid(f),
            "gen_seconds": r["seconds"],
        })

    # Reference floor: real training images scored the same way, with the same sample size.
    n = rows[0]["n"] if rows else 500
    x_train, y_train = data.load(args.dataset, train=True)
    idx = torch.randperm(len(x_train), generator=torch.Generator().manual_seed(0))[:n]
    f_real, p_real = predict(clf, x_train[idx])
    floor = {
        "run": "real_train_images", "n": n, "class_acc": (p_real == y_train[idx]).float().mean().item(),
        "fid": frechet_distance(f_real, f_test),
        "kid_x1000": kid(f_real),
    }
    report = {"dataset": args.dataset, "classifier_test_acc": clf_acc, "real_floor": floor, "runs": rows}
    (RESULTS / f"{args.dataset}_metrics.json").write_text(json.dumps(report, indent=2))

    def fmt(r):
        return f"| {r['run']} | {r.get('steps', '')} | {r.get('guidance', '')} | {r['class_acc']:.1%} | {r['fid']:.2f} | {r['kid_x1000']:.2f} |"

    lines = [f"Classifier test accuracy: {clf_acc:.2%}. N = {n} samples per config.", "",
             "| run | steps | guidance | class acc | FID (clf feats) | KID x1000 |", "|---|---|---|---|---|---|"]
    lines += [fmt(r) for r in sorted(rows, key=lambda r: (r["guidance"], -r["steps"]))]
    lines.append(fmt(floor))
    (RESULTS / f"{args.dataset}_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_tradeoffs(args.dataset, rows, floor, args.sweep_guidance)


def plot_tradeoffs(dataset: str, rows: list, floor: dict, w_sweep: float):
    step_rows = sorted([r for r in rows if r["guidance"] == w_sweep and not r.get("eta")], key=lambda r: r["steps"])
    if step_rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for ax, key, label in [(axes[0], "fid", "FID (classifier features) ↓"), (axes[1], "class_acc", "class accuracy ↑")]:
            ddim = [r for r in step_rows if r["method"] == "ddim"]
            ax.plot([r["steps"] for r in ddim], [r[key] for r in ddim], "o-", label="DDIM (η=0)")
            for r in step_rows:
                if r["method"] == "ddpm":
                    ax.axhline(r[key], color="tab:red", ls="--", label=f"DDPM ({r['steps']} steps)")
            ax.axhline(floor[key], color="gray", ls=":", label="real images")
            ax.set_xscale("log")
            ax.set_xlabel("sampling steps")
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
        fig.suptitle(f"{dataset}: quality vs sampling steps (guidance w={w_sweep:g})", fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{dataset}_steps_tradeoff.png", dpi=130)
        plt.close(fig)

    g_rows = sorted([r for r in rows if r["method"] == "ddim" and r["steps"] == 50 and not r.get("eta")], key=lambda r: r["guidance"])
    if len(g_rows) > 1:
        fig, ax1 = plt.subplots(figsize=(5, 3.4))
        ws = [r["guidance"] for r in g_rows]
        ax1.plot(ws, [r["fid"] for r in g_rows], "o-", color="tab:blue")
        ax1.set_xlabel("guidance weight w (0 = unconditional, 1 = conditional)")
        ax1.set_ylabel("FID ↓", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(ws, [r["class_acc"] for r in g_rows], "s-", color="tab:orange")
        ax2.set_ylabel("class accuracy ↑", color="tab:orange")
        ax1.set_title(f"{dataset}: classifier-free guidance (DDIM 50)", fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{dataset}_guidance_tradeoff.png", dpi=130)
        plt.close(fig)


def cmd_timing(args):
    """Wall-clock time per image, measured sequentially on one batch.

    DDIM 50 is timed before and after DDPM to bracket any change in background load.
    """
    model, sched, _ = load_ema_model(Path(args.ckpt_dir) / f"{args.dataset}.pt")
    y = balanced_labels(args.batch)
    configs = [("ddim", 50), ("ddpm", sched.T), ("ddim", 50), ("ddim", 20), ("ddim", 10)]
    results = []
    for method, steps in configs:
        g = torch.Generator().manual_seed(0)
        t0 = time.perf_counter()
        sample(model, sched, (len(y), 1, 28, 28), y, method=method, steps=steps, guidance=args.guidance, generator=g)
        secs = time.perf_counter() - t0
        results.append({"method": method, "steps": steps, "seconds": secs, "ms_per_image": 1000 * secs / len(y)})
        print(f"  {method} {steps}: {secs:.1f}s for {len(y)} images", flush=True)
    ddpm = next(r for r in results if r["method"] == "ddpm")["seconds"]
    for r in results:
        r["speedup_vs_ddpm"] = ddpm / r["seconds"]
    out = {"dataset": args.dataset, "batch": args.batch, "guidance": args.guidance, "threads": torch.get_num_threads(), "results": results}
    (RESULTS / f"{args.dataset}_timing.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_figures(args):
    names = data.CLASS_NAMES[args.dataset]
    runs = load_runs(args.dataset)

    def load_run(name):
        return runs.get(name)

    def first_per_class(r, k):
        """k samples per class, as a (10*k) batch ordered class by class."""
        idx = torch.cat([torch.nonzero(r["y"] == c).flatten()[:k] for c in range(10)])
        return r["x"][idx].float()

    w = args.sweep_guidance
    main_run = load_run(run_name("ddim", 50, w))
    if main_run is not None:
        # Rows are classes: transpose a (class, k) ordering into a grid with nrow=k.
        save_image(data.to_unit(first_per_class(main_run, 10)), RESULTS / f"{args.dataset}_samples.png", nrow=10, padding=1)

    # Same noise, same label, different step counts (DDIM is deterministic, so rows are comparable).
    sampler_rows = [(f"DDPM {1000}", run_name("ddpm", 1000, w))] + [(f"DDIM {s}", run_name("ddim", s, w)) for s in (250, 100, 50, 20, 10)]
    labeled_grid([(lbl, load_run(n)) for lbl, n in sampler_rows], first_per_class, names,
                 RESULTS / f"{args.dataset}_sampler_comparison.png", f"{args.dataset}: one sample per class, guidance w={w:g}")

    guidance_rows = [(f"w = {g:g}", run_name("ddim", 50, g)) for g in (0, 1, 2, 3, 5)]
    labeled_grid([(lbl, load_run(n)) for lbl, n in guidance_rows], first_per_class, names,
                 RESULTS / f"{args.dataset}_guidance_comparison.png", f"{args.dataset}: classifier-free guidance weight (DDIM 50, same noise)")

    # Denoising trajectory: x_t along a 50-step DDIM run.
    model, sched, ckpt = load_ema_model(Path(args.ckpt_dir) / f"{args.dataset}.pt")
    y = torch.arange(10)
    _, traj = sample(model, sched, (10, 1, 28, 28), y, method="ddim", steps=50, guidance=w,
                     generator=torch.Generator().manual_seed(3), return_trajectory=True)
    picks = [0, 10, 20, 30, 40, 45, 48, 50]
    frames = torch.stack([traj[i] for i in picks], dim=1).flatten(0, 1)  # (10 * len(picks), 1, 28, 28)
    save_image(data.to_unit(frames), RESULTS / f"{args.dataset}_denoising.png", nrow=len(picks), padding=1)

    hist = ckpt["history"]
    if hist:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot([h["step"] for h in hist], [h["loss"] for h in hist])
        ax.set_xlabel("training step")
        ax.set_ylabel("noise-prediction MSE")
        ax.set_title(f"{args.dataset}: training loss", fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULTS / f"{args.dataset}_loss.png", dpi=130)
        plt.close(fig)
    print(f"figures written to {RESULTS}/")


def labeled_grid(rows, pick, class_names, path: Path, title: str):
    rows = [(label, r) for label, r in rows if r is not None]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(8, 0.95 * len(rows) + 0.6))
    axes = [axes] if len(rows) == 1 else axes
    for ax, (label, r) in zip(axes, rows):
        grid = make_grid(data.to_unit(pick(r, 1)), nrow=10, padding=1, pad_value=1.0)
        ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].set_xticks([1 + 29 * i + 14 for i in range(10)])
    axes[-1].set_xticklabels(class_names, fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "report", "timing", "figures"):
        s = sub.add_parser(name)
        s.add_argument("--dataset", choices=list(data.DATASETS), default="mnist")
        s.add_argument("--ckpt-dir", default="checkpoints")
        s.add_argument("--threads", type=int, default=1)
        s.add_argument("--sweep-guidance", type=float, default=2.0, help="guidance weight used for the steps sweep")
        if name in ("generate", "timing"):
            s.add_argument("--guidance", type=float, default=2.0)
        if name == "generate":
            s.add_argument("--method", choices=["ddpm", "ddim"], default="ddim")
            s.add_argument("--steps", type=int, default=50)
            s.add_argument("--n", type=int, default=500)
            s.add_argument("--batch", type=int, default=250)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity (0 = deterministic, 1 = DDPM-like)")
            s.add_argument("--shard", type=int, default=0, help="which shard of the batches to generate")
            s.add_argument("--num-shards", type=int, default=1, help="split generation across this many processes")
        if name == "timing":
            s.add_argument("--batch", type=int, default=20)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    {"generate": cmd_generate, "report": cmd_report, "timing": cmd_timing, "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    main()
