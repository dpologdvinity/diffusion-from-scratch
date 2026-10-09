"""Noise schedule ablation on a GPU (run on Kaggle).

Trains an MNIST model with the given schedule (batch 52 throughout, 15,351 steps), then
evaluates it with the same sampler configs, seeds, batch size, and committed classifier as
scripts/run_eval.sh, so the numbers line up with the CPU results. By default it also
re-scores the bundled linear model (models/mnist.pt) as a CPU-vs-GPU consistency check.

    python scripts/gpu_ablation.py --repo <checkout> --out <output dir> [--schedule linear --name linear-matched --skip-baseline]
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = [
    "--method ddpm --steps 1000 --guidance 2",
    "--method ddim --steps 100 --guidance 2",
    "--method ddim --steps 50 --guidance 2",
    "--method ddim --steps 20 --guidance 2",
    "--method ddim --steps 10 --guidance 2",
    "--method ddim --steps 50 --guidance 0",
    "--method ddim --steps 50 --guidance 1",
    "--method ddim --steps 50 --guidance 3",
    "--method ddim --steps 50 --guidance 5",
    "--method ddim --steps 50 --guidance 2 --eta 0.5",
    "--method ddim --steps 50 --guidance 2 --eta 1",
    "--method ddim --steps 10 --guidance 2 --eta 1",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=15351, help="training steps (the linear model's count)")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--device", default="cuda", help="cpu only for a quick smoke test")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--name", default=None, help="results subdirectory (default: the schedule)")
    p.add_argument("--skip-baseline", action="store_true", help="do not re-score the bundled linear model")
    args = p.parse_args()
    out = args.out.resolve()

    def run(*cmd):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", *cmd], check=True, cwd=args.repo)
        print(f"  [{time.perf_counter() - t0:.0f}s] {' '.join(cmd[:3])}", flush=True)

    ckpts = out / "checkpoints"
    name = args.name or args.schedule
    run("ddpm.train", "--dataset", "mnist", "--device", args.device, "--schedule", args.schedule, "--steps", str(args.steps),
        "--batch-size", "52", "--lr", "4e-4", "--warmup", "200", "--log-every", "500", "--ckpt-every", "2000",
        "--preview-every", "0", "--out", str(ckpts))
    subprocess.run([sys.executable, "-c", "from ddpm.train import export_ema; import sys; export_ema(sys.argv[1], sys.argv[2])",
                    str(ckpts / "mnist.pt"), str(out / "models" / f"mnist_{name}.pt")], check=True, cwd=args.repo)

    targets = [(name, ckpts)] + ([] if args.skip_baseline else [("linear", args.repo / "models")])
    for label, ckpt_dir in targets:
        results = out / "results" / label
        common = ["--dataset", "mnist", "--ckpt-dir", str(ckpt_dir), "--results-dir", str(results)]
        for cfg in CONFIGS:
            # Batch 250 matches run_eval.sh, so every config starts from the same noise as the CPU eval.
            run("ddpm.evaluate", "generate", *common, "--n", str(args.n), "--batch", "250", "--device", args.device, *cfg.split())
        run("ddpm.evaluate", "report", *common, "--sweep-guidance", "2")
        run("ddpm.evaluate", "figures", *common, "--sweep-guidance", "2")
        shutil.rmtree(results / "runs")
    print(f"done: results in {out}")


if __name__ == "__main__":
    main()
