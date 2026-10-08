"""Local web server: the demo page plus live sampling from the trained model.

    uv run python -m ddpm.serve        # then open http://127.0.0.1:8000

Sampling runs on one CPU thread, one request at a time, so the server stays light on a shared machine.
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .render import png_data_url, sample_frames
from .train import load_ema_model

_lock = threading.Lock()


DATASETS = ("mnist", "fashion")


class SampleRequest(BaseModel):
    dataset: Literal["mnist", "fashion"] = "mnist"
    label: int = Field(7, ge=0, le=9, description="class index")
    guidance: float = Field(2.0, ge=0, le=10)
    sampler: Literal["ddim", "ddpm"] = "ddim"
    steps: int = Field(50, ge=1, le=1000, description="DDIM steps; ignored for DDPM, which always uses all of them")
    eta: float = Field(0.0, ge=0, le=1)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    frames: int = Field(11, ge=2, le=50)


def create_app(models: dict[str, tuple], web_dir: Path, assets_dir: Path | None) -> FastAPI:
    """models maps dataset name -> (model, schedule, info) for every dataset with weights."""
    app = FastAPI(title="Diffusion from scratch", docs_url="/api/docs", openapi_url="/api/openapi.json")

    @app.get("/api/health")
    def health():
        return {"status": "ok", "datasets": {name: info for name, (_, _, info) in models.items()}}

    @app.post("/api/sample")
    def sample(req: SampleRequest):
        if req.dataset not in models:
            raise HTTPException(422, detail=[{"loc": ["body", "dataset"], "msg": f"no weights loaded for {req.dataset}",
                                              "type": "value_error"}])
        model, sched, _ = models[req.dataset]
        steps = min(req.steps, sched.T)
        with _lock:
            t0 = time.perf_counter()
            final, frames = sample_frames(model, sched, torch.tensor([req.label]), guidance=req.guidance,
                                          sampler=req.sampler, steps=steps, eta=req.eta, seed=req.seed,
                                          frames=req.frames)
            secs = time.perf_counter() - t0
        return {"image": png_data_url(final), "frames": [png_data_url(f) for f in frames], "seconds": round(secs, 3)}

    # Precomputed assets from `make web`, if present, so the static grid also works locally.
    if assets_dir is not None and (assets_dir / "manifest.json").exists():
        app.get("/manifest.json", include_in_schema=False)(lambda: FileResponse(assets_dir / "manifest.json"))
        if (assets_dir / "sprites").is_dir():
            app.mount("/sprites", StaticFiles(directory=assets_dir / "sprites"), name="sprites")
    for name in DATASETS:
        if assets_dir is not None and (assets_dir / name / "manifest.json").exists():
            app.mount(f"/{name}", StaticFiles(directory=assets_dir / name), name=f"assets-{name}")

    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")  # last, so API routes win
    return app


def default_checkpoint(root: Path, dataset: str = "mnist") -> Path:
    """A locally trained checkpoint if there is one, otherwise the weights bundled with the repo."""
    trained = root / "checkpoints" / f"{dataset}.pt"
    return trained if trained.exists() else root / "models" / f"{dataset}.pt"


def main(argv: list[str] | None = None):
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="MNIST checkpoint (default: checkpoints/mnist.pt if trained, else "
                                                "models/mnist.pt); other datasets use their default checkpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    paths = {name: default_checkpoint(root, name) for name in DATASETS}
    if args.ckpt:
        paths["mnist"] = Path(args.ckpt)
        if not paths["mnist"].exists():
            raise SystemExit(f"checkpoint not found: {args.ckpt}. Train one with `make train` first.")
    paths = {name: path for name, path in paths.items() if path.exists()}
    if not paths:
        raise SystemExit("no checkpoint found in checkpoints/ or models/. Train one with `make train` first.")
    torch.set_num_threads(1)
    models = {}
    for name, path in paths.items():
        model, sched, ckpt = load_ema_model(path)
        models[name] = (model, sched, {"train_steps": ckpt["step"], "params": sum(p.numel() for p in model.parameters())})
        print(f"loaded {name} from {path}")
    app = create_app(models, root / "web", root / "docs")
    print(f"serving on http://{args.host}:{args.port}  (API docs at /api/docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
