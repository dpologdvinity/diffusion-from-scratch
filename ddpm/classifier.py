"""Small CNN classifier used only for evaluation.

Two uses:
  * class accuracy of generated samples ("did the model draw the digit we asked for?")
  * a feature space for FID / KID. Standard FID uses an ImageNet Inception network, which
    is a poor fit for 28x28 grayscale images; a classifier trained on the dataset itself is
    the usual substitute. Scores are therefore not comparable to published FID numbers,
    only to each other.

    uv run python -m ddpm.classifier --dataset mnist
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data

FEATURE_DIM = 64


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 7 * 7, FEATURE_DIM)
        self.head = nn.Linear(FEATURE_DIM, num_classes)
        self.dropout = nn.Dropout(0.3)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.fc(self.conv(x).flatten(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.features(x)))


@torch.no_grad()
def predict(model: SmallCNN, x: torch.Tensor, batch: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (features, predicted labels) for images in [-1, 1]."""
    model.eval()
    feats, preds = [], []
    for i in range(0, len(x), batch):
        f = model.features(x[i : i + batch])
        feats.append(f)
        preds.append(model.head(f).argmax(1))
    return torch.cat(feats), torch.cat(preds)


def train(dataset: str, epochs: int = 4, batch: int = 128, seed: int = 0) -> tuple[SmallCNN, float]:
    torch.manual_seed(seed)
    x, y = data.load(dataset, train=True)
    xt, yt = data.load(dataset, train=False)
    model = SmallCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        for i in range(0, len(x), batch):
            idx = perm[i : i + batch]
            # Small random shifts make the features less brittle to generator artifacts.
            xb = torch.roll(x[idx], shifts=tuple(torch.randint(-2, 3, (2,)).tolist()), dims=(2, 3))
            loss = F.cross_entropy(model(xb), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        acc = (predict(model, xt)[1] == yt).float().mean().item()
        print(f"  {dataset} classifier epoch {epoch + 1}: test acc {acc:.4f}", flush=True)
    return model, acc


def load_or_train(dataset: str, ckpt_dir: Path = Path("checkpoints")) -> SmallCNN:
    path = ckpt_dir / f"classifier_{dataset}.pt"
    model = SmallCNN()
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu")["model"])
    else:
        t0 = time.perf_counter()
        model, acc = train(dataset)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "test_acc": acc}, path)
        print(f"  trained classifier in {time.perf_counter() - t0:.0f}s -> {path}")
    model.eval()
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(data.DATASETS), default="mnist")
    load_or_train(p.parse_args().dataset)
