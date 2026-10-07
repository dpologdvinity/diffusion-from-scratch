"""Datasets held fully in memory as tensors (MNIST-sized data is ~50 MB as uint8).

Random-index batching from a resident tensor is much faster on CPU than a DataLoader.
"""

import torch
from torchvision import datasets

DATASETS = {"mnist": datasets.MNIST, "fashion": datasets.FashionMNIST}

FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
CLASS_NAMES = {"mnist": [str(i) for i in range(10)], "fashion": FASHION_CLASSES}


def load(name: str, train: bool = True, root: str = "data") -> tuple[torch.Tensor, torch.Tensor]:
    """Return images scaled to [-1, 1] with shape (N, 1, 28, 28), and integer labels (N,)."""
    ds = DATASETS[name](root, train=train, download=True)
    x = ds.data.float().div(127.5).sub(1).unsqueeze(1)
    return x, ds.targets.long()


def to_unit(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for saving/plotting."""
    return (x.clamp(-1, 1) + 1) / 2
