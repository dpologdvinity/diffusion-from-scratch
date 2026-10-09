"""Turn sampler output into animation frames and PNG images for the web frontend."""

import base64
import io

import torch
from PIL import Image
from torchvision.utils import make_grid

from . import data
from .sampling import sample


@torch.no_grad()
def sample_frames(model, sched, digits: torch.Tensor, guidance: float, sampler: str, steps: int, eta: float,
                  seed: int, frames: int, cancel=None) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample one image per requested digit and return (final images, evenly spaced trajectory states).

    The frames always include the starting noise x_T and the final sample, and never exceed the
    trajectory length. Equal seeds give equal starting noise, so settings can be compared directly.
    """
    g = torch.Generator().manual_seed(seed)
    final, traj = sample(model, sched, (len(digits), 1, 28, 28), digits, method=sampler, steps=steps, eta=eta,
                         guidance=guidance, generator=g, return_trajectory=True, cancel=cancel)
    idx = torch.linspace(0, len(traj) - 1, min(frames, len(traj))).round().long().unique()
    return final, [traj[i] for i in idx]


def png_row(x: torch.Tensor) -> bytes:
    """The images side by side as one grayscale PNG (N*28 x 28)."""
    grid = make_grid(data.to_unit(x), nrow=len(x), padding=0)[0]
    buf = io.BytesIO()
    Image.fromarray((grid * 255).round().byte().numpy(), mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def png_data_url(x: torch.Tensor) -> str:
    return "data:image/png;base64," + base64.b64encode(png_row(x)).decode()
