"""DDPM noise schedule and the forward (noising) process.

Notation follows Ho et al. 2020, "Denoising Diffusion Probabilistic Models":

    beta_t        variance of the Gaussian noise added at step t
    alpha_t     = 1 - beta_t
    alpha_bar_t = prod_{s <= t} alpha_s

Because a sum of Gaussians is Gaussian, x_t can be sampled from x_0 in one shot:

    q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I )

Timesteps are 0-indexed: t in {0, ..., T-1}. alpha_bar at index -1 is defined as 1
(no noise), which the samplers use for the final step.
"""

import math

import torch


def linear_betas(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from the original DDPM paper."""
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021): alpha_bar follows a squared cosine."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)


def extract(v: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Index a per-timestep vector by a batch of timesteps, reshaped to broadcast over x."""
    return v.to(t.device)[t].view(-1, *([1] * (ndim - 1)))


class NoiseSchedule:
    def __init__(self, T: int = 1000, kind: str = "linear", dtype: torch.dtype = torch.float32):
        if kind == "linear":
            betas = linear_betas(T)
        elif kind == "cosine":
            betas = cosine_betas(T)
        else:
            raise ValueError(f"unknown schedule {kind!r}")
        self.T = T
        self.kind = kind
        # Computed in float64 (cumprod over 1000 terms), stored in `dtype`.
        self.betas = betas.to(dtype)
        self.alphas = (1 - betas).to(dtype)
        self.alpha_bar = torch.cumprod(1 - betas, dim=0).to(dtype)

    def alpha_bar_at(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar_t with the convention alpha_bar_{-1} = 1."""
        ab = self.alpha_bar.to(t.device)[t.clamp(min=0)]
        return torch.where(t >= 0, ab, torch.ones_like(ab))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) using the closed form."""
        ab = extract(self.alpha_bar, t, x0.dim())
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
