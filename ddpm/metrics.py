"""Distribution metrics computed on classifier features."""

import numpy as np
import torch
from scipy import linalg


def frechet_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """FID formula: ||mu_a - mu_b||^2 + Tr(S_a + S_b - 2 (S_a S_b)^{1/2}).

    Biased upward for small sample sizes, so only compare scores computed with the same N.
    """
    a, b = a.double().numpy(), b.double().numpy()
    mu_a, mu_b = a.mean(0), b.mean(0)
    s_a, s_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = np.real(linalg.sqrtm(s_a @ s_b))
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(s_a + s_b - 2 * covmean))


def kernel_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """KID (Binkowski et al. 2018): unbiased MMD^2 with the kernel k(x, y) = (x.y / d + 1)^3.

    Unlike FID it is unbiased, so it stays meaningful with a few hundred samples.
    """
    a, b = a.double(), b.double()
    d = a.shape[1]
    k_aa = (a @ a.T / d + 1) ** 3
    k_bb = (b @ b.T / d + 1) ** 3
    k_ab = (a @ b.T / d + 1) ** 3
    m, n = len(a), len(b)
    sum_aa = (k_aa.sum() - k_aa.diagonal().sum()) / (m * (m - 1))
    sum_bb = (k_bb.sum() - k_bb.diagonal().sum()) / (n * (n - 1))
    return float(sum_aa + sum_bb - 2 * k_ab.mean())
