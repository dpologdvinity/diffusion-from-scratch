"""Reverse-process samplers: DDPM ancestral sampling, DDIM, and classifier-free guidance.

All samplers use an epsilon-prediction model: model(x_t, t, y) -> predicted noise.

Both update rules are written through the predicted clean image

    x0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)

which lets us optionally clip x0_hat to the data range [-1, 1] (standard practice for
images; it removes rare out-of-range predictions at high noise levels).
"""

import torch

from .schedule import NoiseSchedule, extract


def guided_eps(model, x, t, y, guidance: float, null_label: int) -> torch.Tensor:
    """Classifier-free guidance (Ho & Salimans 2022).

        eps = eps_uncond + guidance * (eps_cond - eps_uncond)

    guidance = 0 -> unconditional model, 1 -> plain conditional model,
    > 1 -> extrapolate away from the unconditional prediction (sharper, more on-class
    samples at the cost of diversity). Conditional and unconditional passes are batched
    into a single forward call.
    """
    if y is None:
        return model(x, t, torch.full((x.shape[0],), null_label, device=x.device))
    if guidance == 1.0:
        return model(x, t, y)
    null = torch.full_like(y, null_label)
    eps_c, eps_u = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, null])).chunk(2)
    return eps_u + guidance * (eps_c - eps_u)


def predict_x0(sched: NoiseSchedule, x_t, t, eps, clip: bool) -> torch.Tensor:
    ab = extract(sched.alpha_bar, t, x_t.dim())
    x0 = (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()
    return x0.clamp(-1, 1) if clip else x0


def ddpm_step(sched: NoiseSchedule, x_t, t, eps, noise, clip: bool = True) -> torch.Tensor:
    """One ancestral step x_t -> x_{t-1}, sampling from the true posterior q(x_{t-1} | x_t, x0_hat).

        mean = sqrt(ab_{t-1}) beta_t / (1 - ab_t) * x0_hat
             + sqrt(alpha_t) (1 - ab_{t-1}) / (1 - ab_t) * x_t
        var  = beta_tilde_t = (1 - ab_{t-1}) / (1 - ab_t) * beta_t
    """
    nd = x_t.dim()
    x0 = predict_x0(sched, x_t, t, eps, clip)
    ab_t = extract(sched.alpha_bar, t, nd)
    ab_prev = sched.alpha_bar_at(t - 1).view(-1, *([1] * (nd - 1)))
    beta_t = extract(sched.betas, t, nd)
    alpha_t = extract(sched.alphas, t, nd)

    mean = (ab_prev.sqrt() * beta_t / (1 - ab_t)) * x0 + (alpha_t.sqrt() * (1 - ab_prev) / (1 - ab_t)) * x_t
    var = (1 - ab_prev) / (1 - ab_t) * beta_t
    return mean + var.sqrt() * noise


def ddim_step(sched: NoiseSchedule, x_t, t, t_prev, eps, noise, eta: float = 0.0, clip: bool = True) -> torch.Tensor:
    """One DDIM step x_t -> x_{t_prev} for any t_prev < t (Song et al. 2021, eq. 12).

        x_prev = sqrt(ab_prev) * x0_hat                      # "predicted x0"
               + sqrt(1 - ab_prev - sigma^2) * eps_hat       # "direction pointing to x_t"
               + sigma * z                                   # fresh noise
        sigma  = eta * sqrt((1 - ab_prev) / (1 - ab_t)) * sqrt(1 - ab_t / ab_prev)

    eta = 0 is deterministic; eta = 1 with t_prev = t - 1 is exactly the DDPM step.
    Because the update only needs alpha_bar at the two endpoints, we can skip timesteps.
    """
    nd = x_t.dim()
    x0 = predict_x0(sched, x_t, t, eps, clip)
    ab_t = extract(sched.alpha_bar, t, nd)
    ab_prev = sched.alpha_bar_at(t_prev).view(-1, *([1] * (nd - 1)))
    # Re-derive eps from the (possibly clipped) x0 so both terms stay consistent with x_t.
    eps = (x_t - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()

    sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
    return ab_prev.sqrt() * x0 + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps + sigma * noise


def ddim_timesteps(T: int, steps: int) -> list[int]:
    """Evenly spaced subsequence of [0, T-1], descending, always including T-1 (and 0 when steps > 1)."""
    if steps == 1:
        return [T - 1]  # a single step must start from pure noise, not t = 0
    ts = torch.linspace(0, T - 1, steps).round().long().unique().tolist()
    return ts[::-1]


@torch.no_grad()
def sample(
    model,
    sched: NoiseSchedule,
    shape: tuple,
    y: torch.Tensor | None = None,
    method: str = "ddim",
    steps: int = 50,
    eta: float = 0.0,
    guidance: float = 1.0,
    clip: bool = True,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    device: torch.device | str | None = None,
):
    """Generate samples starting from pure noise x_T ~ N(0, I).

    method="ddpm" runs all T ancestral steps (steps/eta are ignored).
    method="ddim" runs `steps` steps on an evenly spaced timestep subsequence.

    Noise is always drawn from the (CPU) generator and then moved to `device` (default: the
    model's device), so a GPU run starts from exactly the same noise as a CPU run.
    """
    if device is None:
        device = next((p.device for p in model.parameters()), torch.device("cpu"))
    null_label = getattr(model, "num_classes", 0)
    x = torch.randn(shape, generator=generator).to(device)
    traj = [x]
    if y is not None:
        y = y.to(device)

    def randn_like(x):
        return torch.randn(x.shape, generator=generator).to(device)

    def batch_t(v):
        return torch.full((shape[0],), v, dtype=torch.long, device=device)

    if method == "ddpm":
        for t in range(sched.T - 1, -1, -1):
            tt = batch_t(t)
            eps = guided_eps(model, x, tt, y, guidance, null_label)
            x = ddpm_step(sched, x, tt, eps, randn_like(x), clip)
            if return_trajectory:
                traj.append(x)
    elif method == "ddim":
        ts = ddim_timesteps(sched.T, steps)
        for i, t in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            tt = batch_t(t)
            eps = guided_eps(model, x, tt, y, guidance, null_label)
            noise = randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = ddim_step(sched, x, tt, batch_t(t_prev), eps, noise, eta, clip)
            if return_trajectory:
                traj.append(x)
    else:
        raise ValueError(f"unknown method {method!r}")

    return (x, traj) if return_trajectory else x
