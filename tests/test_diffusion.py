import pytest
import torch

from ddpm.sampling import ddim_step, ddim_timesteps, ddpm_step, guided_eps, sample
from ddpm.schedule import NoiseSchedule
from ddpm.unet import UNet


@pytest.fixture(scope="module")
def sched():
    return NoiseSchedule(T=1000)


class OracleModel(torch.nn.Module):
    """Predicts the exact noise for a dataset consisting of the single point x0.

    Given x_t, the only x0 consistent with the data is the stored one, so
    eps = (x_t - sqrt(ab_t) x0) / sqrt(1 - ab_t). A correct sampler must return x0.
    """

    num_classes = 0

    def __init__(self, sched, x0):
        super().__init__()
        self.sched, self.x0 = sched, x0

    def forward(self, x, t, y):
        ab = self.sched.alpha_bar[t].view(-1, *([1] * (x.dim() - 1)))
        return (x - ab.sqrt() * self.x0) / (1 - ab).sqrt()


# --- schedule -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_alpha_bar_decreases_from_one_to_near_zero(kind):
    s = NoiseSchedule(1000, kind)
    ab = s.alpha_bar
    assert torch.all(ab[1:] < ab[:-1])
    assert 0.99 < ab[0] < 1.0
    assert ab[-1] < 1e-3


def test_alpha_bar_at_minus_one_is_one(sched):
    assert sched.alpha_bar_at(torch.tensor([-1, 0]))[0] == 1.0


def test_q_sample_matches_closed_form_moments(sched):
    torch.manual_seed(0)
    x0 = torch.full((20000, 1), 0.5)
    t = torch.full((20000,), 300)
    xt = sched.q_sample(x0, t, torch.randn_like(x0))
    ab = sched.alpha_bar[300]
    assert xt.mean().item() == pytest.approx(ab.sqrt().item() * 0.5, abs=0.02)
    assert xt.var().item() == pytest.approx((1 - ab).item(), abs=0.02)


# --- sampler math ---------------------------------------------------------------------


@pytest.mark.parametrize("clip", [False, True])
def test_ddim_with_eta_one_equals_ddpm_step(clip):
    """Song et al. 2021: DDIM with eta=1 over consecutive steps is the DDPM posterior.

    Runs in float64: near t=0, 1 - alpha_bar_t ~ 1e-4, so float32 rounding of the stored
    schedule is amplified to ~1e-4 differences even though the formulas are identical.
    """
    sched = NoiseSchedule(1000, dtype=torch.float64)
    torch.manual_seed(0)
    x = torch.randn(16, 1, 4, 4, dtype=torch.float64)
    eps = torch.randn_like(x)
    z = torch.randn_like(x)
    for t in [999, 500, 10, 1, 0]:
        tt = torch.full((16,), t)
        a = ddpm_step(sched, x, tt, eps, z, clip)
        b = ddim_step(sched, x, tt, tt - 1, eps, z, eta=1.0, clip=clip)
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("method,steps", [("ddpm", None), ("ddim", 10), ("ddim", 50)])
def test_samplers_recover_x0_with_oracle_model(sched, method, steps):
    x0 = torch.tensor([[0.3, -0.7]])
    model = OracleModel(sched, x0)
    out = sample(model, sched, (4, 2), method=method, steps=steps or 0, clip=False, generator=torch.Generator().manual_seed(0))
    torch.testing.assert_close(out, x0.expand(4, 2), rtol=0, atol=1e-3)


def test_ddim_timesteps_cover_range():
    ts = ddim_timesteps(1000, 50)
    assert ts[0] == 999 and ts[-1] == 0 and len(ts) == 50
    assert all(a > b for a, b in zip(ts, ts[1:]))


def test_ddim_eta_zero_is_deterministic(sched):
    model = UNet(base_ch=16, ch_mults=(1, 2), num_res_blocks=1, attn_levels=(), num_classes=3)
    model.eval()
    y = torch.tensor([0, 1, 2])
    a = sample(model, sched, (3, 1, 28, 28), y, steps=5, generator=torch.Generator().manual_seed(1))
    b = sample(model, sched, (3, 1, 28, 28), y, steps=5, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b)


# --- classifier-free guidance -----------------------------------------------------------


class LabelEchoModel(torch.nn.Module):
    """eps = label value (null label -> 10), so guidance arithmetic is easy to check."""

    num_classes = 10

    def forward(self, x, t, y):
        return y.float().view(-1, 1).expand_as(x)


@pytest.mark.parametrize("w,expected", [(0.0, 10.0), (1.0, 3.0), (2.0, -4.0), (3.0, -11.0)])
def test_guidance_interpolates_and_extrapolates(w, expected):
    x = torch.zeros(2, 1)
    eps = guided_eps(LabelEchoModel(), x, torch.zeros(2, dtype=torch.long), torch.tensor([3, 3]), w, null_label=10)
    # eps_u + w (eps_c - eps_u) = 10 + w (3 - 10)
    torch.testing.assert_close(eps, torch.full_like(x, expected))


def test_unconditional_sampling_uses_null_label():
    x = torch.zeros(2, 1)
    eps = guided_eps(LabelEchoModel(), x, torch.zeros(2, dtype=torch.long), None, 5.0, null_label=10)
    torch.testing.assert_close(eps, torch.full_like(x, 10.0))


# --- UNet ---------------------------------------------------------------------------------


def test_unet_shapes_and_conditioning():
    torch.manual_seed(0)
    model = UNet(base_ch=16, ch_mults=(1, 2, 4), num_res_blocks=1, num_classes=10)
    torch.nn.init.normal_(model.out[-1].weight, std=0.1)  # undo zero init so outputs differ
    x = torch.randn(4, 1, 28, 28)
    t = torch.tensor([0, 10, 500, 999])
    out = model(x, t, torch.tensor([0, 1, 2, 10]))  # 10 = null label
    assert out.shape == x.shape
    model.eval()
    same_t = torch.full((4,), 100)
    a = model(x, same_t, torch.zeros(4, dtype=torch.long))
    b = model(x, same_t, torch.ones(4, dtype=torch.long))
    assert not torch.allclose(a, b), "class label must change the prediction"
