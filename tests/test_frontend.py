import io

import pytest
import torch
from PIL import Image

from ddpm.schedule import NoiseSchedule
from ddpm.unet import UNet


def tiny_model():
    torch.manual_seed(0)
    model = UNet(base_ch=8, ch_mults=(1, 2), num_res_blocks=1, attn_levels=(), num_classes=10).eval()
    torch.nn.init.normal_(model.out[-1].weight, std=0.1)  # undo zero init so samples vary
    return model


@pytest.fixture(scope="module")
def model():
    return tiny_model()


@pytest.fixture(scope="module")
def sched():
    return NoiseSchedule(1000)


# --- rendering helpers ------------------------------------------------------------------


def test_sample_frames_shapes_and_endpoints(model, sched):
    from ddpm.render import sample_frames

    final, frames = sample_frames(model, sched, torch.tensor([3, 7]), guidance=2.0, sampler="ddim", steps=10, eta=0.0, seed=1, frames=4)
    assert final.shape == (2, 1, 28, 28)
    assert len(frames) == 4
    assert all(f.shape == (2, 1, 28, 28) for f in frames)
    torch.testing.assert_close(frames[-1], final)


def test_sample_frames_clamps_to_trajectory_length(model, sched):
    from ddpm.render import sample_frames

    _, frames = sample_frames(model, sched, torch.tensor([0]), guidance=1.0, sampler="ddim", steps=10, eta=0.0, seed=0, frames=50)
    assert len(frames) == 11  # x_T plus one state per DDIM step


def test_same_seed_same_result(model, sched):
    from ddpm.render import sample_frames

    kwargs = dict(guidance=2.0, sampler="ddim", steps=5, eta=0.0, seed=5, frames=2)
    a, _ = sample_frames(model, sched, torch.tensor([1, 2]), **kwargs)
    b, _ = sample_frames(model, sched, torch.tensor([1, 2]), **kwargs)
    torch.testing.assert_close(a, b)


def test_png_row_dimensions():
    from ddpm.render import png_row

    img = Image.open(io.BytesIO(png_row(torch.zeros(10, 1, 28, 28))))
    assert img.size == (280, 28)
    assert img.mode == "L"


def test_png_data_url_prefix():
    from ddpm.render import png_data_url

    assert png_data_url(torch.zeros(1, 1, 28, 28)).startswith("data:image/png;base64,")
