import io
import json
import re
from pathlib import Path

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


# --- static export ----------------------------------------------------------------------


def test_export_writes_consistent_manifest(model, tmp_path):
    from ddpm.export_web import export

    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<!doctype html>")
    out = tmp_path / "docs"
    manifest = export(model, NoiseSchedule(50), out, web_dir, guidances=(0, 2), steps=(5,), traj_frames=3,
                      model_info={"train_steps": 1, "params": 2})

    assert (out / "index.html").exists()
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert set(manifest["grid"]) == {"w0_s5", "w2_s5"}
    assert manifest["ddpm"]["sprite"] and len(manifest["trajectory"]["frames"]) == 3
    sprites = list(manifest["grid"].values()) + [manifest["ddpm"]["sprite"]] + manifest["trajectory"]["frames"]
    for rel in sprites:
        assert not rel.startswith("/")
        assert Image.open(out / rel).size == (280, 28)
    assert manifest["model"] == {"train_steps": 1, "params": 2}


# --- local server -----------------------------------------------------------------------


@pytest.fixture()
def client(model, tmp_path):
    from fastapi.testclient import TestClient

    from ddpm.serve import create_app

    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "index.html").write_text("<!doctype html><title>demo</title>")
    assets = tmp_path / "docs"
    assets.mkdir()
    (assets / "manifest.json").write_text('{"grid": {}}')
    app = create_app(model, NoiseSchedule(50), web_dir, assets, {"train_steps": 3, "params": 4})
    return TestClient(app)


def test_health_reports_model(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": {"train_steps": 3, "params": 4}}


def test_sample_returns_frames(client):
    r = client.post("/api/sample", json={"digit": 4, "steps": 5, "frames": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["image"].startswith("data:image/png;base64,")
    assert len(body["frames"]) == 4
    assert all(f.startswith("data:image/png;base64,") for f in body["frames"])


@pytest.mark.parametrize("payload", [{"digit": 10}, {"steps": 0}, {"sampler": "x"}, {"frames": 1}, {"eta": 2}])
def test_invalid_requests_rejected(client, payload):
    assert client.post("/api/sample", json=payload).status_code == 422


def test_ddpm_ignores_steps(client):
    r = client.post("/api/sample", json={"sampler": "ddpm", "steps": 5, "frames": 3})
    assert r.status_code == 200
    assert len(r.json()["frames"]) == 3


def test_requests_are_serialized(client, monkeypatch):
    import threading
    import time

    import ddpm.serve as serve

    active, peak = [0], [0]
    real = serve.sample_frames

    def slow(*args, **kwargs):
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.2)
        active[0] -= 1
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "sample_frames", slow)
    threads = [threading.Thread(target=client.post, args=("/api/sample",), kwargs={"json": {"steps": 2, "frames": 2}})
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 1


def test_missing_checkpoint_message():
    from ddpm.serve import main

    with pytest.raises(SystemExit, match="make train"):
        main(["--ckpt", "does-not-exist.pt"])


def test_serves_page_and_manifest(client):
    assert client.get("/").status_code == 200
    assert "demo" in client.get("/").text
    assert client.get("/manifest.json").json() == {"grid": {}}


# --- web page ---------------------------------------------------------------------------

WEB = Path(__file__).resolve().parent.parent / "web"


def test_page_uses_relative_urls():
    # GitHub Pages serves the site under /diffusion-from-scratch/, so absolute paths would break.
    for f in ("index.html", "app.js", "style.css"):
        text = (WEB / f).read_text()
        for absolute in ('src="/', "href=\"/", 'fetch("/', "fetch('/", "url(/"):
            assert absolute not in text, f"{f} contains {absolute}"


def test_page_has_required_controls():
    html = (WEB / "index.html").read_text()
    for el_id in ("static-view", "guidance", "steps", "digits", "play", "live-view", "status"):
        assert f'id="{el_id}"' in html, el_id
    for input_id in re.findall(r'<input[^>]*\bid="([^"]+)"', html):
        assert f'for="{input_id}"' in html, f"input #{input_id} has no label"
