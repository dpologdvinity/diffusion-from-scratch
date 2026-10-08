# Interactive frontend: design

## Goal

Let people *see* what the model does without reading code.

- **Recruiters and visitors** open a static page (GitHub Pages) and drag sliders for guidance weight and sampler steps; samples update instantly from precomputed images. Zero setup.
- **Engineers who clone the repo** run `make serve`, open `localhost:8000`, pick a digit, and watch it denoise live with any seed, guidance, step count, sampler, and η.

Both modes use the **same page**, so the two frontends never drift apart.

## Constraints

- Static mode is fully static: no server, no inference, works from `docs/` on GitHub Pages.
- Plain HTML, CSS, and vanilla JS; no framework or build step.
- Live mode adds only `fastapi` and `uvicorn` (free, MIT/BSD).
- CPU budget: the server uses one torch thread and runs one sampling request at a time; asset precomputation runs once, in the background, at `nice -n 10` on at most two threads.
- The server binds to `127.0.0.1` only.

## Components

| Unit | Responsibility | Depends on |
|---|---|---|
| `web/index.html`, `web/app.js`, `web/style.css` | The UI. On load, fetches `manifest.json` and renders static mode; then probes `GET /api/health` and, if it answers, reveals live controls. | nothing (browser only) |
| `ddpm/export_web.py` | Precomputes static assets into `docs/`: copies `web/*`, writes sprite sheets (PNG) and `manifest.json`. | trained checkpoint, `ddpm.sampling` |
| `ddpm/serve.py` | FastAPI app: serves `web/` at `/` plus the exported `manifest.json` and sprites from `docs/` when present, `GET /api/health`, `POST /api/sample`. Loads the EMA checkpoint once at startup. | trained checkpoint, `ddpm.sampling`, fastapi, uvicorn |
| `Makefile` | `make web` (export static assets), `make serve` (run local server). | the above |

## Static assets (`docs/`)

Precomputed for digits 0–9 with a fixed seed per digit (the same starting noise for every setting, so changes are attributable to the control, not to randomness):

- **Guidance × steps grid:** guidance w ∈ {0, 1, 2, 3, 5} × DDIM steps ∈ {10, 20, 50, 100}, plus DDPM 1000 at w = 2. One 28×28 image per digit per cell.
- **Denoising trajectories:** for each digit, 11 frames of x_t along a 50-step DDIM run at w = 2, for the animation.
- **Sprite layout:** one PNG per (setting) holding the 10 digits in a row; `manifest.json` lists settings → sprite file, plus model metadata (training steps, parameter count) read from the checkpoint.

Size target: well under 1 MB total (28×28 grayscale PNGs compress to a few hundred bytes each).

## Live API

- `GET /api/health` → `{"status": "ok", "datasets": {"mnist": {"train_steps": int, "params": int}, ...}}` (one entry per dataset with weights)
- `POST /api/sample` with JSON `{dataset: "mnist" | "fashion", label: 0–9, guidance: 0–10, sampler: "ddim" | "ddpm", steps: 1–1000 (DDIM only), eta: 0–1, seed: 0–2^31-1, frames: 2–50}` → `{"image": <PNG data URL>, "frames": [<PNG data URL>, ...], "seconds": float}`
- Validation via Pydantic models; out-of-range values return 422 with field errors, which the UI shows inline.
- A lock serializes sampling so concurrent requests queue instead of multiplying CPU use.

## UI behavior

- Static mode: a digit row for the current (guidance, steps) setting; two sliders snap to the precomputed values; a "watch it denoise" button plays the trajectory frames.
- Live mode (only when `/api/health` responds): adds digit picker, free seed input, sampler toggle, η slider, and a Generate button; results animate from noise to sample. A small status line shows "live model" vs "precomputed samples".
- If the health probe fails, the page stays in static mode with no error shown. If `manifest.json` is missing (live server run before `make web`), the static grid is hidden and only live controls show.
- Accessible basics: labelled controls, keyboard-operable sliders, `prefers-reduced-motion` shows the final frame instead of animating, images carry alt text describing digit and settings.

## Testing

- `export_web`: with a tiny randomly initialized model, writes a manifest whose every referenced sprite exists and has the expected dimensions.
- `serve`: FastAPI `TestClient` with a tiny model: health returns metadata; a valid request returns the right number of frames as PNG data URLs; invalid digit/steps return 422.
- CI runs both with the existing suite (no trained checkpoint needed).

## Out of scope

Hosting the live server online, mobile-specific layouts beyond a responsive single column.

## Addendum: FashionMNIST

Added after the first release. A dataset picker switches both modes. Static assets live per dataset: MNIST at the site root (`manifest.json`, `sprites/`) and FashionMNIST under `fashion/`; each manifest carries its class `labels`, and the page resolves sprite paths against the manifest's folder. The live server loads every dataset with weights (`checkpoints/<dataset>.pt`, else the bundled `models/<dataset>.pt`) and rejects requests for a dataset it has no weights for with a 422.
