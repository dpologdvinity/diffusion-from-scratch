// Static mode reads precomputed sprites from manifest.json. Live mode turns on only when the
// local server (make serve) answers api/health. All URLs are relative so the page also works
// under the GitHub Pages path prefix.

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const FRAME_MS = 120;

let manifest = null;
let playing = false;

function stepsLabel(index) {
  return index === manifest.steps.length ? `DDPM, ${manifest.ddpm.steps} steps` : `DDIM, ${manifest.steps[index]} steps`;
}

function currentSetting() {
  const si = Number($("steps").value);
  if (si === manifest.steps.length) {
    return { src: manifest.ddpm.sprite, guidance: manifest.ddpm.guidance, label: stepsLabel(si) };
  }
  const g = manifest.guidance[Number($("guidance").value)];
  return { src: manifest.grid[`w${g}_s${manifest.steps[si]}`], guidance: g, label: stepsLabel(si) };
}

function render() {
  if (playing) return;
  const isDdpm = Number($("steps").value) === manifest.steps.length;
  $("guidance").disabled = isDdpm;
  const s = currentSetting();
  $("guidance-value").textContent = isDdpm ? `${s.guidance} (fixed for DDPM)` : String(s.guidance);
  $("steps-value").textContent = s.label;
  $("digits").src = s.src;
  $("digits").alt = `Digits 0 to 9 generated with guidance ${s.guidance}, ${s.label}.`;
}

const animationRuns = new WeakMap(); // img -> id of its current animation; older runs stop ticking

function animate(img, frames, altFinal, done) {
  const run = (animationRuns.get(img) || 0) + 1;
  animationRuns.set(img, run);
  if (reducedMotion) {
    img.src = frames[frames.length - 1];
    img.alt = altFinal;
    done();
    return;
  }
  let i = 0;
  const tick = () => {
    if (animationRuns.get(img) !== run) return;
    img.src = frames[i];
    img.alt = `Denoising, frame ${i + 1} of ${frames.length}.`;
    i += 1;
    if (i < frames.length) setTimeout(tick, FRAME_MS);
    else {
      img.alt = altFinal;
      done();
    }
  };
  tick();
}

function play() {
  const t = manifest.trajectory;
  const frames = t.frames;
  frames.forEach((src) => { new Image().src = src; }); // warm the cache before animating
  playing = true;
  $("play").disabled = true;
  animate($("digits"), frames, `Digits 0 to 9 after ${t.steps} DDIM steps at guidance ${t.guidance}.`, () => {
    setTimeout(() => {
      playing = false;
      $("play").disabled = false;
      render();
    }, 900);
  });
}

async function loadManifest() {
  try {
    const r = await fetch("manifest.json");
    if (!r.ok) throw new Error(String(r.status));
    manifest = await r.json();
  } catch {
    $("static-view").hidden = true;
    return false;
  }
  $("guidance").max = String(manifest.guidance.length - 1);
  $("guidance").value = String(Math.max(0, manifest.guidance.indexOf(2)));
  $("steps").max = String(manifest.steps.length); // last stop is DDPM
  $("steps").value = String(Math.max(0, manifest.steps.indexOf(50)));
  $("guidance").addEventListener("input", render);
  $("steps").addEventListener("input", render);
  $("play").addEventListener("click", play);
  render();
  return true;
}

async function probeLive() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 1500);
  try {
    const r = await fetch("api/health", { signal: ctrl.signal });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

function showErrors(body) {
  const detail = Array.isArray(body && body.detail) ? body.detail : [];
  $("live-error").textContent = detail.length
    ? detail.map((d) => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join("; ")
    : "Sampling failed.";
}

function setupLive() {
  $("live-view").hidden = false;
  const bind = (id) => {
    const update = () => { $(`${id}-value`).textContent = $(id).value; };
    $(id).addEventListener("input", update);
    update();
  };
  bind("live-guidance");
  bind("live-eta");
  const syncSampler = () => {
    const ddim = $("live-sampler").value === "ddim";
    $("live-steps").disabled = !ddim;
    $("live-eta").disabled = !ddim;
  };
  $("live-sampler").addEventListener("change", syncSampler);
  syncSampler();

  $("live-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = event.submitter || $("live-form").querySelector("button");
    const req = {
      digit: Number($("live-digit").value),
      guidance: Number($("live-guidance").value),
      sampler: $("live-sampler").value,
      steps: Number($("live-steps").value),
      eta: Number($("live-eta").value),
      seed: Number($("live-seed").value),
      frames: 25,
    };
    $("live-error").textContent = "";
    button.disabled = true;
    $("live-meta").textContent = "Sampling…";
    try {
      const r = await fetch("api/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
      const body = await r.json();
      if (!r.ok) {
        showErrors(body);
        $("live-meta").textContent = "";
        return;
      }
      const steps = req.sampler === "ddpm" ? "DDPM 1000 steps" : `DDIM ${req.steps} steps`;
      $("live-meta").textContent = `Digit ${req.digit}, guidance ${req.guidance}, ${steps}, seed ${req.seed}: ${body.seconds}s on one CPU thread.`;
      document.querySelector(".live-output").hidden = false;
      animate($("live-image"), body.frames, `Generated digit ${req.digit}, guidance ${req.guidance}, ${steps}.`, () => {});
    } catch {
      $("live-error").textContent = "Could not reach the local server.";
      $("live-meta").textContent = "";
    } finally {
      button.disabled = false;
    }
  });
}

async function init() {
  const hasStatic = await loadManifest();
  const health = await probeLive();
  const trained = (info) => (info && info.train_steps ? ` (model trained ${info.train_steps.toLocaleString()} steps)` : "");
  if (health) {
    setupLive();
    $("status").textContent = `Live model running locally${trained(health.model)}.`;
  } else if (hasStatic) {
    $("status").textContent = `Precomputed samples${trained(manifest.model)}. Clone the repo and run make serve for live sampling.`;
  } else {
    $("status").textContent = "No samples available. Run make web to precompute them, or make serve for live sampling.";
  }
}

init();
