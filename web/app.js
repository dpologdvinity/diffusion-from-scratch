// Static mode reads precomputed sprites from each dataset's manifest.json. Live mode turns on only
// when the local server (make serve) answers api/health. All URLs are relative so the page also
// works under the GitHub Pages path prefix.

const $ = (id) => document.getElementById(id);
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const FRAME_MS = 120;

const DATASETS = [
  { id: "mnist", base: "", labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] },
  { id: "fashion", base: "fashion/",
    labels: ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"] },
];

let manifest = null; // manifest of the selected dataset, with paths resolved
const manifests = {};
let live = {}; // dataset -> model info, for datasets the local server can sample
let current = "mnist";
let playing = false;

// Manifest paths are relative to the manifest's own folder; resolve them against the page.
function withBase(m, base) {
  for (const key of Object.keys(m.grid)) m.grid[key] = base + m.grid[key];
  m.ddpm.sprite = base + m.ddpm.sprite;
  m.trajectory.frames = m.trajectory.frames.map((f) => base + f);
  return m;
}

function classList() {
  return (manifest && manifest.labels ? manifest.labels : []).join(", ");
}

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
  const guidanceText = isDdpm ? `${s.guidance} (fixed for DDPM)` : String(s.guidance);
  $("guidance-value").textContent = guidanceText;
  $("steps-value").textContent = s.label;
  // Slider positions are indices into the manifest; tell assistive tech the real values.
  $("guidance").setAttribute("aria-valuetext", guidanceText);
  $("steps").setAttribute("aria-valuetext", s.label);
  $("digits").src = s.src;
  $("digits").alt = `One sample per class (${classList()}) generated with guidance ${s.guidance}, ${s.label}.`;
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
  $("guidance-value").textContent = `${t.guidance} (during playback)`;
  $("steps-value").textContent = `DDIM, ${t.steps} steps (during playback)`;
  animate($("digits"), frames, `One sample per class (${classList()}) after ${t.steps} DDIM steps at guidance ${t.guidance}.`, () => {
    setTimeout(() => {
      playing = false;
      $("play").disabled = false;
      render();
    }, 900);
  });
}

async function fetchManifest(d) {
  try {
    const r = await fetch(`${d.base}manifest.json`);
    if (!r.ok) return null;
    return withBase(await r.json(), d.base);
  } catch {
    return null;
  }
}

function configureSliders() {
  $("guidance").max = String(manifest.guidance.length - 1);
  $("guidance").value = String(Math.max(0, manifest.guidance.indexOf(2)));
  $("steps").max = String(manifest.steps.length); // last stop is DDPM
  $("steps").value = String(Math.max(0, manifest.steps.indexOf(50)));
}

function trained(info) {
  return info && info.train_steps ? ` (model trained ${info.train_steps.toLocaleString()} steps)` : "";
}

function updateStatus() {
  if (live[current]) {
    $("status").textContent = `Live model running locally${trained(live[current])}.`;
  } else if (manifest) {
    $("status").textContent = `Precomputed samples${trained(manifest.model)}. Clone the repo and run make serve for live sampling.`;
  } else {
    $("status").textContent = "No samples available. Run make web to precompute them, or make serve for live sampling.";
  }
}

function selectDataset(id) {
  current = id;
  manifest = manifests[id];
  $("static-view").hidden = !manifest;
  if (manifest) {
    configureSliders();
    render();
  }
  $("live-view").hidden = !live[id];
  const labels = (manifest && manifest.labels) || DATASETS.find((d) => d.id === id).labels;
  const picker = $("live-label");
  const keep = Number(picker.value || 7);
  picker.replaceChildren(...labels.map((name, i) => new Option(name, String(i), false, i === keep)));
  updateStatus();
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

// The DDIM-only fields are disabled for DDPM; leave them out so stale values can't fail validation.
function buildRequest(v) {
  const req = {
    dataset: v.dataset, label: Number(v.label), guidance: Number(v.guidance), sampler: v.sampler, seed: Number(v.seed), frames: 25,
  };
  if (v.sampler === "ddim") {
    req.steps = Number(v.steps);
    req.eta = Number(v.eta);
  }
  return req;
}

function errorMessage(status, text) {
  try {
    const detail = JSON.parse(text).detail;
    if (Array.isArray(detail) && detail.length) {
      return detail.map((d) => `${d.loc[d.loc.length - 1]}: ${d.msg}`).join("; ");
    }
  } catch {
    // not JSON: fall through to the status code
  }
  return `Server error (HTTP ${status}).`;
}

function setupLive() {
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
    const req = buildRequest({
      dataset: current,
      label: $("live-label").value,
      guidance: $("live-guidance").value,
      sampler: $("live-sampler").value,
      steps: $("live-steps").value,
      eta: $("live-eta").value,
      seed: $("live-seed").value,
    });
    $("live-error").textContent = "";
    button.disabled = true;
    $("live-meta").textContent = "Sampling…";
    let r;
    try {
      r = await fetch("api/sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
      });
    } catch {
      $("live-error").textContent = "Could not reach the local server.";
      $("live-meta").textContent = "";
      button.disabled = false;
      return;
    }
    try {
      const text = await r.text();
      if (!r.ok) {
        $("live-error").textContent = errorMessage(r.status, text);
        $("live-meta").textContent = "";
        return;
      }
      const body = JSON.parse(text);
      const steps = req.sampler === "ddpm" ? "DDPM 1000 steps" : `DDIM ${req.steps} steps`;
      const name = $("live-label").options[$("live-label").selectedIndex].text;
      $("live-meta").textContent = `${name}, guidance ${req.guidance}, ${steps}, seed ${req.seed}: ${body.seconds}s on one CPU thread.`;
      document.querySelector(".live-output").hidden = false;
      animate($("live-image"), body.frames, `Generated ${name}, guidance ${req.guidance}, ${steps}.`, () => {});
    } catch {
      $("live-error").textContent = errorMessage(r.status, "");
      $("live-meta").textContent = "";
    } finally {
      button.disabled = false;
    }
  });
}

// ?dataset=fashion opens that dataset directly (falls back to the first available one).
function initialDataset(search, available) {
  const wanted = new URLSearchParams(search).get("dataset");
  return available.includes(wanted) ? wanted : available[0];
}

async function init() {
  for (const d of DATASETS) manifests[d.id] = await fetchManifest(d);
  const health = await probeLive();
  live = (health && health.datasets) || {};
  const available = DATASETS.filter((d) => manifests[d.id] || live[d.id]).map((d) => d.id);
  for (const opt of Array.from($("dataset").options || [])) opt.hidden = !available.includes(opt.value);
  $("dataset-control").hidden = available.length < 2;
  $("guidance").addEventListener("input", render);
  $("steps").addEventListener("input", render);
  $("play").addEventListener("click", play);
  $("dataset").addEventListener("change", () => selectDataset($("dataset").value));
  if (Object.keys(live).length) setupLive();
  if (!available.length) {
    $("static-view").hidden = true;
    updateStatus();
    return;
  }
  const first = initialDataset(window.location ? window.location.search : "", available);
  $("dataset").value = first;
  selectDataset(first);
}

init();
