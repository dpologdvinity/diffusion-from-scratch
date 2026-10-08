// Loads web/app.js in a sandbox with a stubbed DOM and manual timers, so UI logic can be
// tested without a browser. Run with: node --test tests/web/*.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

function load() {
  const timers = [];
  const el = () => ({ hidden: false, disabled: false, value: "0", textContent: "", src: "", alt: "", max: "0",
    attrs: {}, setAttribute(k, v) { this.attrs[k] = String(v); }, addEventListener() {}, querySelector: () => el() });
  const elements = {};
  const sandbox = {
    document: { getElementById: (id) => (elements[id] ??= el()), querySelector: () => el() },
    window: { matchMedia: () => ({ matches: false }) },
    fetch: () => Promise.reject(new Error("offline")),
    setTimeout: (fn) => { timers.push(fn); return timers.length; },
    clearTimeout() {},
    AbortController: class { constructor() { this.signal = {}; } abort() {} },
    Image: class {},
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(readFileSync(new URL("../../web/app.js", import.meta.url), "utf8"), sandbox);
  const flush = () => { while (timers.length) timers.shift()(); };
  const run = (code) => vm.runInContext(code, sandbox);
  const get = (id) => sandbox.document.getElementById(id);
  return { sandbox, flush, run, get };
}

const MANIFEST = {
  guidance: [0, 1, 2, 3, 5], steps: [10, 20, 50, 100],
  grid: Object.fromEntries([0, 1, 2, 3, 5].flatMap((g) => [10, 20, 50, 100].map((s) => [`w${g}_s${s}`, `g${g}s${s}.png`]))),
  ddpm: { guidance: 2, steps: 1000, sprite: "ddpm.png" },
  trajectory: { guidance: 2, steps: 50, frames: ["t0.png", "t1.png"] },
};

test("a newer animation on the same image cancels the older one", () => {
  const { sandbox, flush } = load();
  const img = { src: "", alt: "" };
  sandbox.animate(img, ["a1", "a2", "a3", "a4"], "A done", () => {});
  sandbox.animate(img, ["b1", "b2"], "B done", () => {});
  flush();
  assert.equal(img.src, "b2");
  assert.equal(img.alt, "B done");
});

test("a DDPM request leaves out the disabled DDIM-only fields", () => {
  const { sandbox } = load();
  const values = { dataset: "mnist", label: "3", guidance: "2", sampler: "ddpm", steps: "", eta: "0.5", seed: "4" };
  const req = sandbox.buildRequest(values);
  assert.equal(req.sampler, "ddpm");
  assert.equal("steps" in req, false);
  assert.equal("eta" in req, false);
  assert.equal(sandbox.buildRequest({ ...values, sampler: "ddim", steps: "20" }).steps, 20);
});

test("errors that are not validation JSON still get a useful message", () => {
  const { sandbox } = load();
  assert.equal(sandbox.errorMessage(500, "Internal Server Error"), "Server error (HTTP 500).");
  const detail = JSON.stringify({ detail: [{ loc: ["body", "digit"], msg: "too big" }] });
  assert.equal(sandbox.errorMessage(422, detail), "digit: too big");
});

test("sliders announce their real values, not their positions", () => {
  const { run, get } = load();
  run(`manifest = ${JSON.stringify(MANIFEST)}`);
  get("guidance").value = "4";
  get("steps").value = "4";
  run("render()");
  assert.equal(get("guidance").attrs["aria-valuetext"], "2 (fixed for DDPM)");
  assert.equal(get("steps").attrs["aria-valuetext"], "DDPM, 1000 steps");
  get("steps").value = "0";
  run("render()");
  assert.equal(get("guidance").attrs["aria-valuetext"], "5");
});

test("while the trajectory plays, the labels describe the trajectory setting", () => {
  const { run, get } = load();
  run(`manifest = ${JSON.stringify(MANIFEST)}`);
  get("guidance").value = "0";
  get("steps").value = "0";
  run("render(); play()");
  assert.equal(get("guidance-value").textContent, "2 (during playback)");
  assert.equal(get("steps-value").textContent, "DDIM, 50 steps (during playback)");
});

test("a dataset's manifest paths are resolved under its folder", () => {
  const { sandbox } = load();
  const m = sandbox.withBase(JSON.parse(JSON.stringify(MANIFEST)), "fashion/");
  assert.equal(m.grid.w2_s50, "fashion/g2s50.png");
  assert.equal(m.ddpm.sprite, "fashion/ddpm.png");
  assert.deepEqual(m.trajectory.frames, ["fashion/t0.png", "fashion/t1.png"]);
  assert.equal(sandbox.withBase(JSON.parse(JSON.stringify(MANIFEST)), "").grid.w0_s10, "g0s10.png");
});

test("live requests name the dataset and class", () => {
  const { sandbox } = load();
  const req = sandbox.buildRequest({ dataset: "fashion", label: "8", guidance: "1", sampler: "ddim", steps: "20", eta: "0", seed: "3" });
  assert.equal(req.dataset, "fashion");
  assert.equal(req.label, 8);
  assert.equal("digit" in req, false);
});
