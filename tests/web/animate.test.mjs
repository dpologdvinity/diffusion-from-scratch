// Loads web/app.js in a sandbox with a stubbed DOM and manual timers, so UI logic can be
// tested without a browser. Run with: node --test tests/web/*.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

function load() {
  const timers = [];
  const el = () => ({ hidden: false, disabled: false, value: "0", textContent: "", src: "", alt: "", max: "0",
    addEventListener() {}, querySelector: () => el() });
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
  return { sandbox, flush };
}

test("a newer animation on the same image cancels the older one", () => {
  const { sandbox, flush } = load();
  const img = { src: "", alt: "" };
  sandbox.animate(img, ["a1", "a2", "a3", "a4"], "A done", () => {});
  sandbox.animate(img, ["b1", "b2"], "B done", () => {});
  flush();
  assert.equal(img.src, "b2");
  assert.equal(img.alt, "B done");
});
