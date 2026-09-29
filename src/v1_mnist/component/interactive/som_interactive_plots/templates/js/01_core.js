// 01_core.js - Payload, colour ramps, small utilities, tooltip, shared state.
// Fragment: all js/*.js files are concatenated (in numeric order) into ONE function scope
// by html_report.py, so they share variables. Do not add import/export.

const D = JSON.parse(document.getElementById("som-data").textContent);
const M = D.meta, N = M.num_neurons, SVGNS = "http://www.w3.org/2000/svg";
const PX = M.feature_prefix;
const splitName = s => s === "val" ? "validation" : "train";

// ---------------------------------------------------------------- colour
const SEQ_BLUE = ["#e3eefc","#b7d3f6","#86b6ef","#5598e7","#2a78d6","#1c5cab","#104281","#0d366b"];
const SEQ_ORANGE = ["#fdeae1","#f9cbb5","#f4a583","#ee8055","#eb6834","#c24d1c","#943812","#6b270c"];
const DIVERGING = ["#0d366b","#1c5cab","#5598e7","#b7d3f6","#f0efec","#f5c6c5","#e97f7e","#d03b3b","#8f1f1f"];
const hex2rgb = h => [1,3,5].map(i => parseInt(h.slice(i, i+2), 16));
const rgb2hex = c => "#" + c.map(v => Math.round(v).toString(16).padStart(2, "0")).join("");
function ramp(stops, t) {
  if (t == null || !isFinite(t)) return null;
  t = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(t)), f = t - i;
  const a = hex2rgb(stops[i]), b = hex2rgb(stops[i+1]);
  return rgb2hex(a.map((v, k) => v + (b[k] - v) * f));
}
function lum(h) {
  const c = hex2rgb(h).map(v => { v /= 255; return v <= 0.04045 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); });
  return 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2];
}
const inkOn = fill => (fill && lum(fill) < 0.28) ? "#ffffff" : "#0b0b0b";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------------------------------------------------------------- utils
// 3 significant digits: works for weights (~0.6), distances and fc2 dims
// whose raw activations only span ~1e-4.
const fmt = v => {
  if (v == null || !isFinite(v)) return "–";
  v = Number(v);
  if (v === 0) return "0";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return String(Number(v.toPrecision(3)));
};
const pct = (v, d = 1) => v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%";
const int = v => v == null ? "–" : Number(v).toLocaleString();
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function extent(arr) {
  let lo = Infinity, hi = -Infinity;
  for (const v of arr) if (v != null && isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return lo === Infinity ? [0, 1] : [lo, hi];
}
const rowCol = n => [Math.floor(n / M.grid[0]), n % M.grid[0]];
const featName = f => `${PX}_${f}`;

// ---------------------------------------------------------------- tooltip
const tip = document.getElementById("tip");
function showTip(evt, title, rows) {
  tip.innerHTML = `<div class="t">${title}</div>` +
    rows.map(([k, v]) => `<div class="r"><span>${k}</span><b>${v}</b></div>`).join("");
  tip.style.display = "block";
  const pad = 14, r = tip.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.display = "none"; };

// ---------------------------------------------------------------- state
const state = { view: "hits", opt: null, split: M.detail_split, selected: null, tab: "classes" };

// neighbour distances per neuron (U-matrix)
const nbr = Array.from({ length: N }, () => []);
for (const [i, j, d] of D.edges) { nbr[i].push(d); nbr[j].push(d); }
const meanNbr = nbr.map(a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
const edgeExt = extent(D.edges.map(e => e[2]));

// mean weight vector across neurons (Weights tab reference line)
const meanW = D.w[0].map((_, k) => D.w.reduce((s, r) => s + r[k], 0) / N);

