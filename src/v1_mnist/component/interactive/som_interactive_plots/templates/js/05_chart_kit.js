// 05_chart_kit.js - Tiny SVG chart helpers (axes, ticks, bars) shared by the detail tabs.
// Fragment: all js/*.js files are concatenated (in numeric order) into ONE function scope
// by html_report.py, so they share variables. Do not add import/export.

// -- small chart kit ------------------------------------------------------
function chart(body, w = 460, h = 250, m = { t: 12, r: 12, b: 34, l: 44 }) {
  const s = el("svg", { viewBox: `0 0 ${w} ${h}` });
  body.appendChild(s);
  return { s, w, h, m, iw: w - m.l - m.r, ih: h - m.t - m.b };
}
const lin = (d0, d1, r0, r1) => x => r0 + (x - d0) / ((d1 - d0) || 1) * (r1 - r0);
// cover=true extends the last tick to >= hi, so a chart whose domain is
// [0, lastTick] never clips its tallest mark.
function niceTicks(lo, hi, n = 5, cover = false) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(k => span / k <= n) || mag * 10;
  const end = cover ? Math.ceil(hi / step - 1e-9) * step : hi;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= end + step * 1e-6; v += step) out.push(+v.toPrecision(12));
  return out;
}
function yAxis(c, y, ticks, f = v => v) {
  const g = el("g", { class: "axis" }, c.s);
  for (const t of ticks) {
    el("line", { class: "gridline", x1: c.m.l, x2: c.w - c.m.r, y1: y(t), y2: y(t) }, g);
    const tx = el("text", { x: c.m.l - 6, y: y(t) + 3.5, "text-anchor": "end" }, g); tx.textContent = f(t);
  }
}
function xLabel(c, text) {
  const t = el("text", { class: "lbl", x: c.m.l + c.iw / 2, y: c.h - 4, "text-anchor": "middle" }, c.s); t.textContent = text;
}
function barPath(x, y, w, h, r = 4) {
  r = Math.min(r, w / 2, h);
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
function featSelect(body, label, key) {
  const wrap = document.createElement("div"); wrap.className = "inline-ctl";
  wrap.innerHTML = `<span>${label}</span>`;
  const sel = document.createElement("select");
  for (const f of topF) sel.add(new Option(featName(f), f));
  sel.value = String(detailFeat[key]);
  sel.addEventListener("change", () => { detailFeat[key] = Number(sel.value); drawDetail(); });
  wrap.appendChild(sel); body.appendChild(wrap);
  return wrap;
}

