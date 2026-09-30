// 06_detail_tabs.js - One function per detail tab (classes, histogram, box, violin, scatter, weights, errors, samples).
// Fragment: all js/*.js files are concatenated (in numeric order) into ONE function scope
// by html_report.py, so they share variables. Do not add import/export.

// -- Classes (NNSOM 'pie' / 'stem' buttons) --------------------------------
function tabClasses(body, i) {
  const ds = D.splits[M.detail_split], counts = ds.cat[i];
  const c = chart(body);
  const max = Math.max(...counts, 1), ticks = niceTicks(0, max, 4, true);
  const y = lin(0, ticks[ticks.length - 1] || max, c.m.t + c.ih, c.m.t);
  yAxis(c, y, ticks, v => int(v));
  const bw = c.iw / M.num_classes, gap = 2;
  const best = counts.indexOf(Math.max(...counts));
  counts.forEach((v, d) => {
    const x = c.m.l + d * bw + gap / 2, top = y(v), h = c.m.t + c.ih - top;
    if (v > 0) el("path", { d: barPath(x, top, bw - gap, h), fill: css("--series-1") }, c.s);
    const hit = el("rect", { x: c.m.l + d * bw, y: c.m.t, width: bw, height: c.ih, fill: "transparent" }, c.s);
    hit.addEventListener("mousemove", e => showTip(e, `Digit ${d}`, [["samples", int(v)], ["share", pct(v / (ds.hits[i] || 1))]]));
    hit.addEventListener("mouseleave", hideTip);
    const t = el("text", { class: "lbl", x: c.m.l + d * bw + bw / 2, y: c.m.t + c.ih + 15, "text-anchor": "middle" }, c.s);
    t.textContent = d;
    if (d === best && v > 0) {
      const lt = el("text", { x: c.m.l + d * bw + bw / 2, y: top - 5, "text-anchor": "middle", "font-size": 11,
                               fill: css("--ink"), "font-weight": 600 }, c.s);
      lt.textContent = `${int(v)} (${pct(v / ds.hits[i], 0)})`;
    }
  });
  xLabel(c, "true digit");
  const note = document.createElement("div"); note.className = "note";
  note.textContent = "Samples per true digit in this neuron. Hover a bar for exact counts.";
  body.appendChild(note);
}

// -- Histogram (NNSOM 'hist') ----------------------------------------------
function tabHist(body, i, idx) {
  featSelect(body, "Feature", "hist");
  const f = detailFeat.hist, all = P0.feats[String(f)];
  const [lo, hi] = extent(all), B = 24, bw = (hi - lo) / B || 1;
  const binOf = v => Math.min(B - 1, Math.max(0, Math.floor((v - lo) / bw)));
  const hN = new Array(B).fill(0), hA = new Array(B).fill(0);
  for (const v of all) if (v != null) hA[binOf(v)]++;
  for (const k of idx) { const v = all[k]; if (v != null) hN[binOf(v)]++; }
  const nN = idx.length, nA = all.length;
  const dN = hN.map(v => v / nN), dA = hA.map(v => v / nA);
  body.insertAdjacentHTML("beforeend",
    `<div class="key"><span><i style="background:${css("--series-1")}"></i>This neuron (${int(nN)})</span>` +
    `<span><i style="background:none;border-top:2px solid ${css("--muted")};height:0;border-radius:0"></i>All ${splitName(M.detail_split)} samples</span></div>`);
  const c = chart(body);
  const ymax = Math.max(...dN, ...dA), ticks = niceTicks(0, ymax, 4, true);
  const y = lin(0, ticks[ticks.length - 1] || ymax, c.m.t + c.ih, c.m.t);
  const x = lin(lo, hi, c.m.l, c.m.l + c.iw);
  yAxis(c, y, ticks, v => pct(v, 0));
  const pw = c.iw / B;
  dN.forEach((v, b) => {
    if (v > 0) el("path", { d: barPath(c.m.l + b * pw + 1, y(v), pw - 2, c.m.t + c.ih - y(v), 3), fill: css("--series-1") }, c.s);
  });
  let dStr = "";
  dA.forEach((v, b) => { const x0 = c.m.l + b * pw, yy = y(v); dStr += (b ? "L" : "M") + x0 + "," + yy + "H" + (x0 + pw); });
  el("path", { d: dStr, fill: "none", stroke: css("--muted"), "stroke-width": 2 }, c.s);
  const ax = el("g", { class: "axis" }, c.s);
  for (const t of niceTicks(lo, hi, 5)) {
    const tx = el("text", { x: x(t), y: c.m.t + c.ih + 15, "text-anchor": "middle" }, ax); tx.textContent = fmt(t, 1);
  }
  dN.forEach((v, b) => {
    const r = el("rect", { x: c.m.l + b * pw, y: c.m.t, width: pw, height: c.ih, fill: "transparent" }, c.s);
    r.addEventListener("mousemove", e => showTip(e, `${fmt(lo + b * bw, 2)} – ${fmt(lo + (b + 1) * bw, 2)}`,
      [["this neuron", `${hN[b]} (${pct(v)})`], ["all samples", pct(dA[b])]]));
    r.addEventListener("mouseleave", hideTip);
  });
  xLabel(c, `${featName(f)} (scaled, −1…1)`);
}

// -- Box / Violin (NNSOM 'box', 'violin') ----------------------------------
function quantiles(a) {
  const s = a.filter(v => v != null).sort((p, q) => p - q), n = s.length;
  const q = p => { const h = (n - 1) * p, lo = Math.floor(h); return s[lo] + (s[Math.min(n - 1, lo + 1)] - s[lo]) * (h - lo); };
  const q1 = q(0.25), q3 = q(0.75), iqr = q3 - q1;
  const wl = s.find(v => v >= q1 - 1.5 * iqr), wh = [...s].reverse().find(v => v <= q3 + 1.5 * iqr);
  return { n, min: s[0], max: s[n - 1], q1, med: q(0.5), q3, wl, wh, s };
}
function commonRange() {
  let lo = Infinity, hi = -Infinity;
  for (const f of topF) { const [a, b] = extent(P0.feats[String(f)]); lo = Math.min(lo, a); hi = Math.max(hi, b); }
  return [lo, hi];
}
function tabBox(body, i, idx) {
  const [lo, hi] = commonRange();
  body.insertAdjacentHTML("beforeend",
    `<div class="key"><span><i style="background:${css("--series-1")}"></i>This neuron</span>` +
    `<span><i style="background:${css("--muted")};width:3px"></i>Median of all samples</span></div>`);
  const c = chart(body, 460, 44 + topF.length * 34, { t: 8, r: 16, b: 30, l: 64 });
  const x = lin(lo, hi, c.m.l, c.m.l + c.iw);
  const ax = el("g", { class: "axis" }, c.s);
  for (const t of niceTicks(lo, hi, 6)) {
    el("line", { class: "gridline", x1: x(t), x2: x(t), y1: c.m.t, y2: c.m.t + c.ih }, ax);
    const tx = el("text", { x: x(t), y: c.m.t + c.ih + 15, "text-anchor": "middle" }, ax); tx.textContent = fmt(t, 1);
  }
  const rowH = c.ih / topF.length;
  topF.forEach((f, r) => {
    const vals = idx.map(k => P0.feats[String(f)][k]), q = quantiles(vals), qa = quantiles(P0.feats[String(f)]);
    const cy = c.m.t + rowH * r + rowH / 2, bh = Math.min(16, rowH * 0.55);
    const lt = el("text", { class: "lbl", x: c.m.l - 8, y: cy + 4, "text-anchor": "end" }, c.s); lt.textContent = featName(f);
    el("line", { x1: x(q.wl), x2: x(q.wh), y1: cy, y2: cy, stroke: css("--series-1"), "stroke-width": 1.5 }, c.s);
    el("rect", { x: x(q.q1), y: cy - bh / 2, width: Math.max(1.5, x(q.q3) - x(q.q1)), height: bh, rx: 3,
                 fill: "#b7d3f6", stroke: css("--series-1"), "stroke-width": 1.5 }, c.s);
    el("line", { x1: x(q.med), x2: x(q.med), y1: cy - bh / 2, y2: cy + bh / 2, stroke: "#0d366b", "stroke-width": 2 }, c.s);
    el("line", { x1: x(qa.med), x2: x(qa.med), y1: cy - bh / 2 - 4, y2: cy + bh / 2 + 4, stroke: css("--muted"), "stroke-width": 2, "stroke-dasharray": "2 2" }, c.s);
    const hit = el("rect", { x: c.m.l, y: cy - rowH / 2, width: c.iw, height: rowH, fill: "transparent" }, c.s);
    hit.addEventListener("mousemove", e => showTip(e, featName(f), [["n", int(q.n)], ["min / max", `${fmt(q.min)} / ${fmt(q.max)}`],
      ["Q1 / median / Q3", `${fmt(q.q1)} / ${fmt(q.med)} / ${fmt(q.q3)}`], ["median of all samples", fmt(qa.med)]]));
    hit.addEventListener("mouseleave", hideTip);
  });
  xLabel(c, "scaled value, −1…1 (top-variance fc2 dimensions)");
}
function kde(vals, lo, hi, n = 48) {
  const s = vals.filter(v => v != null), m = s.length;
  const mean = s.reduce((a, b) => a + b, 0) / m;
  const sd = Math.sqrt(s.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, m - 1)) || (hi - lo) / 20 || 1;
  const bw = 1.06 * sd * Math.pow(m, -0.2);
  const pts = [];
  for (let k = 0; k <= n; k++) {
    const x = lo + (hi - lo) * k / n; let d = 0;
    for (const v of s) d += Math.exp(-0.5 * ((x - v) / bw) ** 2);
    pts.push([x, d / (m * bw * Math.sqrt(2 * Math.PI))]);
  }
  return pts;
}
function tabViolin(body, i, idx) {
  const [lo, hi] = commonRange();
  const c = chart(body, 460, 270, { t: 10, r: 12, b: 38, l: 44 });
  const y = lin(lo, hi, c.m.t + c.ih, c.m.t);
  yAxis(c, y, niceTicks(lo, hi, 5), v => fmt(v, 1));
  const colW = c.iw / topF.length;
  const dens = topF.map(f => {
    const vals = idx.map(k => P0.feats[String(f)][k]);
    const [a, b] = extent(vals);
    return { f, vals, pts: vals.length > 1 && b > a ? kde(vals, a, b) : null, q: quantiles(vals) };
  });
  const dmax = Math.max(...dens.flatMap(d => d.pts ? d.pts.map(p => p[1]) : [0])) || 1;
  dens.forEach((d, k) => {
    const cx = c.m.l + colW * k + colW / 2, half = colW * 0.42;
    if (d.pts) {
      const L = d.pts.map(([v, p]) => `${cx - p / dmax * half},${y(v)}`), R = d.pts.map(([v, p]) => `${cx + p / dmax * half},${y(v)}`).reverse();
      el("path", { d: "M" + L.join("L") + "L" + R.join("L") + "Z", fill: "#b7d3f6", stroke: css("--series-1"), "stroke-width": 1.5 }, c.s);
    } else {
      for (const v of d.vals) el("circle", { cx, cy: y(v), r: 4, fill: css("--series-1") }, c.s);
    }
    el("line", { x1: cx - 6, x2: cx + 6, y1: y(d.q.med), y2: y(d.q.med), stroke: "#0d366b", "stroke-width": 2 }, c.s);
    const t = el("text", { class: "lbl", x: cx, y: c.m.t + c.ih + 15, "text-anchor": "middle" }, c.s); t.textContent = featName(d.f);
    const hit = el("rect", { x: cx - colW / 2, y: c.m.t, width: colW, height: c.ih, fill: "transparent" }, c.s);
    hit.addEventListener("mousemove", e => showTip(e, featName(d.f), [["n", int(d.q.n)], ["median", fmt(d.q.med)], ["range", `${fmt(d.q.min)} – ${fmt(d.q.max)}`]]));
    hit.addEventListener("mouseleave", hideTip);
  });
  const note = document.createElement("div"); note.className = "note";
  note.textContent = "Width = density of the neuron's samples at that value (scaled, −1…1); dark tick = median.";
  body.appendChild(note);
}

// -- Scatter (NNSOM 'scatter') ---------------------------------------------
function tabScatter(body, i, idx) {
  const ctl = featSelect(body, "x", "sx");
  const ys = document.createElement("span"); ys.textContent = "y"; ctl.appendChild(ys);
  const sel = document.createElement("select");
  for (const f of topF) sel.add(new Option(featName(f), f));
  sel.value = String(detailFeat.sy);
  sel.addEventListener("change", () => { detailFeat.sy = Number(sel.value); drawDetail(); });
  ctl.appendChild(sel);
  const nWrong = idx.filter(k => P0.label[k] !== P0.pred[k]).length;
  body.insertAdjacentHTML("beforeend",
    `<div class="key"><span><i style="background:#d4d3cd"></i>All ${splitName(M.detail_split)} samples</span>` +
    `<span><i style="background:${css("--series-1")}"></i>This neuron, correct</span>` +
    `<span><i style="background:${css("--critical")}"></i>This neuron, misclassified (${nWrong})</span></div>`);
  const W = 460, H = 300, m = { t: 10, r: 12, b: 34, l: 44 };
  const fx = P0.feats[String(detailFeat.sx)], fy = P0.feats[String(detailFeat.sy)];
  const [x0, x1] = extent(fx), [y0, y1] = extent(fy);
  const c = chart(body, W, H, m);
  const x = lin(x0, x1, m.l, W - m.r), y = lin(y0, y1, H - m.b, m.t);
  yAxis(c, y, niceTicks(y0, y1, 5), v => fmt(v, 1));
  const ax = el("g", { class: "axis" }, c.s);
  for (const t of niceTicks(x0, x1, 5)) { const tx = el("text", { x: x(t), y: H - m.b + 15, "text-anchor": "middle" }, ax); tx.textContent = fmt(t, 1); }
  const fo = el("foreignObject", { x: 0, y: 0, width: W, height: H }, c.s);
  const cv = document.createElement("canvas"); const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr * 2; cv.height = H * dpr * 2; cv.style.width = W + "px"; cv.style.height = H + "px";
  fo.appendChild(cv);
  const g = cv.getContext("2d"); g.scale(dpr * 2, dpr * 2);
  g.fillStyle = "rgba(160,158,150,0.35)";
  for (let k = 0; k < fx.length; k++) if (fx[k] != null && fy[k] != null) g.fillRect(x(fx[k]) - 1, y(fy[k]) - 1, 2, 2);
  const mine = new Set(idx);
  const dots = el("g", {}, c.s);
  for (const k of idx) {
    const wrong = P0.label[k] !== P0.pred[k];
    const d = el("circle", { cx: x(fx[k]), cy: y(fy[k]), r: wrong ? 4.5 : 3.5,
      fill: wrong ? css("--critical") : css("--series-1"), stroke: css("--surface"), "stroke-width": 1.5 }, dots);
    d.addEventListener("mousemove", e => showTip(e, `Sample ${P0.sample_id[k]}`, [["true / predicted", `${P0.label[k]} / ${P0.pred[k]}`],
      ["confidence", fmt(P0.conf[k], 3)], [featName(detailFeat.sx), fmt(fx[k], 3)], [featName(detailFeat.sy), fmt(fy[k], 3)]]));
    d.addEventListener("mouseleave", hideTip);
  }
  xLabel(c, `${featName(detailFeat.sx)} (x) vs ${featName(detailFeat.sy)} (y)`);
}

// -- Weights (NNSOM plt_wgts) ----------------------------------------------
function tabWeights(body, i) {
  body.insertAdjacentHTML("beforeend",
    `<div class="key"><span><i style="background:${css("--series-1")}"></i>Neuron ${i} weights</span>` +
    `<span><i style="background:${css("--muted")}"></i>Mean over all neurons</span>` +
    `<span>★ = top-variance dimension</span></div>`);
  const w = D.w[i], K = w.length;
  const all = D.w.flat(), [lo, hi] = extent(all);
  const c = chart(body, 460, 250, { t: 12, r: 12, b: 34, l: 40 });
  const x = lin(0, K - 1, c.m.l, c.m.l + c.iw), y = lin(lo, hi, c.m.t + c.ih, c.m.t);
  yAxis(c, y, niceTicks(lo, hi, 5), v => fmt(v, 1));
  const ax = el("g", { class: "axis" }, c.s);
  for (const t of niceTicks(0, K - 1, 6)) { const tx = el("text", { x: x(t), y: c.m.t + c.ih + 15, "text-anchor": "middle" }, ax); tx.textContent = t; }
  for (const f of topF) { const t = el("text", { x: x(f), y: c.m.t + c.ih - 3, "text-anchor": "middle", "font-size": 9, fill: css("--muted") }, c.s); t.textContent = "★"; }
  const line = (arr, color, dash) => el("path", { d: arr.map((v, k) => (k ? "L" : "M") + x(k) + "," + y(v)).join(""),
    fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash || "none", "stroke-linejoin": "round" }, c.s);
  line(meanW, css("--muted"), "4 3"); line(w, css("--series-1"));
  const cross = el("line", { y1: c.m.t, y2: c.m.t + c.ih, stroke: css("--axis"), visibility: "hidden" }, c.s);
  const dot = el("circle", { r: 4, fill: css("--series-1"), stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }, c.s);
  const hit = el("rect", { x: c.m.l, y: c.m.t, width: c.iw, height: c.ih, fill: "transparent" }, c.s);
  hit.addEventListener("mousemove", e => {
    const r = c.s.getBoundingClientRect(), px = (e.clientX - r.left) * (c.w / r.width);
    const k = Math.max(0, Math.min(K - 1, Math.round((px - c.m.l) / c.iw * (K - 1))));
    cross.setAttribute("x1", x(k)); cross.setAttribute("x2", x(k)); cross.setAttribute("visibility", "visible");
    dot.setAttribute("cx", x(k)); dot.setAttribute("cy", y(w[k])); dot.setAttribute("visibility", "visible");
    showTip(e, `${featName(k)}${topF.includes(k) ? " ★" : ""}`, [["this neuron", fmt(w[k], 3)], ["mean of all neurons", fmt(meanW[k], 3)]]);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dot.setAttribute("visibility", "hidden"); });
  xLabel(c, "fc2 dimension (scaled weight space)");
}

// -- Errors ----------------------------------------------------------------
function tabErrors(body, i) {
  const ds = D.splits[M.detail_split], conf = ds.confusion[i];
  const n = ds.hits[i], e = ds.errors[i];
  let h = `<p class="desc">${int(e)} of ${int(n)} samples misclassified by the CNN (${pct(ds.error_rate[i])}); ` +
          `this neuron holds ${pct(ds.error_share[i])} of all ${splitName(M.detail_split)} errors.</p>`;
  if (!conf.length) { body.innerHTML = h + `<div class="empty">No misclassified samples in this neuron.</div>`; return; }
  h += `<table><thead><tr><th>True digit</th><th>Predicted as</th><th>Count</th><th>Share of neuron's errors</th></tr></thead><tbody>`;
  for (const [k, v] of conf) { const [t, p] = k.split(">"); h += `<tr><td>${t}</td><td>${p}</td><td>${v}</td><td>${pct(v / e)}</td></tr>`; }
  body.innerHTML = h + "</tbody></table>";
}

// -- Nearest samples (NNSOM 'topn') ----------------------------------------
function tabSamples(body, i, idx) {
  const rows = [...idx].sort((a, b) => (P0.dist[a] ?? 1e9) - (P0.dist[b] ?? 1e9)).slice(0, M.topn);
  let h = `<p class="desc">The ${rows.length} samples closest to this neuron's weight vector. Misclassified rows in red.</p>`;
  h += `<table><thead><tr><th>Sample ID</th><th>True</th><th>Predicted</th><th>Confidence</th><th>Distance</th></tr></thead><tbody>`;
  for (const k of rows) {
    const wrong = P0.label[k] !== P0.pred[k];
    h += `<tr class="${wrong ? "wrong" : ""}"><td>${P0.sample_id[k]}</td><td>${P0.label[k]}</td><td>${P0.pred[k]}${wrong ? " ✕" : ""}</td>` +
         `<td>${fmt(P0.conf[k], 3)}</td><td>${fmt(P0.dist[k], 3)}</td></tr>`;
  }
  body.innerHTML = h + "</tbody></table>";
}

