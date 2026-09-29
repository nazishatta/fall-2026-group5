// 03_map.js - Map drawing, tooltips, legend and the view/option/split controls.
// Fragment: all js/*.js files are concatenated (in numeric order) into ONE function scope
// by html_report.py, so they share variables. Do not add import/export.

// ---------------------------------------------------------------- map
const svg = document.getElementById("map");
(function setupMap() {
  const xs = D.pos[0], ys = D.pos[1].map(v => -v);
  const pad = 0.75;
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  svg.setAttribute("viewBox", `${x0} ${y0} ${x1 - x0} ${y1 - y0}`);
})();

function valueFor(i) {
  const s = S();
  switch (state.view) {
    case "digit": return s.hits[i] ? s.cat[i][state.opt] / s.hits[i] : null;
    case "dominant": case "purity": return s.purity[i];
    case "component": return D.w[i][state.opt];
    case "featmean": return s.feat_mean[i][state.opt];
    case "error": case "complex": return s.error_rate[i];
    case "umatrix": return meanNbr[i];
    default: return null;
  }
}

function scaleFor() {
  const s = S(), v = state.view;
  if (v === "component") {
    const vals = D.w.map(r => r[state.opt]);
    const m = Math.max(...vals.map(Math.abs)) || 1;
    return { stops: DIVERGING, lo: -m, hi: m, f: x => (x + m) / (2*m), cap: `weight of ${featName(state.opt)}`, d: 3 };
  }
  if (v === "digit") return { stops: SEQ_BLUE, lo: 0, hi: 1, f: x => x, cap: `share of digit ${state.opt}`, pct: true };
  if (v === "dominant" || v === "purity") {
    const [lo] = extent(s.purity);
    return { stops: SEQ_BLUE, lo, hi: 1, f: x => (x - lo) / ((1 - lo) || 1), cap: "purity", pct: true };
  }
  if (v === "featmean") {
    const [lo, hi] = extent(s.feat_mean.map(r => r[state.opt]));
    return { stops: SEQ_BLUE, lo, hi, f: x => (x - lo) / ((hi - lo) || 1), cap: `mean ${featName(state.opt)}`, d: 2 };
  }
  if (v === "error" || v === "complex") {
    const [, hi] = extent(s.error_rate);
    const top = hi || 1;
    return { stops: SEQ_ORANGE, lo: 0, hi: top, f: x => x / top, cap: "CNN error rate", pct: true };
  }
  if (v === "umatrix") {
    const [lo, hi] = edgeExt;
    return { stops: SEQ_BLUE, lo, hi, f: x => (x - lo) / ((hi - lo) || 1), cap: "weight distance between neighbours", d: 3 };
  }
  if (v === "hits") return { stops: SEQ_BLUE, lo: 0, hi: Math.max(...s.hits), f: () => 0.62, cap: null };
  return null;
}

function drawMap() {
  svg.textContent = "";
  const s = S(), v = state.view, sc = scaleFor();
  const maxHits = Math.max(...s.hits) || 1;
  const font = 0.3;

  if (v === "umatrix") {
    const g = el("g", {}, svg);
    for (const [i, j, d] of D.edges) {
      const p = el("polygon", { points: edgePts(i, j), fill: ramp(sc.stops, sc.f(d)) }, g);
      p.addEventListener("mousemove", e => showTip(e, `Link ${i} ↔ ${j}`, [["weight distance", fmt(d, 3)]]));
      p.addEventListener("mouseleave", hideTip);
    }
  }

  for (let i = 0; i < N; i++) {
    const g = el("g", { class: "hex", "data-i": i }, svg);
    const val = valueFor(i);
    const hits = s.hits[i];
    let faceFill = css("--surface-2"), faceScale = 1, inner = null, label = null, stroke = css("--axis"), sw = 0.6;

    if (v === "hits") {
      faceFill = css("--surface");
      if (hits) inner = { s: Math.max(0.12, Math.sqrt(hits / maxHits)) * 0.92, fill: SEQ_BLUE[4] };
      label = hits ? String(hits) : null;
    } else if (v === "umatrix") {
      faceScale = 0.32; faceFill = "#6e6d68"; stroke = "#6e6d68";
    } else if (v === "topology") {
      label = String(i); faceFill = css("--surface-2");
    } else if (v === "error" || v === "purity") {
      faceFill = css("--surface");
      if (hits) inner = { s: Math.max(0.2, Math.sqrt(hits / maxHits)) * 0.95, fill: ramp(sc.stops, sc.f(val)) };
    } else {
      faceFill = val == null ? "url(#nodata)" : ramp(sc.stops, sc.f(val));
      if (v === "dominant" && hits) label = String(s.dominant[i]);
      if (v === "complex" && hits) {
        label = String(s.dominant[i]) + (s.errors[i] ? "→" + s.dominant_wrong[i] : "");
        if (s.errors[i]) { stroke = "#0b0b0b"; sw = 0.6 + 3.2 * (s.error_rate[i] / (sc.hi || 1)); }
      }
    }

    el("polygon", { class: "face", points: hexPts(i, faceScale), fill: faceFill, stroke, "stroke-width": sw,
                    "vector-effect": "non-scaling-stroke" }, g);
    if (inner) el("polygon", { points: hexPts(i, inner.s), fill: inner.fill, "pointer-events": "none" }, g);
    if (label) {
      const [x, y] = P(i);
      let fill = "#0b0b0b";
      if (v === "hits") fill = inner && inner.s > 0.5 ? "#ffffff" : "#0b0b0b";
      else if (faceFill && faceFill.startsWith("#")) fill = inkOn(faceFill);
      const size = v === "topology" ? font * 0.95 : (label.length >= 4 ? font * 0.78 : font);
      const t = el("text", { x, y: y + size * 0.35, "text-anchor": "middle", "font-size": size,
                              "font-weight": 600, fill, "pointer-events": "none" }, g);
      // Dark text over a small inner hexagon gets a white halo so it stays
      // readable across the blue/white boundary.
      if (fill === "#0b0b0b" && (v === "hits" || v === "error" || v === "purity")) {
        t.setAttribute("stroke", "#ffffff"); t.setAttribute("stroke-width", "3px");
        t.setAttribute("paint-order", "stroke"); t.setAttribute("stroke-linejoin", "round");
        t.setAttribute("vector-effect", "non-scaling-stroke");
      }
      t.textContent = label;
    }
    if (state.selected === i) {
      el("polygon", { points: hexPts(i, 1.02), fill: "none", stroke: css("--focus"), "stroke-width": 3,
                      "vector-effect": "non-scaling-stroke", "pointer-events": "none" }, g);
    }
    g.addEventListener("mousemove", e => showTip(e, neuronTitle(i), tipRows(i)));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => select(i));
  }

  const defs = el("defs", {}, svg);
  const pat = el("pattern", { id: "nodata", width: 0.12, height: 0.12, patternUnits: "userSpaceOnUse",
                              patternTransform: "rotate(45)" }, defs);
  el("rect", { width: 0.12, height: 0.12, fill: css("--surface") }, pat);
  el("rect", { width: 0.05, height: 0.12, fill: css("--nodata") }, pat);

  drawLegend(sc);
}

function neuronTitle(i) { const [r, c] = rowCol(i); return `Neuron ${i} · row ${r}, col ${c}`; }

function tipRows(i) {
  const s = S(), v = state.view, rows = [];
  rows.push([`${splitName(state.split)} samples`, int(s.hits[i])]);
  if (v === "digit") rows.push([`digit ${state.opt}`, `${int(s.cat[i][state.opt])} (${pct(valueFor(i))})`]);
  if (v === "dominant" || v === "purity") { rows.push(["dominant digit", s.hits[i] ? s.dominant[i] : "–"]); rows.push(["purity", pct(s.purity[i])]); }
  if (v === "component") rows.push([`weight ${featName(state.opt)}`, fmt(D.w[i][state.opt], 3)]);
  if (v === "featmean") rows.push([`mean ${featName(state.opt)}`, fmt(s.feat_mean[i][state.opt], 3)]);
  if (v === "error" || v === "complex") {
    rows.push(["misclassified", `${int(s.errors[i])} (${pct(s.error_rate[i])})`]);
    rows.push(["share of all errors", pct(s.error_share[i])]);
    if (v === "complex" && s.errors[i]) rows.push(["dominant true → wrong", `${s.dominant[i]} → ${s.dominant_wrong[i]}`]);
  }
  if (v === "umatrix") rows.push(["mean neighbour distance", fmt(meanNbr[i], 3)]);
  if (s.hits[i] === 0) rows.push(["", "empty neuron"]);
  rows.push(["", "click for details"]);
  return rows;
}

function drawLegend(sc) {
  const L = document.getElementById("legend");
  L.innerHTML = "";
  const s = S(), v = state.view;
  if (sc && sc.cap && v !== "hits") {
    const grad = `linear-gradient(90deg, ${sc.stops.join(",")})`;
    const f = x => sc.pct ? pct(x, 0) : fmt(x, sc.d ?? 2);
    L.insertAdjacentHTML("beforeend",
      `<span>${sc.cap}</span><span>${f(sc.lo)}</span><span class="bar" style="background:${grad}"></span><span>${f(sc.hi)}</span>`);
  }
  if (v === "hits" || v === "error" || v === "purity")
    L.insertAdjacentHTML("beforeend", `<span>Hexagon size = samples (max ${int(Math.max(...s.hits))})</span>`);
  if (v === "complex") L.insertAdjacentHTML("beforeend", `<span>Thicker dark border = higher error rate</span>`);
  const empty = s.hits.filter(h => h === 0).length;
  if (["digit","dominant","featmean","complex"].includes(v) && empty)
    L.insertAdjacentHTML("beforeend", `<span><span class="nodata-sw"></span> empty neuron (${empty})</span>`);
}

// ---------------------------------------------------------------- controls
const viewSel = document.getElementById("view"), optSel = document.getElementById("opt");
for (const k in VIEWS) viewSel.add(new Option(VIEWS[k].label, k));
viewSel.addEventListener("change", () => { state.view = viewSel.value; setupOpt(); render(); });
optSel.addEventListener("change", () => { state.opt = Number(optSel.value); render(); });
document.querySelectorAll("#split-seg button").forEach(b => b.addEventListener("click", () => {
  state.split = b.dataset.split; render();
}));
function setupOpt() {
  const V = VIEWS[state.view], wrap = document.getElementById("opt-wrap");
  if (!V.opt) { wrap.style.display = "none"; state.opt = null; return; }
  wrap.style.display = "";
  document.getElementById("opt-label").textContent = V.opt.label;
  optSel.innerHTML = "";
  for (const [val, txt] of V.opt.items()) optSel.add(new Option(txt, val));
  state.opt = V.opt.def; optSel.value = String(V.opt.def);
}
function render() {
  const V = VIEWS[state.view];
  document.getElementById("map-title").textContent = V.title();
  document.getElementById("map-desc").textContent = V.desc;
  document.getElementById("split-wrap").style.opacity = V.usesSplit ? 1 : 0.4;
  document.querySelectorAll("#split-seg button").forEach(b =>
    b.setAttribute("aria-pressed", String(b.dataset.split === state.split)));
  svg.setAttribute("aria-label", V.title());
  drawMap();
}

