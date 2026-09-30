// 04_detail_panel.js - Click-to-inspect panel: selecting a neuron, stats tiles, tab routing.
// Fragment: all js/*.js files are concatenated (in numeric order) into ONE function scope
// by html_report.py, so they share variables. Do not add import/export.

// ---------------------------------------------------------------- detail
const P0 = D.points;
const byNeuron = Array.from({ length: N }, () => []);
if (P0) P0.bmu.forEach((b, k) => { if (b >= 0) byNeuron[b].push(k); });
const TABS = [
  ["classes", "Classes (pie / stem)"], ["hist", "Histogram"], ["box", "Box"], ["violin", "Violin"],
  ["scatter", "Scatter"], ["weights", "Weights"], ["errors", "Errors"], ["samples", "Nearest samples"],
];
const tabsEl = document.getElementById("d-tabs");
for (const [k, txt] of TABS) {
  const b = document.createElement("button");
  b.textContent = txt; b.dataset.tab = k; b.setAttribute("role", "tab");
  b.addEventListener("click", () => { state.tab = k; drawDetail(); });
  tabsEl.appendChild(b);
}
const detailFeat = { hist: topF[0], sx: topF[0], sy: topF[1] ?? topF[0] };

function select(i) { state.selected = i; drawMap(); drawDetail(); }

function drawDetail() {
  const i = state.selected, ds = D.splits[M.detail_split];
  document.querySelectorAll("#d-tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === state.tab)));
  document.getElementById("d-title").textContent = neuronTitle(i);
  document.getElementById("d-desc").textContent =
    `Detail panel uses the ${splitName(M.detail_split)} split (like NNSOM's onpick window). Click another neuron on the map to switch.`;
  const T = D.splits.train, V = D.splits.val;
  const stats = [
    ["Train samples", int(T.hits[i])],
    ["Validation samples", int(V.hits[i])],
    ["Dominant digit", ds.hits[i] ? ds.dominant[i] : "–"],
    ["Purity", pct(ds.purity[i])],
    ["Error rate", ds.hits[i] ? `${pct(ds.error_rate[i])}` : "–"],
    ["Mean neighbour dist.", fmt(meanNbr[i], 3)],
  ];
  document.getElementById("d-stats").innerHTML =
    stats.map(([k, v]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}</div></div>`).join("");
  const body = document.getElementById("detail-body");
  body.innerHTML = "";
  const idx = byNeuron[i];
  if (state.tab !== "weights" && (!P0 || idx.length === 0)) {
    body.innerHTML = `<div class="empty">No ${splitName(M.detail_split)} samples map to this neuron. The Weights tab still shows what it learned.</div>`;
    return;
  }
  ({ classes: tabClasses, hist: tabHist, box: tabBox, violin: tabViolin, scatter: tabScatter,
     weights: tabWeights, errors: tabErrors, samples: tabSamples })[state.tab](body, i, idx);
}

