// main.js - Page header text and first render.
// Fragment: html_report.py joins the js files into ONE function scope, in the order listed in
// settings.JS_PARTS and js/README.md, so they share variables. Do not add import/export.

// ---------------------------------------------------------------- header
document.getElementById("title").textContent = `SOM explorer — ${M.model}`;
const T = D.splits.train, V = D.splits.val;
const emptyV = V ? V.hits.filter(h => !h).length : 0;
document.getElementById("summary").innerHTML =
  `<b>${M.grid[0]}×${M.grid[1]}</b> grid (${N} neurons) · <b>${M.feature_dim}</b>-d ${PX} embeddings · ` +
  (T ? `train <b>${int(T.n)}</b>` : "") + (V ? ` · validation <b>${int(V.n)}</b> (CNN accuracy ${pct(V.accuracy, 2)}, ${emptyV} empty neurons)` : "") +
  ` · top-variance dims: <b>${topF.map(featName).join(", ")}</b> · hover any neuron, click for details.`;

// ---------------------------------------------------------------- init
viewSel.value = state.view; setupOpt(); render();
const ds0 = D.splits[M.detail_split];
select(ds0.hits.indexOf(Math.max(...ds0.hits)));
