// 02_views.js - Map view definitions (VIEWS) and hexagon geometry.
// Fragment: all js/*.js files are concatenated (in numeric order) into ONE function scope
// by html_report.py, so they share variables. Do not add import/export.

// ---------------------------------------------------------------- views
const S = () => D.splits[state.split];
const topF = M.top_features;
const VIEWS = {
  hits: {
    label: "Hit histogram",
    title: () => `Hit histogram — ${splitName(state.split)}`,
    desc: "Inner hexagon size = number of samples whose best-matching unit is this neuron.",
    usesSplit: true,
  },
  umatrix: {
    label: "Neuron distance (U-matrix)",
    title: () => "Neuron distance map (U-matrix)",
    desc: "Colour of the link between two neighbours = distance between their weight vectors. Dark bands mark cluster boundaries.",
  },
  digit: {
    label: "Digit share",
    title: () => `Share of digit ${state.opt} per neuron — ${splitName(state.split)}`,
    desc: "Colour = fraction of the neuron's samples whose true label is this digit.",
    usesSplit: true,
    opt: { label: "Digit", items: () => [...Array(M.num_classes).keys()].map(d => [d, `Digit ${d}`]), def: 0 },
  },
  dominant: {
    label: "Dominant digit & purity",
    title: () => `Dominant digit and purity — ${splitName(state.split)}`,
    desc: "Number = most common true digit in the neuron. Colour = purity (share of that digit).",
    usesSplit: true,
  },
  component: {
    label: "Component plane (weights)",
    title: () => `Component plane — ${featName(state.opt)}`,
    desc: "Colour = the neuron's learned weight for this fc2 dimension (scaled space, grey = 0).",
    opt: { label: "fc2 dimension", items: () => featureItems(), def: topF[0] },
  },
  featmean: {
    label: "Feature average",
    title: () => `Average ${featName(state.opt)} per neuron — ${splitName(state.split)}`,
    desc: "Colour = mean value of this fc2 dimension across the neuron's samples (SOM input scale, −1…1 — same scale as the component plane).",
    usesSplit: true,
    opt: { label: "fc2 dimension", items: () => featureItems(), def: topF[0] },
  },
  error: {
    label: "Error rate & support",
    title: () => `Error rate and support — ${splitName(state.split)}`,
    desc: "Colour = CNN error rate among the neuron's samples. Hexagon size = number of samples.",
    usesSplit: true,
  },
  purity: {
    label: "Purity & support",
    title: () => `Class purity and support — ${splitName(state.split)}`,
    desc: "Colour = share of the dominant digit. Hexagon size = number of samples.",
    usesSplit: true,
  },
  complex: {
    label: "Complex error map",
    title: () => `Complex error map — ${splitName(state.split)}`,
    desc: "Label = dominant true digit → most common wrong prediction. Colour and border width = error rate.",
    usesSplit: true,
  },
  topology: {
    label: "Topology (numbered)",
    title: () => "SOM topology with neuron IDs",
    desc: "Neuron index used throughout this report. Row 0 is the bottom row.",
  },
};
function featureItems() {
  const top = topF.map(f => [f, `★ ${featName(f)} (top variance)`]);
  const rest = [...Array(M.feature_dim).keys()].filter(f => !topF.includes(f)).map(f => [f, featName(f)]);
  return [...top, ...rest];
}

// ---------------------------------------------------------------- geometry
const HX = [-0.5, 0, 0.5, 0.5, 0, -0.5];
const HY = [1, 2, 1, -1, -2, -1].map(v => v * Math.sqrt(0.75) / 3);
const P = i => [D.pos[0][i], -D.pos[1][i]];                // flip y for SVG
function hexPts(i, s = 1) {
  const [x, y] = P(i);
  return HX.map((dx, k) => `${(x + dx*s).toFixed(4)},${(y - HY[k]*s).toFixed(4)}`).join(" ");
}
function edgePts(i, j) {
  const [x1, y1] = P(i), [x2, y2] = P(j);
  const mx = (x1 + x2)/2, my = (y1 + y2)/2, a = Math.atan2(y2 - y1, x2 - x1);
  const z = Math.sqrt(0.75) / 3;
  const ex = [-0.5, 0, 0.5, 0], ey = [0, z, 0, -z];
  return ex.map((dx, k) => {
    const rx = dx*Math.cos(a) - ey[k]*Math.sin(a), ry = dx*Math.sin(a) + ey[k]*Math.cos(a);
    return `${(mx + rx).toFixed(4)},${(my + ry).toFixed(4)}`;
  }).join(" ");
}

