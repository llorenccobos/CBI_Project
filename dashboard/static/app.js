/* Flood Risk Monitor — front-end application (MapLibre GL + Plotly, no build step). */
"use strict";

const API = "/api";
const TZ = "Europe/Madrid";
const RISKS = ["normal", "elevated", "dangerous", "critical"];
const RISK_LABEL = { normal: "Normal", elevated: "Elevated", dangerous: "Dangerous", critical: "Critical", unknown: "No data" };
// Free vector basemaps (no API key): OpenFreeMap, data © OpenStreetMap contributors
const BASEMAPS = {
  light: "https://tiles.openfreemap.org/styles/positron",
  dark: "https://tiles.openfreemap.org/styles/dark",
};
const CUSTOM_SOURCES = ["weather", "bus", "metro", "segments", "stations", "stops", "sensors", "vehicles"];

const state = {
  meta: null, fc: null, wx: null, segments: null, vehicles: [],
  hIdx: 0, steps: [0], metric: "level", mode: "live", issue: null,
  infra: new Set(["metro", "street"]), risks: new Set(RISKS),
  weatherVar: "", selected: null, playing: null, theme: "light",
};

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const fetchJSON = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json();
};
const parseT = (s) => new Date(s.endsWith("Z") || s.includes("+") ? s : s + "Z");
const fmtT = (s, opts = {}) => parseT(s).toLocaleString("en-GB", {
  timeZone: TZ, weekday: "short", day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", ...opts,
});
const fmtCm = (v) => (v === null || v === undefined ? "–" : `${v.toFixed(1)} cm`);
const pct = (p) => (p === null || p === undefined ? "–" : p < 0.01 ? "<1 %" : `${Math.round(p * 100)} %`);
const color = (risk) => (state.meta ? state.meta.risk_colors[risk] : "#999");
const curH = () => state.steps[state.hIdx];
const qs = () => (state.mode === "replay" && state.issue ? `?issue=${encodeURIComponent(state.issue)}` : "");
function toast(msg, ms = 6000) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}
function loading(on) { $("loading").classList.toggle("hidden", !on); }

// value / risk for a sensor at the current horizon and metric
function sensorView(s, h = curH()) {
  if (h === 0) {
    const lv = s.current.level;
    return { value: lv, lo: lv, hi: lv, risk: s.current.risk, p: null, conf: null, tier: "observed", hz: null };
  }
  const hz = s.horizons.find((x) => x.h === h);
  const al = state.fc.alert_level;
  let value, risk;
  if (state.metric === "max") { value = hz.max.median; risk = hz.risk_max; }
  else if (state.metric === "upper") { value = hz.level.hi; risk = hz.risk_upper; }
  else { value = hz.level.median; risk = hz.risk; }
  return { value, lo: hz.level.lo, hi: hz.level.hi, risk, p: hz.p_exceed_max[al], conf: hz.confidence,
           tier: hz.tier, hz, alarm: hz.alarm };
}

// ------------------------------------------------------------------ map
let map;
function initMap() {
  map = new maplibregl.Map({
    container: "map",
    style: BASEMAPS.light,
    center: state.meta.area.center, zoom: 12.4, minZoom: 10.5, maxZoom: 17.5,
    maxBounds: [[1.95, 41.25], [2.38, 41.55]], attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");
  return new Promise((res) => map.on("load", res));
}

function setBasemap(theme) {
  // swap the basemap style while keeping all operational sources and layers
  map.setStyle(BASEMAPS[theme], {
    transformStyle: (prev, next) => {
      if (!prev) return next;
      const sources = { ...next.sources };
      for (const k of CUSTOM_SOURCES) if (prev.sources[k]) sources[k] = prev.sources[k];
      const custom = prev.layers.filter((l) => CUSTOM_SOURCES.includes(l.source));
      return { ...next, sources, layers: [...next.layers, ...custom] };
    },
  });
  map.once("styledata", () => render());
}

async function addLayers() {
  const [metro, stations, bus, stops, segs, grid, veh] = await Promise.all(
    ["metro_lines", "metro_stations", "bus_routes", "bus_stops", "segments", "weather_grid", "vehicles"]
      .map((n) => fetchJSON(`${API}/layers/${n}`)));
  const defaults = { _color: "#9AA0A6", _halo: 10, _vis: 1, _risk: "unknown" };
  segs.features.forEach((f) => Object.assign(f.properties, defaults));
  state.segments = segs;
  state.vehicles = prepareVehicles(veh);
  map.addSource("weather", { type: "geojson", data: grid });
  map.addLayer({ id: "weather-fill", type: "fill", source: "weather",
    paint: { "fill-color": "#888", "fill-opacity": 0 }, layout: { visibility: "none" } });
  map.addLayer({ id: "weather-line", type: "line", source: "weather",
    paint: { "line-color": "#5b7083", "line-width": 0.5, "line-opacity": 0.35 }, layout: { visibility: "none" } });

  map.addSource("bus", { type: "geojson", data: bus });
  map.addLayer({ id: "bus-lines", type: "line", source: "bus",
    paint: { "line-color": ["get", "colour"], "line-width": ["interpolate", ["linear"], ["zoom"], 11, 0.8, 15, 2.2],
             "line-opacity": 0.42 }, layout: { "line-cap": "round", "line-join": "round" } });
  map.addSource("metro", { type: "geojson", data: metro });
  map.addLayer({ id: "metro-casing", type: "line", source: "metro",
    paint: { "line-color": "#ffffff", "line-width": ["interpolate", ["linear"], ["zoom"], 11, 3.5, 15, 7] },
    layout: { "line-cap": "round", "line-join": "round" } });
  map.addLayer({ id: "metro-lines", type: "line", source: "metro",
    paint: { "line-color": ["get", "colour"], "line-width": ["interpolate", ["linear"], ["zoom"], 11, 2, 15, 4.5],
             "line-opacity": 0.9 }, layout: { "line-cap": "round", "line-join": "round" } });

  // monitored segments: uncertainty halo + risk-coloured line
  map.addSource("segments", { type: "geojson", data: segs, promoteId: "sensor_id" });
  map.addLayer({ id: "seg-halo", type: "line", source: "segments",
    paint: { "line-color": ["get", "_color"], "line-width": ["get", "_halo"], "line-blur": 6, "line-opacity": 0.38 },
    layout: { "line-cap": "round", "line-join": "round" } });
  map.addLayer({ id: "seg-casing", type: "line", source: "segments",
    paint: { "line-color": "#1b1f24", "line-width": ["interpolate", ["linear"], ["zoom"], 11, 4.5, 16, 11], "line-opacity": 0.7 },
    layout: { "line-cap": "round", "line-join": "round" } });
  map.addLayer({ id: "seg-line", type: "line", source: "segments",
    paint: { "line-color": ["get", "_color"], "line-width": ["interpolate", ["linear"], ["zoom"], 11, 2.8, 16, 8] },
    layout: { "line-cap": "round", "line-join": "round" } });

  map.addSource("stations", { type: "geojson", data: stations });
  map.addLayer({ id: "stations", type: "circle", source: "stations", minzoom: 12,
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 12, 2.5, 16, 6], "circle-color": "#fff",
             "circle-stroke-color": "#1b1f24", "circle-stroke-width": 1.4 } });
  map.addSource("stops", { type: "geojson", data: stops });
  map.addLayer({ id: "stops", type: "circle", source: "stops", minzoom: 14.5,
    paint: { "circle-radius": 2.6, "circle-color": "#e2001a", "circle-stroke-color": "#fff", "circle-stroke-width": 0.8 },
    layout: { visibility: "none" } });

  // sensor markers (segment mid points)
  const pts = { type: "FeatureCollection", features: segs.features.map((f) => {
    const c = f.geometry.coordinates; const m = c[Math.floor(c.length / 2)];
    return { type: "Feature", geometry: { type: "Point", coordinates: m }, properties: { ...f.properties } };
  }) };
  state.sensorPts = pts;
  map.addSource("sensors", { type: "geojson", data: pts, promoteId: "sensor_id" });
  map.addLayer({ id: "sensor-pulse", type: "circle", source: "sensors",
    paint: { "circle-radius": 14, "circle-color": ["get", "_color"], "circle-opacity": 0,
             "circle-stroke-color": ["get", "_color"], "circle-stroke-width": 2, "circle-stroke-opacity": 0 } });
  map.addLayer({ id: "sensor-pts", type: "circle", source: "sensors",
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 5, 16, 9], "circle-color": ["get", "_color"],
             "circle-stroke-color": "#fff", "circle-stroke-width": 2 } });

  // simulated vehicles
  map.addSource("vehicles", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
  map.addLayer({ id: "veh-bus", type: "circle", source: "vehicles", filter: ["==", ["get", "kind"], "bus"],
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 2.4, 16, 5], "circle-color": ["get", "colour"],
             "circle-stroke-color": "#fff", "circle-stroke-width": 1 } });
  map.addLayer({ id: "veh-metro", type: "circle", source: "vehicles", filter: ["==", ["get", "kind"], "metro"],
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 11, 3.6, 16, 7.5], "circle-color": ["get", "colour"],
             "circle-stroke-color": "#111", "circle-stroke-width": 1.5 } });

  // interactions
  for (const id of ["seg-line", "seg-halo", "sensor-pts"]) {
    map.on("click", id, (e) => openDetail(e.features[0].properties.sensor_id));
    map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
  }
  const pop = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 10 });
  map.on("mousemove", "sensor-pts", (e) => {
    const s = state.fc.sensors.find((x) => x.sensor_id === e.features[0].properties.sensor_id);
    if (!s) return;
    const v = sensorView(s);
    pop.setLngLat(e.lngLat).setHTML(`<b>${s.name}</b><br>${s.sensor_id} · ${s.infra_type}<br>
      ${curH() === 0 ? "Measured" : "Forecast +" + curH() + " h"}: <b>${fmtCm(v.value)}</b>
      ${v.p !== null ? `<br>P(> ${s.thresholds[state.fc.alert_level]} cm in window): <b>${pct(v.p)}</b>` : ""}`).addTo(map);
  });
  map.on("mouseleave", "sensor-pts", () => pop.remove());
  map.on("mousemove", "stations", (e) => {
    const p = e.features[0].properties;
    pop.setLngLat(e.lngLat).setHTML(`<b>${p.name}</b><br>Metro ${JSON.parse(p.lines).join(", ")}`).addTo(map);
  });
  map.on("mouseleave", "stations", () => pop.remove());
  map.on("mousemove", "veh-metro", (e) => {
    const p = e.features[0].properties;
    pop.setLngLat(e.lngLat).setHTML(`<b>Metro ${p.line}</b> · SIMULATED position`).addTo(map);
  });
  map.on("mouseleave", "veh-metro", () => pop.remove());
  map.on("mousemove", "veh-bus", (e) => {
    const p = e.features[0].properties;
    pop.setLngLat(e.lngLat).setHTML(`<b>Bus ${p.line}</b> · SIMULATED position`).addTo(map);
  });
  map.on("mouseleave", "veh-bus", () => pop.remove());
}

// ------------------------------------------------------------------ rendering
function render() {
  if (!state.fc || !state.segments) return;
  const h = curH();
  const byId = Object.fromEntries(state.fc.sensors.map((s) => [s.sensor_id, s]));
  const al = state.fc.alert_level;
  const decorate = (f) => {
    const s = byId[f.properties.sensor_id];
    const v = s ? sensorView(s) : { risk: "unknown", value: null, lo: null, hi: null };
    const thr = s ? s.thresholds[al] : 30;
    const width = v.hi !== null && v.lo !== null ? Math.min(1.5, (v.hi - v.lo) / thr) : 0;
    const visible = state.infra.has(f.properties.infra_type) && state.risks.has(v.risk === "unknown" ? "normal" : v.risk);
    return { ...f, properties: { ...f.properties, _color: color(v.risk), _risk: v.risk, _halo: 10 + 26 * width,
                                 _alarm: v.alarm ? 1 : 0, _vis: visible ? 1 : 0 } };
  };
  const segs = { ...state.segments, features: state.segments.features.map(decorate) };
  map.getSource("segments").setData(segs);
  map.getSource("sensors").setData({ ...state.sensorPts, features: state.sensorPts.features.map(decorate) });
  const filt = ["==", ["get", "_vis"], 1];
  for (const id of ["seg-halo", "seg-casing", "seg-line", "sensor-pts", "sensor-pulse"]) map.setFilter(id, filt);
  const aw = h > state.meta.operational_max_h;
  map.setPaintProperty("seg-line", "line-dasharray", aw ? [1.2, 0.8] : [1, 0]);
  map.setPaintProperty("seg-halo", "line-opacity", h === 0 ? 0 : aw ? 0.22 : 0.38);
  renderTime(); renderSummary(); renderAlerts(); renderWeather(); renderLegend();
  if (state.selected) renderDetail();
}

function renderTime() {
  const h = curH();
  const t0 = state.fc.issued_at;
  const vt = new Date(parseT(t0).getTime() + h * 3600e3).toISOString();
  $("valid-time").textContent = fmtT(vt);
  $("valid-tier").textContent = h === 0 ? "Latest measurement" :
    h > state.meta.operational_max_h ? `+${h} h · awareness (low confidence)` : `+${h} h · operational forecast`;
  $("sum-horizon").textContent = h === 0 ? "now" : `+${h} h`;
  [...$("ticks").children].forEach((el, i) => el.classList.toggle("on", i === state.hIdx));
}

function renderSummary() {
  const counts = { normal: 0, elevated: 0, dangerous: 0, critical: 0 };
  let alarms = 0;
  for (const s of state.fc.sensors) {
    if (!state.infra.has(s.infra_type)) continue;
    const v = sensorView(s);
    counts[v.risk === "unknown" ? "normal" : v.risk]++;
    if (v.alarm) alarms++;
  }
  $("risk-tiles").innerHTML = RISKS.map((r) =>
    `<div class="tile" style="--c:${color(r)}"><div class="n">${counts[r]}</div><div class="l">${RISK_LABEL[r]}</div></div>`).join("");
  const h = curH();
  $("sum-hint").innerHTML = h === 0 ? "Colours show the latest measured water level." :
    `${alarms} location(s) with an <b>alert</b>: probability of exceeding the ${state.fc.alert_level} threshold within +${h} h above the alarm cut-off (tuned for ≥ 90 % recall).`;
}

function renderAlerts() {
  const h = curH();
  const al = state.fc.alert_level;
  const rows = state.fc.sensors.filter((s) => state.infra.has(s.infra_type)).map((s) => {
    const v = sensorView(s);
    // ranking: probability in window (forecast) or measured level (now)
    const key = h === 0 ? (s.current.level || 0) / s.thresholds[al] : v.p;
    return { s, v, key };
  }).sort((a, b) => b.key - a.key);
  $("alert-list").innerHTML = rows.map(({ s, v }) => {
    const right = h === 0
      ? `<div class="pv">${fmtCm(v.value)}</div><div class="pl">measured</div>`
      : `<div class="pv">${pct(v.p)}</div><div class="pl">P(&gt;${s.thresholds[al]} cm)</div>`;
    const sub = h === 0
      ? `${s.current.quality !== "good" ? "⚠ " + s.current.quality + " · " : ""}${RISK_LABEL[v.risk]}`
      : `${fmtCm(v.value)} <span style="opacity:.75">(${v.lo.toFixed(0)}–${v.hi.toFixed(0)})</span>${s.first_exceedance ? ` · ETA ${s.first_exceedance.after_h}–${s.first_exceedance.within_h} h` : ""}`;
    return `<div class="alert" data-id="${s.sensor_id}" style="--c:${color(v.risk)}">
      <div class="bar"></div>
      <div><div class="nm">${s.name}</div>
        <div class="sub"><span class="tag">${s.infra_type === "metro" ? "METRO" : "STREET"}</span>${v.alarm ? '<span class="tag alarm-tag">ALERT</span>' : ""}${sub}</div></div>
      <div class="pr">${right}</div></div>`;
  }).join("");
  $("alert-list").querySelectorAll(".alert").forEach((el) => el.addEventListener("click", () => {
    const id = el.dataset.id;
    const f = state.segments.features.find((x) => x.properties.sensor_id === id);
    if (f) {
      const c = f.geometry.coordinates;
      map.flyTo({ center: c[Math.floor(c.length / 2)], zoom: Math.max(map.getZoom(), 14.5), speed: 1.4 });
    }
    openDetail(id);
  }));
}

// ------------------------------------------------------------------ weather layer
const WX_SCALES = {
  precipitation: { stops: [[0, "rgba(0,0,0,0)"], [0.2, "#c6e3ff"], [1, "#6fb1ff"], [4, "#2f6fe0"], [10, "#5b2bbf"], [25, "#c4119f"]], unit: "mm/h", label: "Rainfall intensity" },
  acc: { stops: [[0, "rgba(0,0,0,0)"], [1, "#c6e3ff"], [5, "#6fb1ff"], [15, "#2f6fe0"], [40, "#5b2bbf"], [80, "#c4119f"]], unit: "mm", label: "Accumulated rainfall" },
  temperature_2m: { stops: [[0, "#2c7bb6"], [10, "#abd9e9"], [18, "#ffffbf"], [26, "#fdae61"], [34, "#d7191c"]], unit: "°C", label: "Temperature" },
  wind_speed_10m: { stops: [[0, "rgba(0,0,0,0)"], [10, "#d9f0e3"], [25, "#66c2a4"], [45, "#2ca25f"], [70, "#006d2c"]], unit: "km/h", label: "Wind speed" },
  relative_humidity_2m: { stops: [[30, "#fff7bc"], [60, "#a1dab4"], [80, "#41b6c4"], [100, "#225ea8"]], unit: "%", label: "Relative humidity" },
};

async function loadWeather() {
  if (!state.weatherVar) return;
  try { state.wx = await fetchJSON(`${API}/weather${qs()}`); } catch (e) { state.wx = null; toast("Weather layer unavailable: " + e.message); }
}

function renderWeather() {
  const vis = state.weatherVar ? "visible" : "none";
  map.setLayoutProperty("weather-fill", "visibility", vis);
  map.setLayoutProperty("weather-line", "visibility", vis);
  if (!state.weatherVar || !state.wx) { $("weather-hint").textContent = ""; return; }
  const sc = WX_SCALES[state.weatherVar];
  const h = curH();
  const vals = state.wx.horizons[String(h)] || {};
  const expr = ["match", ["get", "cell_id"]];
  let n = 0;
  for (const [cid, rec] of Object.entries(vals)) {
    const v = rec[state.weatherVar];
    if (v === null || v === undefined) continue;
    expr.push(cid, v); n++;
  }
  expr.push(-999);
  const interp = ["interpolate", ["linear"], expr, ...sc.stops.flat()];
  map.setPaintProperty("weather-fill", "fill-color", n ? ["case", ["==", expr, -999], "rgba(0,0,0,0)", interp] : "rgba(0,0,0,0)");
  map.setPaintProperty("weather-fill", "fill-opacity", 0.5);
  $("weather-hint").textContent = n ? `Source: ${state.wx.source}. Grid ≈ 3 km (independent from the water-level layer).`
    : `No weather data for this time (${state.wx.source}).`;
}

function renderLegend() {
  const al = state.fc ? state.fc.alert_level : "critical";
  const t = state.meta.thresholds[Object.keys(state.meta.thresholds)[0]];
  let html = `<h4 class="lg-toggle" title="Show / hide legend">Legend <span>${state.legendOpen === false ? "▸" : "▾"}</span></h4>`;
  if (state.legendOpen === false) { $("legend").innerHTML = html; bindLegend(); return; }
  html += `<h4>Water level (monitored segments)</h4>
    <div class="row"><span class="ln" style="background:${color("normal")}"></span>Normal &lt; ${t.elevated} cm</div>
    <div class="row"><span class="ln" style="background:${color("elevated")}"></span>Elevated ${t.elevated}–${t.dangerous} cm</div>
    <div class="row"><span class="ln" style="background:${color("dangerous")}"></span>Dangerous ${t.dangerous}–${t.critical} cm</div>
    <div class="row"><span class="ln" style="background:${color("critical")}"></span>Critical ≥ ${t.critical} cm</div>
    <div class="row"><span class="ln" style="background:${color("elevated")};box-shadow:0 0 6px 4px ${color("elevated")}66"></span>Glow width = uncertainty</div>
    <div class="row"><span class="ln" style="background:repeating-linear-gradient(90deg,#888 0 6px,transparent 6px 10px)"></span>Dashed = awareness horizon</div>
    <div class="sep"></div>
    <div class="row"><span class="ln" style="background:#C8102E;height:4px"></span>Metro line (real geometry)</div>
    <div class="row"><span class="ln" style="background:#e2001a;height:2px;opacity:.6"></span>Bus route (real geometry)</div>
    <div class="row"><span style="width:9px;height:9px;border-radius:50%;background:#43B02A;border:1.5px solid #111;display:inline-block;margin:0 7px 0 6px"></span>Train / bus · <b>simulated</b></div>`;
  if (state.weatherVar && WX_SCALES[state.weatherVar]) {
    const sc = WX_SCALES[state.weatherVar];
    html += `<div class="sep"></div><h4>${sc.label} (${sc.unit})</h4>
      <div class="grad" style="background:linear-gradient(90deg,${sc.stops.map((s) => s[1]).join(",")})"></div>
      <div class="gl"><span>${sc.stops[0][0]}</span><span>${sc.stops[sc.stops.length - 1][0]}+</span></div>`;
  }
  $("legend").innerHTML = html;
  bindLegend();
}
function bindLegend() {
  $("legend").querySelector(".lg-toggle").onclick = () => { state.legendOpen = state.legendOpen === false; renderLegend(); };
}

// ------------------------------------------------------------------ detail drawer
let detailCache = null;
async function openDetail(id) {
  state.selected = id;
  $("drawer").classList.add("open");
  $("drawer-inner").innerHTML = `<div class="d-sec">Loading…</div>`;
  try {
    detailCache = await fetchJSON(`${API}/sensor/${id}${qs()}`);
    renderDetail();
  } catch (e) { $("drawer-inner").innerHTML = `<div class="d-sec">Error: ${e.message}</div>`; }
  setTimeout(() => map.resize(), 220);
}
function closeDetail() {
  state.selected = null; detailCache = null;
  $("drawer").classList.remove("open");
  setTimeout(() => map.resize(), 220);
}

function confBars(c) {
  const n = c.label === "high" ? 3 : c.label === "medium" ? 2 : 1;
  return `<span class="conf">${[1, 2, 3].map((i) => `<i class="${i <= n ? "on" : ""}"></i>`).join("")}</span>`;
}

function renderDetail() {
  const d = detailCache;
  if (!d || d.sensor.sensor_id !== state.selected) return;
  const s = d.forecast; const al = state.fc.alert_level; const thr = s.thresholds;
  const h = curH() === 0 ? 1 : curH();
  const hz = s.horizons.find((x) => x.h === h);
  const v = sensorView(s);
  const lineBadges = (d.sensor.lines || []).map((l) => {
    const metro = /^L\d/.test(l);
    const col = metro ? (METRO_COLOURS[l] || "#555") : "#e2001a";
    return `<span class="line-badge" style="background:${col}">${l}</span>`;
  }).join("") || "<span class='hint'>none</span>";
  const stops = (d.sensor.stops || []).map((x) => `${x.name} <span class="hint">(${(x.lines || []).join(", ")})</span>`).join("<br>") || "–";
  const fe = s.first_exceedance;
  const callout = fe
    ? `<div class="callout" style="--c:${color("critical")}"><b>${al[0].toUpperCase() + al.slice(1)} threshold (${thr[al]} cm) may be exceeded within ${fe.after_h}–${fe.within_h} h</b>
       (probability ${pct(fe.probability)} at the first alerting horizon). Verify on site and follow the operational protocol.</div>`
    : `<div class="callout" style="--c:${color("normal")}">No exceedance of the ${al} threshold (${thr[al]} cm) expected within the next 72 h at the alarm cut-off.</div>`;
  const drv = hz.drivers || [];
  const maxAbs = Math.max(0.5, ...drv.map((x) => Math.abs(x.cm)));
  const drvHtml = drv.map((x) => {
    const w = (Math.abs(x.cm) / maxAbs) * 50;
    const pos = x.cm >= 0;
    return `<div class="drv"><span>${x.label}</span><div class="track"><div class="mid"></div>
      <div class="fill" style="${pos ? `left:50%;width:${w}%;background:${color("dangerous")}` : `right:50%;width:${w}%;background:#3b82f6`}"></div></div>
      <span style="text-align:right">${x.cm > 0 ? "+" : ""}${x.cm.toFixed(1)} cm</span></div>`;
  }).join("");
  const rowsHtml = s.horizons.map((x) => `<tr class="${x.tier === "awareness" ? "aw" : ""} ${x.h === curH() ? "sel" : ""}">
      <td>+${x.h} h</td><td>${fmtT(x.valid_time, { weekday: undefined })}</td>
      <td><span class="rdot" style="background:${color(x.risk)}"></span> ${x.level.median.toFixed(1)} <span class="hint">[${x.level.lo.toFixed(1)}–${x.level.hi.toFixed(1)}]</span></td>
      <td>${x.max.median.toFixed(1)}</td><td><b>${pct(x.p_exceed_max[al])}</b>${x.alarm ? " ⚠" : ""}</td>
      <td>${confBars(x.confidence)}</td></tr>`).join("");
  const cur = s.current;
  $("drawer-inner").innerHTML = `
    <div class="d-head" style="--c:${color(v.risk)}">
      <button class="close" id="d-close">✕</button>
      <div class="d-name">${d.sensor.name}</div>
      <div class="d-meta"><span class="tag">${d.sensor.sensor_id}</span><span class="tag">${d.sensor.infra_type === "metro" ? "METRO SECTION" : "STREET SECTION"}</span>
        <span class="tag" style="background:#ffb020;color:#2b1d00;border:0">SIMULATED SENSOR</span>
        ${d.mode === "replay" ? '<span class="tag" style="background:#6b4fbb;color:#fff;border:0">REPLAY</span>' : ""}</div>
    </div>
    <div class="d-sec">
      <div class="kpis">
        <div class="kpi"><div class="k">Measured now</div><div class="v" style="color:${color(cur.risk)}">${fmtCm(cur.level)}</div>
          <div class="s">${cur.quality === "good" ? "quality OK" : "⚠ " + cur.quality}${cur.stale_h > 0 ? ` · last valid ${cur.stale_h} h ago` : ""}</div></div>
        <div class="kpi"><div class="k">Forecast +${h} h</div><div class="v" style="color:${color(hz.risk)}">${fmtCm(hz.level.median)}</div>
          <div class="s">90 % interval ${hz.level.lo.toFixed(1)}–${hz.level.hi.toFixed(1)} cm</div></div>
        <div class="kpi"><div class="k">Max within +${h} h</div><div class="v" style="color:${color(hz.risk_max)}">${fmtCm(hz.max.median)}</div>
          <div class="s">upper bound ${hz.max.hi.toFixed(1)} cm</div></div>
        <div class="kpi"><div class="k">P(&gt; ${thr[al]} cm) within +${h} h</div><div class="v">${pct(hz.p_exceed_max[al])}</div>
          <div class="s">alarm cut-off ${pct(hz.alarm_p)} ${hz.alarm ? "· <b style='color:var(--critical)'>ALERT</b>" : ""}</div></div>
      </div>
      <div style="margin-top:8px;font-size:12px">Model confidence: ${confBars(hz.confidence)} <b>${hz.confidence.label}</b>
        <span class="hint">${hz.confidence.reasons.length ? "· " + hz.confidence.reasons.join(", ") : "· inputs fresh, narrow interval"}</span>
        <div class="hint">Confidence describes how reliable the forecast is. It is <b>not</b> the probability of flooding.</div></div>
    </div>
    <div class="d-sec">${callout}</div>
    <div class="d-sec"><h3>Measured history and forecast</h3><div id="chart"></div>
      <div class="hint">Solid: measured (synthetic) · dashed: forecast median · band: 90 % prediction interval${d.mode === "replay" ? " · dotted grey: what was actually measured afterwards" : ""}. Bars: rainfall (observed / forecast).</div></div>
    <div class="d-sec"><h3>Why this forecast (+${h} h)</h3>${drvHtml || "<span class='hint'>n/a</span>"}
      <div class="hint">Contribution of each factor to the median forecast relative to the model's baseline (LightGBM feature attributions).</div></div>
    <div class="d-sec"><h3>Forecast by horizon</h3>
      <table class="ft"><thead><tr><th>Horizon</th><th>Valid</th><th>Level (cm) [90 %]</th><th>Max</th><th>P(&gt;${thr[al]})</th><th>Conf.</th></tr></thead>
      <tbody>${rowsHtml}</tbody></table>
      <div class="hint">Italic rows (48–72 h): awareness only, low confidence.</div></div>
    <div class="d-sec"><h3>Infrastructure</h3>
      <div class="kv">
        <div class="k">Affected lines</div><div>${lineBadges}</div>
        <div class="k">${d.sensor.infra_type === "metro" ? "Stations" : "Bus stops nearby"}</div><div>${stops}</div>
        <div class="k">Coordinates</div><div>${d.sensor.lat.toFixed(5)}, ${d.sensor.lon.toFixed(5)}</div>
        <div class="k">Segment length</div><div>${d.sensor.segment_length_m ? Math.round(d.sensor.segment_length_m) + " m" : "–"}</div>
        <div class="k">Thresholds</div><div>${thr.elevated} / ${thr.dangerous} / ${thr.critical} cm</div>
        <div class="k">Weather cell</div><div>${d.sensor.weather_cell} (upstream ${d.sensor.upstream_cell})</div>
        <div class="k">Geometry</div><div>${(d.sensor.attrs && d.sensor.attrs.geometry_source) || "–"}</div>
        ${d.sensor.attrs && d.sensor.attrs.note ? `<div class="k">Note</div><div>${d.sensor.attrs.note}</div>` : ""}
      </div></div>`;
  $("d-close").onclick = closeDetail;
  drawChart(d);
}

function drawChart(d) {
  const s = d.forecast; const thr = s.thresholds; const t0 = parseT(d.issued_at);
  const dark = state.theme === "dark";
  const fg = dark ? "#c9d1d9" : "#3b4651"; const grid = dark ? "#2a333d" : "#e7ebef";
  const tz = (iso) => iso;  // plotly shows UTC strings; label axis accordingly
  const hx = d.history.map((r) => tz(r.ts)); const hy = d.history.map((r) => r.v);
  const fx = [d.issued_at, ...s.horizons.map((x) => x.valid_time)];
  const cur = s.current.level ?? 0;
  const med = [cur, ...s.horizons.map((x) => x.level.median)];
  const lo = [cur, ...s.horizons.map((x) => x.level.lo)];
  const hi = [cur, ...s.horizons.map((x) => x.level.hi)];
  const traces = [
    { x: fx, y: lo, mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { x: fx, y: hi, mode: "lines", fill: "tonexty", fillcolor: "rgba(213,0,50,0.13)", line: { width: 0 }, name: "90 % interval", hoverinfo: "skip" },
    { x: hx, y: hy, mode: "lines", line: { color: dark ? "#e6eaee" : "#17202a", width: 2 }, name: "Measured", hovertemplate: "%{y:.1f} cm<extra>measured</extra>" },
    { x: fx, y: med, mode: "lines+markers", line: { color: "#D50032", width: 2, dash: "dash" }, marker: { size: 5 }, name: "Forecast median", hovertemplate: "%{y:.1f} cm<extra>forecast</extra>" },
  ];
  if (d.observed_after && d.observed_after.length) {
    traces.push({ x: d.observed_after.map((r) => r.ts), y: d.observed_after.map((r) => r.v), mode: "lines",
      line: { color: "#8a96a3", width: 1.6, dash: "dot" }, name: "Measured afterwards (hindsight)" });
  }
  traces.push({ x: d.rain_history.map((r) => r.ts), y: d.rain_history.map((r) => r.v), type: "bar", yaxis: "y2",
    marker: { color: "#4a90e2" }, name: "Rain observed (mm/h)", hovertemplate: "%{y:.1f} mm/h<extra>observed</extra>" });
  if (d.rain_forecast && d.rain_forecast.length) {
    traces.push({ x: d.rain_forecast.map((r) => r.ts), y: d.rain_forecast.map((r) => r.v), type: "bar", yaxis: "y2",
      marker: { color: "rgba(74,144,226,0.4)", line: { color: "#4a90e2", width: 0.5 } }, name: "Rain forecast (mm/h)", hovertemplate: "%{y:.1f} mm/h<extra>forecast</extra>" });
  }
  const after = (d.observed_after || []).map((r) => r.v).filter((x) => x !== null);
  const ymax = Math.max(thr.critical * 1.25, ...hy.filter((x) => x !== null), ...hi, ...after) * 1.05;
  const awStart = new Date(t0.getTime() + state.meta.operational_max_h * 3600e3).toISOString();
  const end = s.horizons[s.horizons.length - 1].valid_time;
  const hl = (y, c) => ({ type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: { color: c, width: 1, dash: "dot" } });
  const layout = {
    margin: { l: 42, r: 10, t: 6, b: 30 }, height: 340, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { size: 10.5, color: fg }, showlegend: true, legend: { orientation: "h", y: -0.16, x: 0, font: { size: 9.5 } },
    xaxis: { gridcolor: grid, type: "date", tickformat: "%d %b %H:%M", nticks: 6 },
    yaxis: { domain: [0.36, 1], title: { text: "Water level (cm)" }, gridcolor: grid, range: [0, ymax], zeroline: false },
    yaxis2: { domain: [0.08, 0.30], title: { text: "mm/h" }, gridcolor: grid, autorange: "reversed", rangemode: "tozero" },
    bargap: 0.1,
    shapes: [hl(thr.elevated, color("elevated")), hl(thr.dangerous, color("dangerous")), hl(thr.critical, color("critical")),
      { type: "line", x0: d.issued_at, x1: d.issued_at, yref: "paper", y0: 0, y1: 1, line: { color: fg, width: 1 } },
      { type: "rect", x0: awStart, x1: end, yref: "paper", y0: 0.36, y1: 1, fillcolor: "rgba(140,140,140,0.10)", line: { width: 0 } }],
    annotations: [
      { x: d.issued_at, y: 1, yref: "paper", text: "issued", showarrow: false, yanchor: "bottom", font: { size: 9 } },
      { x: awStart, y: 0.98, yref: "paper", text: "awareness (low confidence)", showarrow: false, xanchor: "left", font: { size: 9, color: "#888" } },
      { xref: "paper", x: 1, y: thr.critical, text: `${thr.critical} cm`, showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 9, color: color("critical") } }],
  };
  Plotly.react("chart", traces, layout, { displayModeBar: false, responsive: true });
}

// ------------------------------------------------------------------ vehicles (simulated)
const METRO_COLOURS = {};
function prepareVehicles(list) {
  const R = 6371000, rad = Math.PI / 180;
  return list.map((v) => {
    if (v.kind === "metro") METRO_COLOURS[v.line] = v.colour;
    const cum = [0];
    for (let i = 1; i < v.path.length; i++) {
      const [x1, y1] = v.path[i - 1], [x2, y2] = v.path[i];
      const dx = (x2 - x1) * rad * Math.cos(((y1 + y2) / 2) * rad) * R, dy = (y2 - y1) * rad * R;
      cum.push(cum[i - 1] + Math.hypot(dx, dy));
    }
    return { ...v, cum, L: cum[cum.length - 1] };
  });
}
function vehiclePos(v, s) {
  const c = v.cum; let lo = 0, hi = c.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c[m] <= s) lo = m; else hi = m; }
  const f = (s - c[lo]) / Math.max(1e-6, c[hi] - c[lo]);
  const a = v.path[lo], b = v.path[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
}
let lastVeh = 0;
function animateVehicles(ts) {
  requestAnimationFrame(animateVehicles);
  if (!$("layer-chips").querySelector('input[value="vehicles"]').checked) return;
  if (ts - lastVeh < 100) return;  // ~10 fps is enough and keeps the map fluid
  lastVeh = ts;
  const t = Date.now() / 1000 * 6;  // accelerated x6 for visibility
  const showBus = $("layer-chips").querySelector('input[value="bus"]').checked;
  const showMetro = $("layer-chips").querySelector('input[value="metro"]').checked;
  const feats = [];
  for (const v of state.vehicles) {
    if ((v.kind === "bus" && !showBus) || (v.kind === "metro" && !showMetro)) continue;
    for (let k = 0; k < v.n; k++) {
      let s = (t * v.speed_mps + (k * 2 * v.L) / v.n) % (2 * v.L);
      if (s > v.L) s = 2 * v.L - s;
      feats.push({ type: "Feature", geometry: { type: "Point", coordinates: vehiclePos(v, s) },
                   properties: { kind: v.kind, line: v.line, colour: v.colour } });
    }
  }
  const src = map.getSource("vehicles");
  if (src) src.setData({ type: "FeatureCollection", features: feats });
}

// ------------------------------------------------------------------ data loading
async function loadForecast() {
  loading(true);
  try {
    state.fc = await fetchJSON(`${API}/forecast${qs()}`);
    $("issue-time").textContent = fmtT(state.fc.issued_at) + " (Madrid)";
    const pill = $("mode-pill");
    pill.textContent = state.fc.mode === "live" ? "LIVE" : "REPLAY";
    pill.className = "pill " + (state.fc.mode === "live" ? "pill-live" : "pill-replay");
    state.steps = [0, ...state.fc.horizons];
    buildTicks();
    if (state.fc.mode === "live" && !state.fc.weather_forecast.ok) {
      toast("No recent live weather forecast run is available: future rainfall inputs are degraded. Forecast confidence is reduced accordingly.", 9000);
    }
    await loadWeather();
    render();
    if (state.selected) openDetail(state.selected);
  } catch (e) {
    toast("Forecast unavailable: " + e.message, 12000);
  } finally { loading(false); }
}

async function loadStatus() {
  try {
    const st = await fetchJSON(`${API}/status`);
    const src = (st.sources || []).find((x) => x.source === "open-meteo:forecast");
    const dot = $("wx-dot"); dot.className = "dot " + (src ? src.status : "unavailable");
    const age = st.sensor_data.age_hours;
    $("wx-label").textContent = src && src.status === "ok" ? "Weather live" : "Weather degraded";
    if (age !== null && age > 6) {
      toast(`Latest sensor/weather data is ${Math.round(age)} h old (last: ${fmtT(st.sensor_data.last_observation)}). Run "python main.py update" or check data sources.`, 10000);
    }
    state.status = st;
  } catch (e) { /* ignore */ }
}

function buildTicks() {
  $("slider").max = state.steps.length - 1;
  if (state.hIdx > state.steps.length - 1) state.hIdx = 0;
  $("slider").value = state.hIdx;
  $("ticks").innerHTML = state.steps.map((h, i) => `<span data-i="${i}">${h === 0 ? "Now" : "+" + h + "h"}</span>`).join("");
  $("ticks").querySelectorAll("span").forEach((el) => el.addEventListener("click", () => setH(+el.dataset.i)));
}
function setH(i) { state.hIdx = i; $("slider").value = i; render(); }

async function loadEvents() {
  try {
    const ev = await fetchJSON(`${API}/events`);
    $("event-select").innerHTML = `<option value="">Select a past episode…</option>` + ev.map((e) =>
      `<option value="${e.issue_suggested}">${fmtT(e.start, { weekday: undefined, year: "numeric" })} · ${e.n_sensors} sensor(s) · peak ${e.peak_cm} cm · ${e.period}</option>`).join("");
  } catch (e) { $("event-select").innerHTML = `<option>Events unavailable</option>`; }
}

// ------------------------------------------------------------------ modals
function md2html(md) {
  const lines = md.split("\n"); let html = ""; let inTable = false;
  for (const ln of lines) {
    if (ln.startsWith("|")) {
      const cells = ln.split("|").slice(1, -1).map((c) => c.trim());
      if (cells.every((c) => /^-+$/.test(c))) continue;
      if (!inTable) { html += "<table><thead><tr>" + cells.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>"; inTable = true; continue; }
      html += "<tr>" + cells.map((c) => `<td>${c}</td>`).join("") + "</tr>"; continue;
    }
    if (inTable) { html += "</tbody></table>"; inTable = false; }
    const b = (s) => s.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
    if (ln.startsWith("# ")) html += `<h1>${b(ln.slice(2))}</h1>`;
    else if (ln.startsWith("## ")) html += `<h2>${b(ln.slice(3))}</h2>`;
    else if (ln.startsWith("> ")) html += `<blockquote>${b(ln.slice(2))}</blockquote>`;
    else if (ln.startsWith("* ")) html += `<div>• ${b(ln.slice(2))}</div>`;
    else if (ln.trim()) html += `<p>${b(ln)}</p>`;
  }
  if (inTable) html += "</tbody></table>";
  return html;
}
async function showModel() {
  const r = await fetchJSON(`${API}/model`);
  $("modal-body").innerHTML = r.available ? md2html(r.markdown) +
    `<h2>Method</h2><p>Global LightGBM quantile regression (8 quantiles) per horizon for the level at t+h and the maximum in (t, t+h];
     conformalized quantile regression (dry/wet regimes) for calibrated 90 % intervals; exceedance probabilities read from the predictive
     distribution and recalibrated (logistic); alarm cut-off chosen on the calibration period to keep recall ≥ 90 %.
     Chronological train / validation / calibration / test splits with 72 h gaps. Forecast rainfall inputs at training time come from
     archived model runs (Open-Meteo previous-runs API), not from perfect observed rainfall.</p>`
    : "<h1>Model performance</h1><p>No evaluation report yet. Run <code>python main.py evaluate</code>.</p>";
  $("modal").classList.remove("hidden");
}
async function showSources() {
  const st = await fetchJSON(`${API}/status`);
  const rows = (st.sources || []).map((s) => `<div class="src-row"><b>${s.source}</b><span><span class="dot ${s.status}"></span> ${s.status}</span>
    <span>${s.message || ""}<br><span class="hint">last success: ${s.last_success ? fmtT(s.last_success) : "never"}</span></span></div>`).join("");
  $("modal-body").innerHTML = `<h1>Data sources & freshness</h1>
    <blockquote>Water-level sensors are <b>SIMULATED</b> (physically-based model driven by real Open-Meteo rainfall). Vehicle positions are <b>SIMULATED</b> along real OSM route geometries.</blockquote>
    <div class="src-row"><b>Sensor data (synthetic)</b><span></span><span>last observation ${st.sensor_data.last_observation ? fmtT(st.sensor_data.last_observation) : "–"} (${st.sensor_data.age_hours} h ago)</span></div>
    <div class="src-row"><b>Live weather forecast run</b><span></span><span>${st.weather_forecast_run ? fmtT(st.weather_forecast_run) : "none available"}</span></div>
    ${rows}
    <div class="src-row"><b>Model</b><span></span><span>${st.model.version || "not trained"} · splits: ${st.model.splits ? st.model.splits.mode : "–"}</span></div>
    <h2>Sources</h2>
    <p>Weather: <a href="https://open-meteo.com" target="_blank">Open-Meteo</a> (AROME 1.3 km / ECMWF / ICON, historical-forecast & previous-runs APIs, CC BY 4.0) ·
       optional <a href="https://apidocs.meteo.cat" target="_blank">Meteocat XEMA</a> and <a href="https://opendata.aemet.es" target="_blank">AEMET OpenData</a> stations (API keys in .env).<br>
       Transport geometry: <a href="https://www.openstreetmap.org" target="_blank">OpenStreetMap</a> (ODbL) — TMB metro and bus route relations; optional
       <a href="https://developer.tmb.cat" target="_blank">TMB Developer</a> GTFS (app_id / app_key).<br>Basemap: OpenFreeMap (OpenMapTiles, © OpenStreetMap contributors).</p>
    <p><button class="tbtn" id="btn-refresh" style="background:var(--accent);border:0">Refresh data now</button></p>`;
  $("modal").classList.remove("hidden");
  $("btn-refresh").onclick = async () => { await fetch(`${API}/refresh`, { method: "POST" }); toast("Data update started (weather, sensors). The forecast refreshes automatically.", 6000); setTimeout(() => { loadStatus(); loadForecast(); }, 30000); };
}

// ------------------------------------------------------------------ controls
function bindControls() {
  $("slider").addEventListener("input", (e) => setH(+e.target.value));
  $("btn-play").addEventListener("click", () => {
    if (state.playing) { clearInterval(state.playing); state.playing = null; $("btn-play").textContent = "▶"; return; }
    $("btn-play").textContent = "❚❚";
    state.playing = setInterval(() => setH((state.hIdx + 1) % state.steps.length), 1300);
  });
  $("metric-select").addEventListener("change", (e) => { state.metric = e.target.value; render(); });
  $("infra-chips").addEventListener("change", (e) => { e.target.checked ? state.infra.add(e.target.value) : state.infra.delete(e.target.value); render(); });
  $("risk-chips").innerHTML = RISKS.map((r) => `<label class="chip"><input type="checkbox" value="${r}" checked><span class="sw" style="background:${color(r)}"></span>${RISK_LABEL[r]}</label>`).join("");
  $("risk-chips").addEventListener("change", (e) => { e.target.checked ? state.risks.add(e.target.value) : state.risks.delete(e.target.value); render(); });
  $("layer-chips").addEventListener("change", (e) => {
    const on = e.target.checked ? "visible" : "none";
    const m = { metro: ["metro-lines", "metro-casing", "stations"], bus: ["bus-lines"], stops: ["stops"], vehicles: ["veh-bus", "veh-metro"] };
    (m[e.target.value] || []).forEach((id) => map.setLayoutProperty(id, "visibility", on));
  });
  $("weather-select").addEventListener("change", async (e) => { state.weatherVar = e.target.value; await loadWeather(); render(); });
  $("mode-ctl").querySelectorAll("button").forEach((b) => b.addEventListener("click", async () => {
    $("mode-ctl").querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b));
    state.mode = b.dataset.mode;
    $("event-select").classList.toggle("hidden", state.mode !== "replay");
    $("mode-hint").textContent = state.mode === "live" ? "Latest data and the most recent weather forecast run."
      : "Re-runs the model as it would have been issued in the past, using archived weather forecasts. The chart also shows what was actually measured afterwards.";
    if (state.mode === "live") { state.issue = null; await loadForecast(); }
    else { await loadEvents(); }
  }));
  $("event-select").addEventListener("change", async (e) => { if (!e.target.value) return; state.issue = e.target.value; state.hIdx = Math.min(state.hIdx || 4, 8); await loadForecast(); });
  $("btn-model").addEventListener("click", showModel);
  $("btn-sources").addEventListener("click", showSources);
  $("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));
  $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("modal").classList.add("hidden"); });
  $("btn-theme").addEventListener("click", () => {
    state.theme = state.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = state.theme;
    setBasemap(state.theme);
    if (detailCache) drawChart(detailCache);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { $("modal").classList.add("hidden"); closeDetail(); }
    if (e.key === "ArrowRight") setH(Math.min(state.hIdx + 1, state.steps.length - 1));
    if (e.key === "ArrowLeft") setH(Math.max(state.hIdx - 1, 0));
  });
}

// ------------------------------------------------------------------ boot
(async function boot() {
  state.meta = await fetchJSON(`${API}/meta`);
  $("app-title").textContent = state.meta.project.name;
  $("app-subtitle").textContent = state.meta.project.subtitle;
  document.title = state.meta.project.name;
  $("logo").src = "/static/" + state.meta.branding.logo_path;
  $("logo").onerror = () => { $("logo").style.display = "none"; };
  document.documentElement.style.setProperty("--accent", state.meta.branding.primary_color);
  bindControls();
  await initMap();
  await addLayers();
  renderLegend();
  await loadStatus();
  await loadForecast();
  requestAnimationFrame(animateVehicles);
  setInterval(() => { if (state.mode === "live") { loadStatus(); loadForecast(); } }, 10 * 60 * 1000);
})();
