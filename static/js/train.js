"use strict";

let DATA = null;     // response from /api/train
let CK = 0;          // current checkpoint index
let timer = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

/* ---------- per-session cache of the last trained result ----------
 * Stored in sessionStorage so navigating between the Inference and Training
 * pages (same browser session) restores the trained view instantly instead of
 * retraining. The actual weights also persist on disk server-side, so the
 * Inference page's "Use trained weights" keeps working across sessions. */
const CACHE_KEY = "moe_train_result_v1";

/* Best Tiny Shakespeare settings (from the parameter sweep). d_model/num_layers
 * stay 0 = "auto", which resolves to the dataset defaults (32 / 2 layers). */
const BEST_DEFAULTS = {
  dataset: "shakespeare", optimizer: "adam", lr: 0.03, epochs: 170,
  d_model: 0, num_layers: 0, slice_chars: 2500, block_size: 32, seed: 42,
  balance_alpha: 0,
};

function cacheResult() {
  try {
    sessionStorage.setItem(CACHE_KEY, JSON.stringify(DATA));
  } catch (e) { /* quota exceeded -> just skip caching */ }
}

function loadCachedResult() {
  try {
    const raw = sessionStorage.getItem(CACHE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}

function clearCachedResult() {
  try { sessionStorage.removeItem(CACHE_KEY); } catch (e) {}
}

function showDoneBanner(restored) {
  const status = $("trainStatus");
  const srcNote = DATA.mode === "char" && DATA.info.source === "fallback"
    ? " (used offline fallback text — no internet)" : "";
  status.className = "trained-banner ok";
  const lead = restored
    ? "&#10003; Restored your last trained model <b>(cached this session)</b> &mdash; no retraining needed."
    : `&#10003; Trained ${DATA.params.epochs} epochs on <b>${DATA.info.name}</b>${srcNote}.`;
  status.innerHTML = `${lead} ` +
    `loss ${DATA.checkpoints[0].loss} → <b>${DATA.final_loss}</b> · ` +
    `${DATA.config.num_layers} layer(s), d_model ${DATA.config.d_model}. ` +
    `Open the <a href="/">Inference page</a> and tick "Use trained weights"` +
    (restored ? `, or press <b>Train model</b> to retrain.` : `.`);
}

/* ---------- heatmap helpers ---------- */
function maxAbs(nested) {
  let m = 0;
  const walk = (x) => Array.isArray(x) ? x.forEach(walk) : (m = Math.max(m, Math.abs(x)));
  walk(nested);
  return m || 1;
}
function heat(v, m) {
  const t = Math.max(-1, Math.min(1, v / (m || 1)));
  const a = (0.1 + 0.65 * Math.abs(t)).toFixed(2);
  const rgb = t >= 0 ? "47,129,247" : "248,81,73";
  return `background:rgba(${rgb},${a})`;
}
function fmt(v, d = 2) { return (v >= 0 ? " " : "") + v.toFixed(d); }

function matrix(mat, rowLabels, colLabels) {
  const m = maxAbs(mat);
  const cols = colLabels || mat[0].map((_, i) => i);
  let html = `<table class="mat"><thead><tr><th></th>` +
    cols.map(c => `<th>${c}</th>`).join("") + `</tr></thead><tbody>`;
  mat.forEach((row, r) => {
    html += `<tr><td class="rowhead">${rowLabels ? esc(String(rowLabels[r])) : r}</td>` +
      row.map(v => `<td style="${heat(v, m)}">${fmt(v)}</td>`).join("") + `</tr>`;
  });
  return html + `</tbody></table>`;
}

/* ---------- dataset info ---------- */
function renderInfo() {
  const info = DATA.info;
  $("dsBadge").textContent = `${info.name} · vocab ${info.vocab_size}`;
  $("dsDesc").textContent = info.description;

  if (DATA.mode === "char") {
    const srcLabel = { downloaded: "downloaded just now", cache: "loaded from local cache",
                       fallback: "offline fallback snippet" }[info.source] || info.source;
    const chips = info.vocab_preview.map(c =>
      `<span class="tok">${esc(c)}</span>`).join("");
    $("dsBody").innerHTML = `
      <div class="ds-meta">
        <span class="badge">source: ${srcLabel}</span>
        <span class="badge">slice: ${info.slice_chars} / ${info.full_chars} chars</span>
        <span class="badge">windows: ${info.num_examples} × ${info.block_size}</span>
      </div>
      <p class="sub" style="margin-top:12px">character vocabulary (${info.vocab_size})</p>
      <div class="corpus-line">${chips}</div>
      <p class="sub" style="margin-top:12px">text preview</p>
      <pre class="text-preview">${esc(info.preview)}</pre>`;
  } else {
    const lines = info.examples.map(e =>
      `<div class="corpus-line">` +
      e.tokens.map(t => `<span class="tok ${t.startsWith("<") ? "special" : ""}">${esc(t)}</span>`).join("") +
      `</div>`).join("");
    $("dsBody").innerHTML = `<div class="corpus-list">${lines}</div>`;
  }
}

/* ---------- loss chart (SVG) ---------- */
function renderLossChart() {
  const W = 900, H = 260, padL = 50, padR = 20, padT = 18, padB = 30;
  const init = DATA.checkpoints[0].loss;
  const series = [{ e: 0, loss: init }].concat(
    DATA.loss_curve.map((l, i) => ({ e: i + 1, loss: l })));
  const maxE = series[series.length - 1].e || 1;
  const maxL = Math.max(...series.map(p => p.loss)) * 1.05 || 1;

  const X = (e) => padL + (e / maxE) * (W - padL - padR);
  const Y = (l) => padT + (1 - l / maxL) * (H - padT - padB);

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const lv = maxL * i / 4, y = Y(lv);
    grid += `<line class="grid-line" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>` +
      `<text class="chart-label" x="${padL - 8}" y="${y + 3}" text-anchor="end">${lv.toFixed(1)}</text>`;
  }
  const pts = series.map(p => `${X(p.e)},${Y(p.loss)}`).join(" ");
  const area = `${X(0)},${Y(0)} ${pts} ${X(maxE)},${Y(0)}`;

  let marks = "";
  DATA.checkpoints.forEach((c, i) => {
    const x = X(c.epoch), active = i === CK;
    marks += `<line class="ck-marker" x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" style="opacity:${active ? 1 : .25}"/>` +
      `<circle class="ck-dot" cx="${x}" cy="${Y(c.loss)}" r="${active ? 5 : 3}" style="opacity:${active ? 1 : .6}"><title>epoch ${c.epoch}, loss ${c.loss}</title></circle>`;
  });

  $("lossChart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" id="lossSvg">
      ${grid}
      <polygon class="loss-area" points="${area}"/>
      <polyline class="loss-path" points="${pts}"/>
      ${marks}
      <line class="axis-line" x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}"/>
      <text class="chart-label" x="${W - padR}" y="${H - padB + 22}" text-anchor="end">epoch &rarr;</text>
      <text class="chart-label" x="${(padL + W - padR) / 2}" y="${H - padB + 22}" text-anchor="middle">${maxE}</text>
    </svg>`;
  $("lossBadge").textContent = `${init.toFixed(2)} → ${DATA.final_loss}`;

  const svg = $("lossSvg");
  svg.style.cursor = "pointer";
  svg.addEventListener("click", (ev) => {
    const r = svg.getBoundingClientRect();
    const epoch = (((ev.clientX - r.left) / r.width * W) - padL) / (W - padL - padR) * maxE;
    let best = 0, bd = 1e9;
    DATA.checkpoints.forEach((c, i) => { const d = Math.abs(c.epoch - epoch); if (d < bd) { bd = d; best = i; } });
    stopPlay(); CK = best; renderCheckpoint();
  });
}

/* ---------- metric headline + sample generations for one checkpoint ----- */
function renderMetricAndSamples(c, liveEpoch) {
  $("metricVal").textContent = (c.metric.value * 100).toFixed(0) + "%";
  $("metricLabel").textContent = c.metric.label;
  $("metricDetail").textContent = c.metric.detail;

  const liveTag = liveEpoch !== undefined
    ? ` <span class="live-pill">live · epoch ${liveEpoch}</span>` : "";
  if (DATA.mode === "char") {
    $("samplesDesc").innerHTML = "<b>Character&#8209;by&#8209;character</b> generation (sampled, temp 0.8) from seed prompts (40 chars). Watch fragments become word&#8209;shaped as the loss drops." + liveTag;
    $("samples").innerHTML = c.samples.map(s =>
      `<div class="sample-row char">
        <span class="seed">${esc(s.prompt)}</span><span class="arrow">⟶</span>
        <span class="cont">${esc(s.generated)}</span>
      </div>`).join("");
  } else {
    $("samplesDesc").innerHTML = "Greedy generation + the probability the model assigns to the <b>expected</b> next token." + liveTag;
    $("samples").innerHTML = c.samples.map(s => {
      const pct = (s.expected_prob * 100).toFixed(0);
      return `<div class="sample-row">
        <span class="prompt">${esc(s.prompt)} →</span>
        <span class="gen"><b>${esc(s.generated)}</b></span>
        <span class="prompt">expected <b style="color:var(--text)">${esc(s.expected)}</b></span>
        <div class="prob-mini" title="P=${s.expected_prob}"><div style="width:${pct}%"></div></div>
        <span class="prompt">p=${s.expected_prob.toFixed(2)}</span>
        <span class="verdict ${s.correct ? "ok" : "no"}">${s.correct ? "✓" : "✗"}</span>
      </div>`;
    }).join("");
  }
}

function renderHeatmaps(c) {
  const rowLabels = (DATA.info && DATA.info.vocab_preview) ? DATA.info.vocab_preview : DATA.vocab;
  $("embedBadge").textContent = `${c.embed.length} × ${c.embed[0].length}`;
  $("embedHeat").innerHTML = matrix(c.embed, rowLabels, c.embed[0].map((_, i) => "d" + i));

  $("routerHeat").innerHTML = matrix(c.W_router,
    c.W_router.map((_, i) => "d" + i), c.expert_usage.map((_, i) => "E" + i));

  const maxU = Math.max(...c.expert_usage, 0.001);
  let bars = c.expert_usage.map((u, i) =>
    `<div class="gate-row"><span class="ename">expert ${i}</span>` +
    `<div class="gate-track"><div class="gate-fill" style="width:${(u / maxU * 100).toFixed(1)}%"></div></div>` +
    `<span>${u.toFixed(3)}</span></div>`).join("");
  if (c.balance != null) {
    const E = DATA.config.num_experts;
    const pct = Math.max(0, Math.min(100,
      (1 - (c.balance - 1) / (E - 1)) * 100)).toFixed(0);  // 100% = perfectly even
    bars += `<div class="balance-row" title="E·Σ P_e² — 1.0 = perfectly even routing, up to ${E} = collapsed onto one expert">` +
      `balance factor <b>${c.balance.toFixed(3)}</b> ` +
      `<span class="muted">(${pct}% even · 1.0 = ideal)</span></div>`;
  }
  $("usageBars").innerHTML = bars;

  // per-expert W1/W2 weight tables (layer 0). Guard for older cached results.
  const ex = $("expertHeat");
  if (ex) {
    if (c.experts && c.experts.length) {
      const d = DATA.config.d_model, h = DATA.config.expert_hidden;
      $("expertBadge").textContent = `layer 0 · ${c.experts.length} experts · W1 ${d}×${h} · W2 ${h}×${d}`;
      const dCols = Array.from({ length: d }, (_, i) => "d" + i);
      ex.innerHTML = c.experts.map((e, i) => {
        const sel = c.expert_usage[i] === Math.max(...c.expert_usage);
        return `<div class="expert-block${sel ? " top" : ""}">
          <p class="sub">Expert ${i}${sel ? " · most used" : ""} · gate ${(c.expert_usage[i]).toFixed(3)}</p>
          <div class="expert-mats">
            <div><span class="mat-cap">W1 (in → hidden)</span><div class="heat-scroll">${matrix(e.W1, dCols)}</div></div>
            <div><span class="mat-cap">W2 (hidden → out)</span><div class="heat-scroll">${matrix(e.W2, null, dCols)}</div></div>
          </div>
        </div>`;
      }).join("");
    } else {
      ex.innerHTML = `<p class="desc">Expert weights unavailable for this (older cached) result — press <b>Train model</b> to regenerate.</p>`;
      $("expertBadge").textContent = "";
    }
  }
}

/* ---------- one checkpoint (interactive view, post-training) ---------- */
function renderCheckpoint() {
  const c = DATA.checkpoints[CK];
  $("ckSlider").value = CK;
  $("ckLabel").textContent = `Checkpoint ${CK + 1}/${DATA.checkpoints.length} · epoch ${c.epoch} · loss ${c.loss}`;
  $("prevCk").disabled = CK === 0;
  $("nextCk").disabled = CK === DATA.checkpoints.length - 1;

  renderMetricAndSamples(c);
  renderHeatmaps(c);
  renderLossChart();
}

/* ---------- autoplay ---------- */
function play() {
  if (CK >= DATA.checkpoints.length - 1) CK = -1;
  $("playCk").innerHTML = "&#10073;&#10073; Pause";
  timer = setInterval(() => {
    if (CK >= DATA.checkpoints.length - 1) { stopPlay(); return; }
    CK++; renderCheckpoint();
  }, 1100);
}
function stopPlay() {
  if (timer) { clearInterval(timer); timer = null; }
  $("playCk").innerHTML = "&#9654; Play";
}

/* ---------- live loss chart (grows as training streams in) ---------- */
function renderLiveChart(live) {
  const W = 900, H = 260, padL = 50, padR = 20, padT = 18, padB = 30;
  const initLoss = live.checkpoints.length ? live.checkpoints[0].loss : 0;
  const series = [{ e: 0, loss: initLoss }].concat(
    live.loss_curve.map((l, i) => ({ e: i + 1, loss: l })));
  const maxE = live.total_epochs || 1;                 // fixed full-width x-axis
  const maxL = Math.max(initLoss, ...live.loss_curve, 0.1) * 1.05;

  const X = (e) => padL + (e / maxE) * (W - padL - padR);
  const Y = (l) => padT + (1 - l / maxL) * (H - padT - padB);

  let grid = "";
  for (let i = 0; i <= 4; i++) {
    const lv = maxL * i / 4, y = Y(lv);
    grid += `<line class="grid-line" x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}"/>` +
      `<text class="chart-label" x="${padL - 8}" y="${y + 3}" text-anchor="end">${lv.toFixed(1)}</text>`;
  }
  const pts = series.map(p => `${X(p.e)},${Y(p.loss)}`).join(" ");
  const last = series[series.length - 1];

  $("lossChart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}">
      ${grid}
      <polyline class="loss-path live" points="${pts}"/>
      <circle class="live-dot" cx="${X(last.e)}" cy="${Y(last.loss)}" r="5"/>
      <line class="axis-line" x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}"/>
      <text class="chart-label" x="${W - padR}" y="${H - padB + 22}" text-anchor="end">epoch &rarr;</text>
      <text class="chart-label" x="${(padL + W - padR) / 2}" y="${H - padB + 22}" text-anchor="middle">${maxE}</text>
    </svg>`;
  $("lossBadge").textContent = `${initLoss.toFixed(2)} → ${last.loss.toFixed(3)}`;
}

function setProgress(epoch, total) {
  const pct = total ? Math.round(epoch / total * 100) : 0;
  $("progressFill").style.width = pct + "%";
  $("progressText").textContent = `epoch ${epoch}/${total} · ${pct}%`;
}

/* ---------- train (streamed via Server-Sent Events) ---------- */
async function train() {
  stopPlay();
  const btn = $("trainBtn");
  btn.disabled = true; btn.textContent = "Training…";
  const status = $("trainStatus");
  status.className = "trained-banner hidden";
  $("progressWrap").classList.remove("hidden");
  $("progressFill").style.width = "0%";
  $("progressText").textContent = "starting…";
  $("liveLoss").innerHTML = "";

  // Live, progressively-built state mirroring the final DATA shape.
  let live = null;

  const handle = (ev) => {
    if (ev.type === "start") {
      live = { mode: ev.mode, dataset: ev.dataset, config: ev.config,
               vocab: ev.vocab, info: ev.info, params: ev.params,
               total_epochs: ev.total_epochs, loss_curve: [], checkpoints: [] };
      DATA = live;
      renderInfo();
      $("results").classList.remove("hidden");
      $("metricVal").textContent = "–"; $("metricLabel").textContent = ""; $("metricDetail").textContent = "";
      $("samples").innerHTML = ""; $("samplesDesc").innerHTML = "";
      $("progressText").textContent = `0/${ev.total_epochs} · 0%`;
    } else if (ev.type === "checkpoint") {
      live.checkpoints.push(ev);
      $("ckSlider").max = live.checkpoints.length - 1;
      $("ckSlider").value = live.checkpoints.length - 1;
      // Render this checkpoint's generations + weights LIVE so text quality
      // is visibly improving while training is still running.
      renderMetricAndSamples(ev, ev.epoch);
      renderHeatmaps(ev);
    } else if (ev.type === "epoch") {
      live.loss_curve.push(ev.loss);
      setProgress(ev.epoch, live.total_epochs);
      $("liveLoss").innerHTML = `live loss: <b>${ev.loss.toFixed(4)}</b> ` +
        `<span style="color:var(--muted)">· grad&#8209;norm ${ev.gradnorm.toFixed(3)}</span>`;
      renderLiveChart(live);
    } else if (ev.type === "done") {
      DATA = { mode: ev.mode, dataset: ev.dataset, config: ev.config,
               vocab: ev.vocab, info: ev.info, params: ev.params,
               loss_curve: ev.loss_curve, gradnorm_curve: ev.gradnorm_curve,
               checkpoints: ev.checkpoints, final_loss: ev.final_loss };
      CK = DATA.checkpoints.length - 1;
      $("ckSlider").max = DATA.checkpoints.length - 1;
      renderInfo();
      renderCheckpoint();                 // switches to the interactive (scrubbable) chart
      setProgress(DATA.params.epochs, DATA.params.epochs);
      cacheResult();                      // remember for this browser session
      showDoneBanner(false);
      setTimeout(() => $("progressWrap").classList.add("hidden"), 600);
    } else if (ev.type === "error") {
      status.className = "trained-banner warn";
      status.innerHTML = "Error: " + ev.message;
      $("progressWrap").classList.add("hidden");
    }
  };

  try {
    const res = await fetch("/api/train_stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset: $("dataset").value,
        optimizer: $("optimizer").value,
        lr: parseFloat($("lr").value),
        epochs: parseInt($("epochs").value, 10),
        seed: parseInt($("seed").value, 10),
        d_model: parseInt($("d_model").value, 10),       // 0 => auto (dataset default)
        num_layers: parseInt($("num_layers").value, 10), // 0 => auto
        slice_chars: parseInt($("slice_chars").value, 10),
        block_size: parseInt($("block_size").value, 10),
        balance_alpha: parseFloat($("balance_alpha").value),
      }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);

    // Parse the Server-Sent Events stream incrementally.
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (raw.startsWith("data:")) handle(JSON.parse(raw.slice(5).trim()));
      }
    }
  } catch (err) {
    status.className = "trained-banner warn";
    status.innerHTML = "Error: " + err;
    $("progressWrap").classList.add("hidden");
  } finally {
    btn.disabled = false; btn.textContent = "Train model";
  }
}

/* ---------- wiring ---------- */
function syncDatasetUI() {
  const isChar = $("dataset").value === "shakespeare";
  document.querySelectorAll(".char-only").forEach(el => el.style.display = isChar ? "" : "none");
}

/* Set a control's value and refresh its label (fires the bound listener). */
function setControl(id, value) {
  const el = $(id);
  if (el === null || value === undefined || value === null) return;
  el.value = value;
  el.dispatchEvent(new Event("input"));
  el.dispatchEvent(new Event("change"));
}

/* Restore the cached training result without retraining. Returns true if a
 * cached result was found and rendered. */
function restoreFromCache() {
  const cached = loadCachedResult();
  if (!cached || !cached.checkpoints || !cached.checkpoints.length) return false;
  DATA = cached;

  // Show the cached trained result, but keep the controls on the BEST defaults
  // (not the cached run's params), so the page always loads ready-to-train at
  // the best settings. Pressing "Train model" then uses the best config.
  for (const [id, val] of Object.entries(BEST_DEFAULTS)) setControl(id, val);
  syncDatasetUI();

  CK = DATA.checkpoints.length - 1;
  $("ckSlider").max = DATA.checkpoints.length - 1;
  $("results").classList.remove("hidden");
  renderInfo();
  renderCheckpoint();
  showDoneBanner(true);
  return true;
}

/* Reset every control to the best Tiny Shakespeare settings. Does not retrain
 * on its own -- the user can then press "Train model". */
function resetToBest() {
  stopPlay();
  for (const [id, val] of Object.entries(BEST_DEFAULTS)) setControl(id, val);
  syncDatasetUI();
  const status = $("trainStatus");
  status.className = "trained-banner ok";
  status.innerHTML = "&#10003; Controls reset to the <b>best Tiny Shakespeare</b> settings " +
    "(d_model 32 · 2 layers · lr 0.03 · 170 epochs). Press <b>Train model</b> to run.";
}

window.addEventListener("DOMContentLoaded", () => {
  const bind = (id, lab, d) => {
    const fn = () => $(lab).textContent = d === 0
      ? parseFloat($(id).value).toFixed(0) : parseFloat($(id).value).toFixed(d);
    $(id).addEventListener("input", fn); fn();
  };
  bind("lr", "lrVal", 3);
  bind("epochs", "epochsVal", 0);
  bind("seed", "seedVal", 0);
  bind("slice_chars", "sliceVal", 0);
  bind("block_size", "blockVal", 0);

  // d_model / layers: 0 means "use the dataset's default".
  const bindAuto = (id, lab) => {
    const fn = () => { const v = parseInt($(id).value, 10); $(lab).textContent = v === 0 ? "auto" : v; };
    $(id).addEventListener("input", fn); fn();
  };
  bindAuto("d_model", "dmodelVal");
  bindAuto("num_layers", "layersVal");

  // load-balance alpha: 0 means "off"
  const balFn = () => { const v = parseFloat($("balance_alpha").value);
    $("balanceVal").textContent = v === 0 ? "off" : v.toFixed(2); };
  $("balance_alpha").addEventListener("input", balFn); balFn();

  $("dataset").addEventListener("change", syncDatasetUI);
  syncDatasetUI();

  $("trainBtn").addEventListener("click", train);
  $("resetBtn").addEventListener("click", resetToBest);
  $("prevCk").addEventListener("click", () => { stopPlay(); if (CK > 0) { CK--; renderCheckpoint(); } });
  $("nextCk").addEventListener("click", () => { stopPlay(); if (CK < DATA.checkpoints.length - 1) { CK++; renderCheckpoint(); } });
  $("playCk").addEventListener("click", () => { timer ? stopPlay() : play(); });
  $("ckSlider").addEventListener("input", () => { stopPlay(); CK = parseInt($("ckSlider").value, 10); renderCheckpoint(); });

  // Restore the last trained model from this session if present; otherwise
  // train once with defaults. This prevents retraining when navigating back
  // and forth between the Inference and Training pages.
  if (!restoreFromCache()) {
    train();
  }
});
