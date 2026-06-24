"use strict";

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */
let DATA = null;     // full response from /api/infer
let STEP = 0;        // current step index into DATA.steps
let timer = null;    // autoplay timer

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Rendering helpers                                                   */
/* ------------------------------------------------------------------ */
function maxAbs(nested) {
  let m = 0;
  const walk = (x) => {
    if (Array.isArray(x)) x.forEach(walk);
    else m = Math.max(m, Math.abs(x));
  };
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

function vec(arr, m, d = 2) {
  m = m || maxAbs(arr);
  return `<div class="vec">` + arr.map(v =>
    `<span class="cell" style="${heat(v, m)}">${fmt(v, d)}</span>`).join("") + `</div>`;
}

function matrix(mat, rowLabels, colLabels, d = 2) {
  const m = maxAbs(mat);
  let html = `<table class="mat"><thead><tr><th></th>`;
  const cols = colLabels || mat[0].map((_, i) => i);
  html += cols.map(c => `<th>${c}</th>`).join("") + `</tr></thead><tbody>`;
  mat.forEach((row, r) => {
    const rl = rowLabels ? rowLabels[r] : r;
    html += `<tr><td class="rowhead">${rl}</td>` +
      row.map(v => `<td style="${heat(v, m)}">${fmt(v, d)}</td>`).join("") + `</tr>`;
  });
  return html + `</tbody></table>`;
}

function card(title, badge, desc, body) {
  const b = badge ? `<span class="badge">${badge}</span>` : "";
  return `<div class="card"><h2>${title}${b}</h2>` +
    (desc ? `<p class="desc">${desc}</p>` : "") + body + `</div>`;
}

/* ------------------------------------------------------------------ */
/* Stage renderers                                                     */
/* ------------------------------------------------------------------ */
function renderTokenize(step) {
  const rows = step.tokens.map((t, i) =>
    `<tr><td class="rowhead">${i}</td>` +
    `<td style="text-align:left;padding-left:10px">${t.text}${t.oov ? ' <span class="badge">OOV&rarr;&lt;unk&gt;</span>' : ""}</td>` +
    `<td>${t.id}</td></tr>`).join("");
  const body = `
    <table class="mat" style="font-size:13px">
      <thead><tr><th>pos</th><th style="text-align:left;padding-left:10px">token</th><th>id</th></tr></thead>
      <tbody>${rows}</tbody></table>`;
  return card("1 · Tokenization", `${step.tokens.length} tokens`,
    "The prompt is split into tokens and mapped to integer ids. <code>&lt;bos&gt;</code> marks the start; unknown words fall back to <code>&lt;unk&gt;</code>.",
    body);
}

function renderEmbed(e) {
  const body = `
    <p class="sub">predicting the token that follows &middot; current token <b>${e.token}</b> (id ${e.token_id}, position ${e.pos})</p>
    <div class="grid cols-3">
      <div><p class="sub">token embedding</p>${vec(e.embedding)}</div>
      <div><p class="sub">positional encoding (pos ${e.pos})</p>${vec(e.positional)}</div>
      <div><p class="sub">x = embed + pos</p>${vec(e.summed)}</div>
    </div>`;
  return card("Embedding", "d_model = 8",
    "The token id indexes the embedding table; a sinusoidal positional encoding is added so the model knows token order.",
    body);
}

function renderAttention(a) {
  let heads = "";
  a.heads.forEach((h, hi) => {
    // table: rows = cached positions, cols = K dims + score + weight bar
    const m = maxAbs(h.K);
    let rows = "";
    h.K.forEach((krow, pi) => {
      const wpct = (h.weights[pi] * 100).toFixed(0);
      rows += `<tr><td class="rowhead">${a.cache_tokens[pi]}</td>` +
        krow.map(v => `<td style="${heat(v, m)}">${fmt(v)}</td>`).join("") +
        `<td>${fmt(h.scores[pi])}</td>` +
        `<td style="text-align:left"><span class="attn-bar" style="width:${Math.max(2, wpct * 0.6)}px"></span> ${(h.weights[pi]).toFixed(2)}</td></tr>`;
    });
    const kcols = h.K[0].map((_, i) => "k" + i);
    heads += `
      <div class="expert-card">
        <p class="sub">head ${hi} &middot; q = ${h.q.map(v => v.toFixed(2)).join(", ")}</p>
        <table class="mat"><thead><tr><th>cache</th>${kcols.map(c => `<th>${c}</th>`).join("")}<th>score</th><th>softmax weight</th></tr></thead>
        <tbody>${rows}</tbody></table>
      </div>`;
  });

  const body = `
    <div class="grid cols-3" style="margin-bottom:14px">
      <div><p class="sub">Q = x·Wq</p>${vec(a.q)}</div>
      <div><p class="sub">K = x·Wk (appended to cache)</p>${vec(a.k)}</div>
      <div><p class="sub">V = x·Wv (appended to cache)</p>${vec(a.v)}</div>
    </div>
    <p class="sub">KV cache &middot; ${a.cache_len} cached positions &middot; scores = q·kᵀ / √d, then softmax</p>
    <div class="grid cols-2">${heads}</div>
    <div class="flow-arrow">&darr; weighted sum of V, project (Wo), add residual</div>
    <p class="sub">hidden after attention</p>${vec(a.hidden)}`;
  return card("Self-Attention <span class='badge'>KV cache</span>", "2 heads",
    "Q/K/V are projected from x. The new K,V are <b>appended to the KV cache</b> so previous tokens are never recomputed — this is what makes decoding fast. Each head attends over all cached positions.",
    body);
}

function renderMoE(mo, model) {
  // router gate bars
  let gates = "";
  mo.gates.forEach((g, i) => {
    const sel = mo.selected.includes(i);
    gates += `<div class="gate-row"><span class="ename">expert ${i}</span>` +
      `<div class="gate-track"><div class="gate-fill ${sel ? "" : "unsel"}" style="width:${(g * 100).toFixed(1)}%"></div></div>` +
      `<span>${g.toFixed(3)}${sel ? " ✓" : ""}</span></div>`;
  });

  // expert cards
  let experts = "";
  for (let i = 0; i < model.config.num_experts; i++) {
    const out = mo.expert_outputs.find(o => o.expert === i);
    if (out) {
      experts += `<div class="expert-card selected">
        <p class="sub">expert ${i} &middot; gate ${out.gate} (renormalized)</p>
        <p class="sub">ReLU(x·W1) → hidden (16)</p>${vec(out.hidden)}
        <p class="sub">hidden·W2 → out (8)</p>${vec(out.out)}</div>`;
    } else {
      experts += `<div class="expert-card dim">
        <p class="sub">expert ${i}</p>
        <div style="color:var(--muted);padding:14px 4px">skipped &mdash; not in top&#8209;2<br/>(saves compute)</div></div>`;
    }
  }

  const body = `
    <p class="sub">router logits = hidden · W_router → softmax over ${model.config.num_experts} experts</p>
    <div class="gate-bar-wrap">${gates}</div>
    <div class="flow-arrow">&darr; keep top&#8209;2, renormalize their gates, run only those experts</div>
    <div class="grid cols-2">${experts}</div>
    <div class="flow-arrow">&darr; Σ gateₑ · outₑ , add residual</div>
    <p class="sub">hidden after MoE</p>${vec(mo.hidden)}`;
  return card("Mixture&#8209;of&#8209;Experts FFN", `top&#8209;2 of ${model.config.num_experts}`,
    "Instead of one feed-forward block, a <b>router</b> scores every expert. Only the top-2 experts run; their gate weights are renormalized and their outputs are combined. This is sparse computation — the heart of MoE.",
    body);
}

function renderSampling(s) {
  const probs = s.probs;
  const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
  const show = order.slice(0, 12);
  const keptTopp = new Set(s.kept_topp);
  const keptTopk = new Set(s.kept_topk);

  let rows = "";
  show.forEach(([p, i]) => {
    let state = "cut", label = "cut by top-k";
    if (i === s.chosen_id) { state = "chosen"; label = "sampled"; }
    else if (keptTopp.has(i)) { state = "kept"; label = "kept"; }
    else if (keptTopk.has(i)) { state = "cut"; label = "cut by top-p"; }
    const cls = state === "chosen" ? "chosen" : (state === "cut" ? "cut" : "");
    const fill = state === "chosen" ? "chosen" : (state === "kept" ? "" : "cut");
    rows += `<div class="prob-row ${cls}"><span class="pname">${DATA.model.vocab[i]}</span>` +
      `<div class="prob-track"><div class="prob-fill ${fill}" style="width:${Math.max(1, p * 100).toFixed(1)}%"></div></div>` +
      `<span class="pval">${p.toFixed(3)} <small style="color:var(--muted)">${label}</small></span></div>`;
  });

  const body = `
    <p class="sub">temperature ${s.params.temperature} &middot; top&#8209;k ${s.params.top_k} &middot; top&#8209;p ${s.params.top_p}</p>
    <p class="desc" style="margin-bottom:10px">logits → divide by temperature → softmax → keep top&#8209;k → keep nucleus (top&#8209;p) → renormalize → sample.</p>
    <div class="legend">
      <span><i style="background:var(--accent)"></i>kept (in nucleus)</span>
      <span><i style="background:#39424f"></i>filtered out (top&#8209;k / top&#8209;p)</span>
      <span><i style="background:var(--green)"></i>sampled token</span>
    </div>
    <div>${rows}</div>
    <div class="chosen-pill">sampled → <b>${s.chosen_token}</b> (id ${s.chosen_id})</div>`;
  return card("Output &amp; Sampling", "vocab = " + DATA.model.vocab.length,
    "The final hidden state is projected to a logit per vocabulary token. Temperature, top&#8209;k and top&#8209;p (nucleus) reshape the distribution before a token is sampled.",
    body);
}

/* ------------------------------------------------------------------ */
/* Step rendering & navigation                                         */
/* ------------------------------------------------------------------ */
function renderStep() {
  const stage = $("stage");
  const step = DATA.steps[STEP];
  let html = "";

  if (step.kind === "tokenize") {
    html = renderTokenize(step);
  } else {
    html += renderEmbed(step.embed);
    const nL = step.layers.length;
    step.layers.forEach((lr) => {
      html += `<div class="flow-arrow">&darr;</div>`;
      if (nL > 1) {
        html += `<div class="layer-band"><span class="layer-tag">Layer ${lr.layer}</span>` +
          `<span class="layer-sub">transformer&#8209;MoE block ${lr.layer + 1} of ${nL}</span></div>`;
      }
      html += renderAttention(lr.attention);
      html += `<div class="flow-arrow">&darr;</div>`;
      html += renderMoE(lr.moe, DATA.model);
    });
    html += `<div class="flow-arrow">&darr;${nL > 1 ? " final layer output &rarr; unembed" : ""}</div>`;
    html += renderSampling(step.sampling);
  }
  stage.innerHTML = html;

  updateStepper();
  updateSequence();
}

function updateStepper() {
  const step = DATA.steps[STEP];
  $("stepLabel").textContent = `Step ${STEP + 1} / ${DATA.steps.length} · ${step.title}`;
  $("prevBtn").disabled = STEP === 0;
  $("nextBtn").disabled = STEP === DATA.steps.length - 1;

  const track = $("stepTrack");
  track.innerHTML = "";
  DATA.steps.forEach((s, i) => {
    const d = document.createElement("div");
    d.className = "dot" + (i === STEP ? " active" : (i < STEP ? " done" : ""));
    d.textContent = i;
    d.title = s.title;
    d.onclick = () => { stopPlay(); STEP = i; renderStep(); };
    track.appendChild(d);
  });
}

function updateSequence() {
  const seq = $("sequence");
  seq.innerHTML = "";
  // prompt tokens
  DATA.tokens.forEach(t => {
    const span = document.createElement("span");
    span.className = "tok prompt" + (t.text === "<bos>" ? " bos" : "");
    span.textContent = t.text;
    seq.appendChild(span);
  });
  const arrow = document.createElement("span");
  arrow.className = "arrow"; arrow.textContent = "→";
  seq.appendChild(arrow);

  // how many generated tokens have been revealed by the current step
  let genRevealed = 0;
  for (let i = 1; i <= STEP; i++) {
    if (DATA.steps[i] && DATA.steps[i].kind === "generate") genRevealed++;
  }
  DATA.generated.forEach((g, i) => {
    const span = document.createElement("span");
    const active = i === genRevealed - 1;
    span.className = "tok gen" + (active ? " active" : "");
    span.style.opacity = i < genRevealed ? "1" : "0.28";
    span.textContent = g.text;
    seq.appendChild(span);
  });
}

/* ------------------------------------------------------------------ */
/* Weight tables                                                       */
/* ------------------------------------------------------------------ */
function renderWeights() {
  const m = DATA.model;
  const d = m.config.d_model, h = m.config.expert_hidden;
  const dCols = Array.from({ length: d }, (_, i) => "d" + i);
  // Make whitespace characters visible as row labels in char mode.
  const ws = { " ": "␠", "\n": "↵", "\t": "⇥" };
  const rowLabels = m.vocab.map(t => ws[t] !== undefined ? ws[t] : t);
  let html = "";
  html += card("Embedding table", `${m.config.vocab_size} × ${d}`,
    "Each row is the learned vector for one vocabulary token (shared across layers).",
    `<div class="heat-scroll">${matrix(m.embed, rowLabels, dCols)}</div>`);

  const nL = m.layers.length;
  m.layers.forEach((layer, li) => {
    const pre = nL > 1 ? `Layer ${li} · ` : "";
    if (nL > 1) {
      html += `<div class="layer-band"><span class="layer-tag">Layer ${li}</span>` +
        `<span class="layer-sub">block ${li + 1} of ${nL}</span></div>`;
    }
    html += card(`${pre}Router weights`, `${d} × ${m.config.num_experts}`,
      "Projects the hidden state to one score per expert.",
      matrix(layer.W_router, dCols, layer.experts.map((_, i) => "E" + i)));
    layer.experts.forEach((e, i) => {
      html += card(`${pre}Expert ${i} weights`, `W1 ${d}×${h} · W2 ${h}×${d}`, "",
        `<div class="grid cols-2">
          <div><p class="sub">W1 (in → hidden)</p><div class="heat-scroll">${matrix(e.W1, dCols)}</div></div>
          <div><p class="sub">W2 (hidden → out)</p><div class="heat-scroll">${matrix(e.W2, null, dCols)}</div></div>
        </div>`);
    });
  });
  $("weightsBody").innerHTML = html;
}

/* ------------------------------------------------------------------ */
/* Autoplay                                                            */
/* ------------------------------------------------------------------ */
function play() {
  if (STEP >= DATA.steps.length - 1) STEP = -1;
  $("playBtn").innerHTML = "&#10073;&#10073; Pause";
  timer = setInterval(() => {
    if (STEP >= DATA.steps.length - 1) { stopPlay(); return; }
    STEP++; renderStep();
  }, 1600);
}
function stopPlay() {
  if (timer) { clearInterval(timer); timer = null; }
  $("playBtn").innerHTML = "&#9654; Play";
}

/* ------------------------------------------------------------------ */
/* Run                                                                 */
/* ------------------------------------------------------------------ */
async function run() {
  stopPlay();
  const btn = $("runBtn");
  btn.disabled = true; btn.textContent = "Running…";
  try {
    const res = await fetch("/api/infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: $("prompt").value,
        temperature: parseFloat($("temperature").value),
        top_k: parseInt($("top_k").value, 10),
        top_p: parseFloat($("top_p").value),
        max_tokens: parseInt($("max_tokens").value, 10),
        seed: parseInt($("seed").value, 10),
        use_trained: $("useTrained").checked,
      }),
    });
    DATA = await res.json();
    STEP = 0;
    $("playBtn").disabled = false;
    updateArchPill();
    updateTrainedBanner();
    renderWeights();
    renderStep();
  } catch (err) {
    $("stage").innerHTML = `<div class="hint">Error: ${err}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "Run inference";
  }
}

function updateArchPill() {
  const c = DATA.model.config;
  const layers = c.num_layers === 1 ? "1 layer" : `${c.num_layers} layers`;
  $("archPill").innerHTML =
    `${layers}&nbsp;&middot;&nbsp;d=${c.d_model}&nbsp;&middot;&nbsp;${c.num_heads}&nbsp;heads&nbsp;&middot;&nbsp;${c.num_experts}&nbsp;experts&nbsp;(top&#8209;2)`;
}

function updateTrainedBanner() {
  const b = $("trainedBanner");
  if (DATA && DATA.trained) {
    b.className = "trained-banner ok";
    const isChar = DATA.model && DATA.model.mode === "char";
    const tip = isChar
      ? "Char&#8209;level model (trained on Tiny Shakespeare). Type any text &mdash; it continues <b>character by character</b>. Lower temperature / top&#8209;k=1 for the model's best guess."
      : "Try the prompt <code>the cat sat on the</code> &mdash; it should now favour <code>mat</code>.";
    b.innerHTML = "&#10003; Running with <b>trained weights</b> (learned on the Training page). " + tip;
  } else if ($("useTrained").checked) {
    b.className = "trained-banner warn";
    b.innerHTML = "No trained weights found yet. Go to the <a href='/train'>Training</a> page and run training first.";
  } else {
    b.className = "trained-banner hidden";
    b.innerHTML = "";
  }
}

async function refreshStatus() {
  try {
    const s = await (await fetch("/api/status")).json();
    const cb = $("useTrained");
    if (s.has_trained_weights) {
      cb.disabled = false;
      cb.parentElement.classList.remove("disabled");
      cb.parentElement.title = "Load weights saved from the Training page";
    } else {
      cb.parentElement.classList.add("disabled");
      cb.parentElement.title = "No trained weights yet — run training first";
    }
  } catch (e) { /* ignore */ }
}

/* ------------------------------------------------------------------ */
/* Wiring                                                              */
/* ------------------------------------------------------------------ */
function bindSlider(id, labelId, d) {
  const fn = () => {
    const v = parseFloat($(id).value);
    $(labelId).textContent = d === 0 ? v.toFixed(0) : v.toFixed(d);
  };
  $(id).addEventListener("input", fn); fn();
}

/* Best sampling settings — light temperature + top-k that surface the model's
 * learned distribution as flowing text instead of greedy repetition loops
 * (top-p left open so top-k does the trimming). */
const BEST_SAMPLING = { temperature: 0.8, top_k: 10, top_p: 0.9, max_tokens: 10, seed: 0 };

function resetToBest() {
  stopPlay();
  for (const [id, val] of Object.entries(BEST_SAMPLING)) {
    const el = $(id);
    el.value = val;
    el.dispatchEvent(new Event("input"));   // refresh the slider's label
  }
  run();   // re-run so the effect is visible immediately
}

window.addEventListener("DOMContentLoaded", () => {
  bindSlider("temperature", "tempVal", 2);
  bindSlider("top_k", "topkVal", 0);
  bindSlider("top_p", "toppVal", 2);
  bindSlider("max_tokens", "maxVal", 0);
  bindSlider("seed", "seedVal", 0);

  $("runBtn").addEventListener("click", run);
  $("resetBtn").addEventListener("click", resetToBest);
  $("useTrained").addEventListener("change", run);
  $("prompt").addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  refreshStatus();
  $("prevBtn").addEventListener("click", () => { stopPlay(); if (STEP > 0) { STEP--; renderStep(); } });
  $("nextBtn").addEventListener("click", () => { stopPlay(); if (STEP < DATA.steps.length - 1) { STEP++; renderStep(); } });
  $("playBtn").addEventListener("click", () => { timer ? stopPlay() : play(); });
  $("toggleWeights").addEventListener("click", () => {
    const body = $("weightsBody");
    body.classList.toggle("hidden");
    $("toggleWeights").innerHTML = body.classList.contains("hidden")
      ? "Show model weight tables &#9662;" : "Hide model weight tables &#9652;";
  });

  run(); // auto-run with defaults on load
});
