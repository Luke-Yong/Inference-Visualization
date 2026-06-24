# LLM Inference & Training Visualizer (MoE)

An interactive, full-stack web app that makes the inner workings of a
**Mixture-of-Experts (MoE) transformer** visible — both the **inference**
forward pass (prompt → tokens → attention → experts → sampling) and the
**training** loop (forward → loss → backprop → gradient step), all running on a
**real, tiny model with genuine hand-written backpropagation** (pure NumPy, no
PyTorch).

The model is intentionally small so that *every* intermediate tensor can be
shown as a readable table/heatmap. The goal is to expose the **mechanics** of
modern LLMs, not to produce coherent text.

> English | [中文](./README.zh-CN.md)

---

## What it does

### Inference page (`/`)
Step through a real forward pass, one stage at a time:

1. **Tokenization** — the prompt is split into tokens and mapped to ids.
2. **Embedding (+ positional encoding)** — each token id becomes a vector.
3. **Self-Attention with a KV cache** — Q/K/V projections, per-head scores,
   softmax weights; new K/V are *appended to the cache* so past tokens aren't
   recomputed (the real reason decoding is fast).
4. **Mixture-of-Experts FFN** — a **router** scores all experts, only the
   **top-2 of 4** run, their gates are renormalized and outputs combined.
5. **Output & Sampling** — logits → temperature → top-k → top-p (nucleus) →
   sample the next token.

Multi-layer models show one **attention + MoE block per layer**, stacked. Every
vector/matrix is a blue/red heatmap. You can step/play through the trace, tune
the sampling controls, inspect the raw weight tables, and tick **“Use trained
weights”** to run the model you trained.

### Training page (`/train`)
Watch the same model actually **learn**, live:

- A genuine cross-entropy next-token loss is minimized with **real
  backpropagation** (Adam/SGD), streamed over **Server-Sent Events** so the
  **loss curve, progress bar, and sample generations animate in real time**.
- Scrub through checkpoints to see the **loss drop**, **next-char accuracy
  climb**, the **embedding heatmap drift** from its random start, and the
  **router/expert-usage specialize**.
- Two datasets: a tiny **word-level corpus** (maximally legible tables) and
  **character-level Tiny Shakespeare** (real English; downloaded on first run,
  cached locally, with an offline fallback).
- Tune dataset, optimizer, learning rate, epochs, model width (`d_model`),
  depth (`layers`), context window, etc. **“Reset to best”** restores the tuned
  defaults; results are **cached per browser session** so navigating between
  pages doesn’t retrain.

---

## Who it helps

- **Students & self-learners** building intuition for how transformers/LLMs
  actually compute — seeing tensors, attention, and routing instead of just
  reading equations.
- **Educators** who want a live, clickable artifact to teach attention, the
  KV cache, MoE routing, temperature/top-k/top-p sampling, and the
  forward/backward training loop.
- **Engineers new to MoE** who want a faithful, minimal mental model of expert
  routing and sparse computation before touching production frameworks.
- **Anyone curious** about what “the model is generating token by token” really
  means under the hood.

It is **not** a production model or a way to get coherent text — the weights are
tiny. It’s a teaching/visualization tool.

---

## How it works (architecture)

```
Browser (HTML/CSS/JS)  ←→  Flask (app.py)  ←→  NumPy model (engine.py / trainer.py)
   step-by-step UI          JSON + SSE              real forward + backprop
```

The model is a stack of `num_layers` transformer-MoE blocks:

```
ids → embed (+ positional)
    → [ Self-Attention (causal, multi-head, KV cache) + residual
        → MoE FFN (router → top-2 experts → weighted sum)    + residual ] × num_layers
    → unembed → logits → sampling
```

- **Experts** are identical small 2-layer MLPs (`W1 → ReLU → W2`); what makes
  them differ is their **weights** — random at init, then **specialized by
  training** under the router’s gating.
- **Training simplifications** (standard and explained in-app): the whole
  sequence is processed at once with a causal mask (the KV cache is an
  *inference* optimization), and the router uses **soft routing** (all experts,
  differentiable) during training while inference keeps the **top-2**.
- Backprop is **hand-written and numerically gradient-checked** (no autograd),
  so it runs on a plain NumPy install — including Python 3.14, which has no
  PyTorch wheels.

### Project layout
| File | Role |
|---|---|
| [`app.py`](./app.py) | Flask server: pages + `/api/infer`, `/api/train_stream` (SSE), `/api/status`, `/api/reset_weights` |
| [`engine.py`](./engine.py) | Inference model: multi-layer forward, KV cache, top-2 MoE, sampling, full step trace, trained-weight loading |
| [`trainer.py`](./trainer.py) | Trainable model + manual backprop + Adam/SGD; `train_stream` generator; saves weights + meta |
| [`dataset.py`](./dataset.py) | Word corpus + char-level Tiny Shakespeare (download/cache + tokenizers) |
| `templates/`, `static/` | Frontend (inference + training pages, styles, visualization JS) |
| `trained_weights.npz`, `trained_meta.json` | Saved model after training (config + vocab + weights) |

---

## Getting started

Requirements: **Python 3.10+** and `pip`. Only **Flask** and **NumPy** are
needed (no PyTorch).

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000**.

- `/` — Inference visualizer
- `/train` — Training visualizer (auto-trains Tiny Shakespeare on first load;
  downloads ~1 MB of text once, then caches it under `data/`)

Typical flow: open **/train**, watch it learn (or press **Train model**), then
open **/** and tick **“Use trained weights”** to run the model you just trained.

> Char-level training on Tiny Shakespeare takes ~1–2 minutes on the dev server;
> it’s streamed (you watch progress) and cached per session.

---

## Notes & honest limitations

- Output is **not** coherent prose — the model is deliberately tiny so the
  tables stay readable. Char-level on Tiny Shakespeare learns spelling/letter
  statistics and pseudo-Shakespearean fragments, not real sentences.
- Expert **specialization is mild** with only 4 tiny experts on a small
  dataset; the *mechanism* is faithful, the *degree* is limited by size.
- The bundled dev server is for local use, not production.

---

## License

[MIT](./LICENSE).
