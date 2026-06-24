"""
Training for the toy MoE language model -- config, dataset & DEPTH driven.

The architecture mirrors `engine.ToyMoEModel`: a stack of `num_layers`
transformer-MoE blocks (each = causal self-attention + Mixture-of-Experts FFN,
both with residual connections), then an unembedding to logits. Every size is a
parameter (d_model, num_layers, vocab ...), so the same code trains:
  * the tiny word-level corpus, or
  * character-level Tiny Shakespeare (larger d_model / more layers).

Training uses REAL hand-written backpropagation (pure numpy, no autograd):
gradients of the cross-entropy next-token loss flow back through unembed and
then through every layer in reverse (MoE -> attention -> residual) down to the
embedding table.

Standard training-time choices:
  * the whole sequence is processed at once with a causal mask
    (a KV cache is an inference-time optimization, not needed for training);
  * the router uses SOFT routing (all experts, weighted by softmax) so it is
    differentiable. Inference keeps only the top-2 experts for efficiency.

After training, weights are saved to `trained_weights.npz` and a JSON sidecar
`trained_meta.json` records the full config (incl. num_layers) + vocabulary +
tokenizer mode, so the inference page rebuilds the exact same model.
"""

import json

import numpy as np

import engine
import dataset as ds

WEIGHTS_PATH = engine.WEIGHTS_PATH
META_PATH = engine.WEIGHTS_META


def _round(arr, n=3):
    return np.round(np.asarray(arr, dtype=float), n).tolist()


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def positional_encoding(pos, d_model):
    pe = np.zeros(d_model)
    for i in range(0, d_model, 2):
        denom = 10000 ** (i / d_model)
        pe[i] = np.sin(pos / denom)
        if i + 1 < d_model:
            pe[i + 1] = np.cos(pos / denom)
    return pe


class MoELM:
    """Trainable multi-layer MoE language model with configurable sizes."""

    def __init__(self, vocab_size, d_model=8, num_heads=2, num_experts=4,
                 expert_hidden=16, num_layers=1, max_len=256, seed=42):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        rng = np.random.default_rng(seed)

        def randn(*shape, scale):
            return rng.standard_normal(shape) * scale

        # ~1/sqrt(fan_in) init so initial logits stay O(1) and loss starts near
        # ln(vocab_size) -- crucial for a visible (non-exploding) learning curve.
        s = 1.0 / np.sqrt(d_model)
        sh = 1.0 / np.sqrt(expert_hidden)

        self.P = {
            "embed": randn(vocab_size, d_model, scale=0.4),
            "W_unembed": randn(d_model, vocab_size, scale=s),
        }
        for l in range(num_layers):
            p = f"L{l}_"
            self.P[p + "Wq"] = randn(d_model, d_model, scale=s)
            self.P[p + "Wk"] = randn(d_model, d_model, scale=s)
            self.P[p + "Wv"] = randn(d_model, d_model, scale=s)
            self.P[p + "Wo"] = randn(d_model, d_model, scale=s)
            self.P[p + "W_router"] = randn(d_model, num_experts, scale=s)
            for e in range(num_experts):
                self.P[p + f"e{e}_W1"] = randn(d_model, expert_hidden, scale=s)
                self.P[p + f"e{e}_W2"] = randn(expert_hidden, d_model, scale=sh)

        self._pos = np.array([positional_encoding(p, d_model) for p in range(max_len)])

    def _prefix(self, l):
        return f"L{l}_"

    # -------------------------- per-layer forward -------------------------- #
    def _layer_forward(self, l, X, mask):
        """One transformer-MoE block. Returns (H2, layer_cache)."""
        P = self.P
        p = self._prefix(l)
        d, H, hd, E = self.d_model, self.num_heads, self.head_dim, self.num_experts
        T = X.shape[0]

        Q = X @ P[p + "Wq"]; K = X @ P[p + "Wk"]; V = X @ P[p + "Wv"]
        O = np.zeros((T, d))
        attn = []
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            Qh, Kh, Vh = Q[:, sl], K[:, sl], V[:, sl]
            S = (Qh @ Kh.T) / np.sqrt(hd)
            S = np.where(mask, -1e9, S)
            A = softmax(S, axis=1)
            O[:, sl] = A @ Vh
            attn.append({"Qh": Qh, "Kh": Kh, "Vh": Vh, "A": A})

        Pp = O @ P[p + "Wo"]
        H1 = X + Pp                                    # residual

        R = H1 @ P[p + "W_router"]
        G = softmax(R, axis=1)
        experts = []
        M = np.zeros((T, d))
        for e in range(E):
            Z = H1 @ P[p + f"e{e}_W1"]
            Hd = np.maximum(0.0, Z)
            U = Hd @ P[p + f"e{e}_W2"]
            M += G[:, e:e + 1] * U
            experts.append({"Z": Z, "Hd": Hd, "U": U})
        H2 = H1 + M                                    # residual

        lc = {"X": X, "O": O, "attn": attn, "H1": H1, "G": G, "experts": experts}
        return H2, lc

    def forward(self, ids, cache=None):
        P = self.P
        T = len(ids)
        X = np.array([P["embed"][i] for i in ids]) + self._pos[:T]
        mask = np.triu(np.ones((T, T)), k=1).astype(bool)

        H = X
        layer_caches = []
        for l in range(self.num_layers):
            H, lc = self._layer_forward(l, H, mask)
            layer_caches.append(lc)

        logits = H @ P["W_unembed"]
        if cache is not None:
            cache.update(dict(ids=ids, T=T, layer_caches=layer_caches, H_top=H))
        return logits

    # -------------------------- per-layer backward -------------------------- #
    def _layer_backward(self, l, lc, dH2, g):
        """Backprop one block. dH2 = grad wrt layer output. Returns dX (grad
        wrt layer input) and accumulates this layer's param grads into g."""
        P = self.P
        p = self._prefix(l)
        H, hd, E, d = self.num_heads, self.head_dim, self.num_experts, self.d_model
        T = dH2.shape[0]

        # ---- MoE ----
        dH1 = dH2.copy()               # residual: H2 = H1 + M
        dM = dH2
        G = lc["G"]; H1 = lc["H1"]
        dG = np.zeros_like(G)
        for e in range(E):
            ex = lc["experts"][e]
            dU = dM * G[:, e:e + 1]
            dG[:, e] = np.sum(dM * ex["U"], axis=1)
            g[p + f"e{e}_W2"] += ex["Hd"].T @ dU
            dHd = dU @ P[p + f"e{e}_W2"].T
            dZ = dHd * (ex["Z"] > 0)
            g[p + f"e{e}_W1"] += H1.T @ dZ
            dH1 += dZ @ P[p + f"e{e}_W1"].T
        dR = G * (dG - np.sum(dG * G, axis=1, keepdims=True))   # softmax jac
        g[p + "W_router"] += H1.T @ dR
        dH1 += dR @ P[p + "W_router"].T

        # ---- attention ----
        dX = dH1.copy()                # residual: H1 = X + Pp
        dPp = dH1
        O = lc["O"]
        g[p + "Wo"] += O.T @ dPp
        dO = dPp @ P[p + "Wo"].T

        dQ = np.zeros((T, d)); dK = np.zeros((T, d)); dV = np.zeros((T, d))
        for h in range(H):
            sl = slice(h * hd, (h + 1) * hd)
            a = lc["attn"][h]
            A, Qh, Kh, Vh = a["A"], a["Qh"], a["Kh"], a["Vh"]
            dOh = dO[:, sl]
            dA = dOh @ Vh.T
            dV[:, sl] += A.T @ dOh
            dS = A * (dA - np.sum(dA * A, axis=1, keepdims=True))
            dS /= np.sqrt(hd)
            dQ[:, sl] += dS @ Kh
            dK[:, sl] += dS.T @ Qh
        X = lc["X"]
        g[p + "Wq"] += X.T @ dQ; dX += dQ @ P[p + "Wq"].T
        g[p + "Wk"] += X.T @ dK; dX += dK @ P[p + "Wk"].T
        g[p + "Wv"] += X.T @ dV; dX += dV @ P[p + "Wv"].T
        return dX

    def loss_and_grads(self, ids):
        cache = {}
        logits = self.forward(ids, cache)
        T = cache["T"]
        P = self.P

        probs = softmax(logits, axis=1)
        n = T - 1
        loss = 0.0
        dL = np.zeros_like(logits)
        for t in range(n):
            tgt = ids[t + 1]
            loss += -np.log(probs[t, tgt] + 1e-12)
            dL[t] = probs[t].copy()
            dL[t, tgt] -= 1.0
            dL[t] /= n
        loss /= n

        g = {k: np.zeros_like(v) for k, v in P.items()}

        # unembed
        g["W_unembed"] += cache["H_top"].T @ dL
        dH = dL @ P["W_unembed"].T

        # layers in reverse
        for l in reversed(range(self.num_layers)):
            dH = self._layer_backward(l, cache["layer_caches"][l], dH, g)

        # embedding (positional encoding is fixed)
        ids_arr = cache["ids"]
        for t in range(T):
            g["embed"][ids_arr[t]] += dH[t]

        return loss, g

    # -------------------------- generation / metrics -------------------------- #
    def greedy_continue(self, prompt_ids, n=4, eos_id=None, max_ctx=None):
        ids = list(prompt_ids)
        out = []
        for _ in range(n):
            ctx = ids if max_ctx is None else ids[-max_ctx:]
            logits = self.forward(ctx)
            nxt = int(np.argmax(logits[-1]))
            out.append(nxt)
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
        return out

    def sample_continue(self, prompt_ids, n, rng, temperature=0.8, top_k=10,
                        max_ctx=None):
        """Generate by temperature + top-k SAMPLING (not greedy).

        Greedy (argmax) decoding from a tiny char model degenerates into
        repetition loops ("the the the"). Sampling from the model's actual
        learned distribution produces more natural, flowing pseudo-text, which
        better reflects what the model learned. `rng` is passed in so the same
        seed gives reproducible samples across checkpoints.
        """
        ids = list(prompt_ids)
        out = []
        for _ in range(n):
            ctx = ids if max_ctx is None else ids[-max_ctx:]
            logits = self.forward(ctx)[-1] / max(temperature, 1e-3)
            probs = softmax(logits)
            if top_k and top_k < probs.shape[0]:
                keep = np.argsort(probs)[::-1][:top_k]
                masked = np.zeros_like(probs)
                masked[keep] = probs[keep]
                probs = masked / masked.sum()
            nxt = int(rng.choice(probs.shape[0], p=probs))
            out.append(nxt)
            ids.append(nxt)
        return out

    def next_token_prob(self, prompt_ids, target_id):
        logits = self.forward(prompt_ids)
        return float(softmax(logits[-1])[target_id])

    def next_token_accuracy(self, sequences):
        correct = total = 0
        for ids in sequences:
            if len(ids) < 2:
                continue
            logits = self.forward(ids)
            pred = np.argmax(logits[:-1], axis=1)
            tgt = np.array(ids[1:])
            correct += int(np.sum(pred == tgt))
            total += len(tgt)
        return correct / total if total else 0.0

    def avg_expert_usage(self, sequences, layer=0):
        totals = np.zeros(self.num_experts)
        count = 0
        for ids in sequences:
            cache = {}
            self.forward(ids, cache)
            G = cache["layer_caches"][layer]["G"]
            totals += G.sum(axis=0)
            count += G.shape[0]
        return (totals / max(count, 1)).tolist()

    # -------------------------- snapshot -------------------------- #
    def snapshot(self, data, sequences):
        usage = self.avg_expert_usage(sequences, layer=0)
        acc = self.next_token_accuracy(sequences)

        samples = []
        if data.mode == "word":
            for prompt, expected in data.eval_pairs():
                pids = data.encode(prompt)[:-1]
                gen = self.greedy_continue(pids, n=4,
                                           eos_id=data.token_to_id.get("<eos>"))
                exp_id = data.token_to_id[expected]
                samples.append({
                    "prompt": prompt,
                    "generated": " ".join(data.vocab[i] for i in gen),
                    "expected": expected,
                    "expected_prob": round(self.next_token_prob(pids, exp_id), 4),
                    "correct": bool(gen and gen[0] == exp_id),
                })
            metric = {"label": "next-token accuracy", "value": round(acc, 3),
                      "detail": f"{sum(s['correct'] for s in samples)}/{len(samples)} prompts correct"}
        else:
            # Sample (temp + top-k) instead of greedy so the displayed text
            # flows instead of looping. Fixed seed per snapshot -> the change
            # you see across checkpoints is the *model learning*, not RNG.
            for seed in data.seeds():
                pids = data.encode(seed)
                rng = np.random.default_rng(1234)
                gen = self.sample_continue(pids, n=40, rng=rng, temperature=0.8,
                                           top_k=10, max_ctx=data.block_len)
                samples.append({"prompt": seed, "generated": data.decode(gen)})
            metric = {"label": "next-char accuracy", "value": round(acc, 3),
                      "detail": f"predicting the correct next character {acc*100:.0f}% of the time"}

        return {
            "embed": _round(self.P["embed"]),
            "W_router": _round(self.P["L0_W_router"]),   # layer 0 router
            "expert_usage": [round(x, 4) for x in usage],
            "samples": samples,
            "metric": metric,
        }

    # -------------------------- io -------------------------- #
    def save(self, data):
        np.savez(WEIGHTS_PATH, **self.P)
        meta = {"mode": data.mode, "dataset": data.name, "vocab": list(data.vocab),
                "config": self.config()}
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def config(self):
        return {"d_model": self.d_model, "num_heads": self.num_heads,
                "head_dim": self.head_dim, "num_experts": self.num_experts,
                "expert_hidden": self.expert_hidden, "num_layers": self.num_layers,
                "vocab_size": self.vocab_size}


# ----------------------------- Adam ----------------------------- #
class Adam:
    def __init__(self, params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        gn = 0.0
        for k in params:
            gk = grads[k]
            gn += float(np.sum(gk * gk))
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * gk
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (gk * gk)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return np.sqrt(gn)


# ----------------------------- training run ----------------------------- #
# Per-dataset architecture defaults. The Shakespeare values were chosen by a
# parameter sweep (see commit notes): d_model=32 / 2 layers gave the best
# next-char accuracy (~0.69, loss ~1.03) without the runtime blow-up of deeper
# models -- 3 layers actually underfit and trained slower at the same epochs.
DATASET_DEFAULTS = {
    "word": dict(d_model=8, num_heads=2, num_experts=4, expert_hidden=16, num_layers=1),
    "shakespeare": dict(d_model=32, num_heads=2, num_experts=4, expert_hidden=32, num_layers=2),
}


def train_stream(dataset_name="word", epochs=60, lr=0.05, optimizer="adam",
                 seed=42, num_snapshots=8, d_model=None, num_layers=None,
                 slice_chars=3000, block_size=32):
    """Run training as a GENERATOR, yielding progress events as they happen.

    Event types (each is a plain dict, JSON-serializable):
      {"type":"start", ...}        once, with config/info/params/total_epochs
      {"type":"epoch", epoch, loss, gradnorm}   after every optimizer step
      {"type":"checkpoint", ...}   at snapshot epochs, includes model.snapshot()
      {"type":"done", ...}         once, the full aggregated payload (weights saved)

    This is what makes the UI show REAL progress: each epoch's loss is emitted
    the moment it is computed, so the client can animate the curve live instead
    of waiting for the whole run to finish.
    """
    epochs = int(epochs)
    data = ds.get_dataset(dataset_name, slice_chars=slice_chars, block_size=block_size)
    sequences = data.training_sequences()

    cfg = dict(DATASET_DEFAULTS.get(dataset_name, DATASET_DEFAULTS["word"]))
    if d_model:
        cfg["d_model"] = int(d_model)
    if num_layers:
        cfg["num_layers"] = int(num_layers)
    max_len = max(64, data.block_len + 4)
    model = MoELM(vocab_size=len(data.vocab), max_len=max_len, seed=int(seed), **cfg)

    opt = Adam(model.P, lr=float(lr)) if optimizer == "adam" else None
    sgd_lr = float(lr)

    if num_snapshots >= epochs + 1:
        ck = set(range(epochs + 1))
    else:
        ck = set(round(i * epochs / (num_snapshots - 1)) for i in range(num_snapshots))

    params = {"epochs": epochs, "lr": float(lr), "optimizer": optimizer,
              "seed": int(seed), "d_model": cfg["d_model"],
              "num_layers": cfg["num_layers"]}

    yield {"type": "start", "mode": data.mode, "dataset": data.name,
           "config": model.config(), "vocab": list(data.vocab),
           "info": data.info(), "params": params, "total_epochs": epochs}

    checkpoints, loss_curve, gradnorm_curve = [], [], []

    def make_ck(epoch, loss):
        return {"epoch": epoch, "loss": round(float(loss), 4),
                **model.snapshot(data, sequences)}

    # epoch 0 snapshot (before any update)
    init_loss = float(np.mean([model.loss_and_grads(ids)[0] for ids in sequences]))
    c0 = make_ck(0, init_loss)
    checkpoints.append(c0)
    yield {"type": "checkpoint", **c0}

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        accum = {k: np.zeros_like(v) for k, v in model.P.items()}
        for ids in sequences:
            loss, g = model.loss_and_grads(ids)
            total_loss += loss
            for k in accum:
                accum[k] += g[k]
        for k in accum:
            accum[k] /= len(sequences)
        avg_loss = total_loss / len(sequences)
        loss_curve.append(round(float(avg_loss), 4))

        if opt is not None:
            gn = opt.step(model.P, accum)
        else:
            gn = 0.0
            for k in model.P:
                gn += float(np.sum(accum[k] ** 2))
                model.P[k] -= sgd_lr * accum[k]
            gn = np.sqrt(gn)
        gradnorm_curve.append(round(float(gn), 4))

        # cheap per-epoch event -> drives the live curve + progress bar
        yield {"type": "epoch", "epoch": ep, "loss": round(float(avg_loss), 4),
               "gradnorm": round(float(gn), 4)}

        if ep in ck:
            c = make_ck(ep, avg_loss)
            checkpoints.append(c)
            yield {"type": "checkpoint", **c}

    model.save(data)

    yield {
        "type": "done",
        "mode": data.mode, "dataset": data.name, "config": model.config(),
        "vocab": list(data.vocab), "info": data.info(), "params": params,
        "loss_curve": loss_curve, "gradnorm_curve": gradnorm_curve,
        "checkpoints": checkpoints,
        "final_loss": loss_curve[-1] if loss_curve else None,
        "saved": True,
    }


def train_run(**kwargs):
    """Non-streaming convenience wrapper: drains train_stream, returns the
    final aggregated payload (handy for tests / CLI use)."""
    final = None
    for ev in train_stream(**kwargs):
        if ev["type"] == "done":
            final = {k: v for k, v in ev.items() if k != "type"}
    return final
