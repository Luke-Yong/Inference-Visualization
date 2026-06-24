"""
Toy Mixture-of-Experts (MoE) transformer used purely for *visualization*.

The model has REAL (but randomly initialized) weights and runs a genuine
autoregressive inference pipeline:

    tokenize -> embed (+positional) -> self-attention (with KV cache)
             -> MoE feed-forward (router + top-k experts)
             -> unembed to logits -> temperature / top-k / top-p sampling

It is intentionally tiny so every intermediate tensor can be shown in a table.
The generated text is NOT meant to be coherent (the weights are not trained);
the goal is to make the *mechanics* of inference visible step by step.
"""

import json
import os

import numpy as np


# Path where trainer.py saves learned weights (loadable for inference).
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "trained_weights.npz")
# Sidecar JSON storing the trained model's config + vocab + tokenizer mode.
WEIGHTS_META = os.path.join(os.path.dirname(__file__), "trained_meta.json")


# ---------------------------------------------------------------------------
# Small fixed vocabulary so generated token ids map back to readable words.
# ---------------------------------------------------------------------------
VOCAB = [
    "<bos>", "<eos>", "<unk>",
    "the", "a", "cat", "dog", "sat", "ran", "on", "mat",
    "sky", "is", "blue", "sun", "rises", "in", "east",
    "ai", "model", "learns", "from", "data", "very", "fast",
]
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}


def _round(arr, n=3):
    """Convert a numpy array to a JSON-friendly nested list of rounded floats."""
    return np.round(np.asarray(arr, dtype=float), n).tolist()


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class ToyMoEModel:
    """A (multi-layer) MoE transformer with deterministic (seeded) weights.

    Untrained, it defaults to a single layer so the inference demo is simple.
    When loading trained weights it adopts whatever depth (`num_layers`) and
    sizes the training run used.
    """

    def __init__(self, model_seed=42):
        self.d_model = 8
        self.num_heads = 2
        self.head_dim = self.d_model // self.num_heads
        self.num_experts = 4
        self.expert_hidden = 16
        self.num_layers = 1
        self.vocab = VOCAB
        self.mode = "word"
        self.token_to_id = dict(TOKEN_TO_ID)
        self.vocab_size = len(VOCAB)

        rng = np.random.default_rng(model_seed)

        def randn(*shape, scale):
            return rng.standard_normal(shape) * scale

        # Token embedding table:  vocab_size x d_model
        self.embed = randn(self.vocab_size, self.d_model, scale=0.6)

        # One block per layer: attention projections + MoE router + experts.
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append({
                "Wq": randn(self.d_model, self.d_model, scale=0.5),
                "Wk": randn(self.d_model, self.d_model, scale=0.5),
                "Wv": randn(self.d_model, self.d_model, scale=0.5),
                "Wo": randn(self.d_model, self.d_model, scale=0.5),
                "W_router": randn(self.d_model, self.num_experts, scale=0.8),
                "experts": [
                    {"W1": randn(self.d_model, self.expert_hidden, scale=0.5),
                     "W2": randn(self.expert_hidden, self.d_model, scale=0.5)}
                    for _ in range(self.num_experts)
                ],
            })

        # Unembedding / output projection:  d_model x vocab_size
        self.W_unembed = randn(self.d_model, self.vocab_size, scale=0.5)

        self.trained = False

    def load_trained(self, path, meta_path=None):
        """Overwrite the random weights with weights produced by trainer.py.

        The .npz keys match trainer.MoELM.P, and the JSON sidecar records the
        config + vocabulary + tokenizer mode so a char-level model trained on
        Tiny Shakespeare rebuilds correctly here. Returns True on success.
        """
        if not os.path.exists(path):
            return False

        # Adopt the trained model's config/vocab/mode if the sidecar exists.
        meta_path = meta_path or WEIGHTS_META
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            cfg = meta["config"]
            self.d_model = cfg["d_model"]
            self.num_heads = cfg["num_heads"]
            self.head_dim = cfg["head_dim"]
            self.num_experts = cfg["num_experts"]
            self.expert_hidden = cfg["expert_hidden"]
            self.num_layers = cfg.get("num_layers", 1)
            self.vocab = list(meta["vocab"])
            self.vocab_size = len(self.vocab)
            self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
            self.mode = meta.get("mode", "word")

        w = np.load(path)
        self.embed = w["embed"]
        self.W_unembed = w["W_unembed"]
        self.layers = []
        for l in range(self.num_layers):
            p = f"L{l}_"
            self.layers.append({
                "Wq": w[p + "Wq"], "Wk": w[p + "Wk"],
                "Wv": w[p + "Wv"], "Wo": w[p + "Wo"],
                "W_router": w[p + "W_router"],
                "experts": [
                    {"W1": w[p + f"e{e}_W1"], "W2": w[p + f"e{e}_W2"]}
                    for e in range(self.num_experts)
                ],
            })
        self.trained = True
        return True

    # ------------------------------------------------------------------ #
    # Tokenization is model-aware: word-level (fixed vocab) or char-level
    # (trained Shakespeare model). Returns a list of token dicts.
    # ------------------------------------------------------------------ #
    def tokenize_prompt(self, prompt):
        if self.mode == "char":
            space = self.token_to_id.get(" ")
            tokens = []
            for ch in prompt:
                tid = self.token_to_id.get(ch)
                if tid is None:
                    tid = space if space is not None else 0
                    tokens.append({"text": ch, "id": tid, "oov": True})
                else:
                    tokens.append({"text": ch, "id": tid, "oov": False})
            if not tokens:  # never feed an empty sequence
                tokens.append({"text": " ", "id": space or 0, "oov": False})
            return tokens

        # word mode
        raw = [w for w in prompt.lower().replace("\n", " ").split(" ") if w]
        cleaned = ["".join(c for c in w if c.isalnum()) for w in raw]
        cleaned = [w for w in cleaned if w]
        unk = self.token_to_id.get("<unk>", 0)
        tokens = [{"text": "<bos>", "id": self.token_to_id.get("<bos>", 0), "oov": False}]
        for w in cleaned:
            if w in self.token_to_id:
                tokens.append({"text": w, "id": self.token_to_id[w], "oov": False})
            else:
                tokens.append({"text": w, "id": unk, "oov": True})
        return tokens

    def eos_id(self):
        return self.token_to_id.get("<eos>", -1)

    # ------------------------------------------------------------------ #
    # Static description of the weights (sent to the frontend once).
    # ------------------------------------------------------------------ #
    def describe(self):
        return {
            "config": {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "expert_hidden": self.expert_hidden,
                "num_layers": self.num_layers,
                "vocab_size": self.vocab_size,
            },
            "mode": self.mode,
            "vocab": self.vocab,
            "embed": _round(self.embed),
            "layers": [
                {
                    "W_router": _round(layer["W_router"]),
                    "experts": [
                        {"W1": _round(e["W1"]), "W2": _round(e["W2"])}
                        for e in layer["experts"]
                    ],
                }
                for layer in self.layers
            ],
        }

    # ------------------------------------------------------------------ #
    # Building blocks.
    # ------------------------------------------------------------------ #
    def positional_encoding(self, pos):
        """Standard sinusoidal positional encoding for one position."""
        pe = np.zeros(self.d_model)
        for i in range(0, self.d_model, 2):
            denom = 10000 ** (i / self.d_model)
            pe[i] = np.sin(pos / denom)
            if i + 1 < self.d_model:
                pe[i + 1] = np.cos(pos / denom)
        return pe

    def _split_heads(self, vec):
        return vec.reshape(self.num_heads, self.head_dim)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def tokenize(prompt):
    """Whitespace tokenizer with <bos> prefix; OOV words map to <unk>."""
    raw = [w for w in prompt.lower().replace("\n", " ").split(" ") if w]
    cleaned = ["".join(ch for ch in w if ch.isalnum()) for w in raw]
    cleaned = [w for w in cleaned if w]

    tokens = [{"text": "<bos>", "id": TOKEN_TO_ID["<bos>"], "oov": False}]
    for w in cleaned:
        if w in TOKEN_TO_ID:
            tokens.append({"text": w, "id": TOKEN_TO_ID[w], "oov": False})
        else:
            tokens.append({"text": w, "id": TOKEN_TO_ID["<unk>"], "oov": True})
    return tokens


# ---------------------------------------------------------------------------
# The full inference run, recording everything into a trace.
# ---------------------------------------------------------------------------
def run_inference(prompt, temperature=0.9, top_k=5, top_p=0.9,
                  max_tokens=6, seed=0, model_seed=42, use_trained=False):
    model = ToyMoEModel(model_seed=model_seed)
    if use_trained:
        model.load_trained(WEIGHTS_PATH)
    temperature = max(float(temperature), 1e-3)
    top_k = int(top_k)
    top_p = float(top_p)
    sample_rng = np.random.default_rng(seed)

    tokens = model.tokenize_prompt(prompt)
    eos = model.eos_id()

    # Persistent KV cache: one list per (layer, head). Each layer keeps its own
    # keys/values, since each layer attends in its own representation space.
    k_cache = [[[] for _ in range(model.num_heads)] for _ in range(model.num_layers)]
    v_cache = [[[] for _ in range(model.num_heads)] for _ in range(model.num_layers)]
    cache_tokens = []  # text label per cached position (shared across layers)

    steps = []

    def process_position(pos, token_id, token_text):
        """Run one token through every layer; append K/V to each layer's cache.

        Returns (logits, layer_records, embed_record) where layer_records is a
        list of {layer, attention, moe} dicts, one per transformer-MoE block.
        """
        x_embed = model.embed[token_id].copy()
        x_pos = model.positional_encoding(pos)
        x = x_embed + x_pos
        cache_tokens.append(token_text)   # once per position, not per layer

        layer_records = []
        h_state = x                        # input to layer 0
        for l, layer in enumerate(model.layers):
            x_in = h_state

            # ---- Attention ----
            q = x_in @ layer["Wq"]
            k = x_in @ layer["Wk"]
            v = x_in @ layer["Wv"]
            qh = model._split_heads(q)
            kh = model._split_heads(k)
            vh = model._split_heads(v)

            # KV-cache update for THIS layer.
            for h in range(model.num_heads):
                k_cache[l][h].append(kh[h].copy())
                v_cache[l][h].append(vh[h].copy())

            head_records = []
            attn_out = np.zeros(model.d_model)
            for h in range(model.num_heads):
                K = np.array(k_cache[l][h])       # (cache_len, head_dim)
                V = np.array(v_cache[l][h])
                scores = (qh[h] @ K.T) / np.sqrt(model.head_dim)
                weights = softmax(scores)
                out_h = weights @ V
                attn_out[h * model.head_dim:(h + 1) * model.head_dim] = out_h
                head_records.append({
                    "q": _round(qh[h]), "K": _round(K),
                    "scores": _round(scores), "weights": _round(weights),
                    "out": _round(out_h),
                })

            attn_proj = attn_out @ layer["Wo"]
            h_attn = x_in + attn_proj            # residual

            attention_record = {
                "q": _round(q), "k": _round(k), "v": _round(v),
                "cache_len": len(cache_tokens),
                "cache_tokens": list(cache_tokens),
                "heads": head_records,
                "attn_out": _round(attn_out),
                "attn_proj": _round(attn_proj),
                "hidden": _round(h_attn),
            }

            # ---- MoE feed-forward ----
            router_logits = h_attn @ layer["W_router"]
            gates_all = softmax(router_logits)
            topk_idx = sorted(np.argsort(gates_all)[::-1][:2].tolist())  # top-2
            sel_gate_raw = np.array([gates_all[i] for i in topk_idx])
            sel_gate = sel_gate_raw / sel_gate_raw.sum()  # renormalize selected

            expert_outputs = []
            moe_out = np.zeros(model.d_model)
            for gate, e_idx in zip(sel_gate, topk_idx):
                W1 = layer["experts"][e_idx]["W1"]
                W2 = layer["experts"][e_idx]["W2"]
                hidden = np.maximum(0.0, h_attn @ W1)   # ReLU
                e_out = hidden @ W2
                moe_out += gate * e_out
                expert_outputs.append({
                    "expert": e_idx, "gate": float(round(gate, 3)),
                    "hidden": _round(hidden), "out": _round(e_out),
                })

            h_moe = h_attn + moe_out             # residual

            moe_record = {
                "router_logits": _round(router_logits),
                "gates": _round(gates_all),
                "selected": topk_idx,
                "selected_gates": _round(sel_gate),
                "expert_outputs": expert_outputs,
                "moe_out": _round(moe_out),
                "hidden": _round(h_moe),
            }

            layer_records.append({
                "layer": l,
                "input": _round(x_in),
                "attention": attention_record,
                "moe": moe_record,
            })
            h_state = h_moe                      # feed into next layer

        embed_record = {
            "token": token_text,
            "token_id": int(token_id),
            "pos": pos,
            "embedding": _round(x_embed),
            "positional": _round(x_pos),
            "summed": _round(x),
        }

        # ---- Output logits ----
        logits = h_state @ model.W_unembed
        return logits, layer_records, embed_record

    def sample(logits):
        """Temperature -> softmax -> top-k -> top-p (nucleus) -> sample."""
        scaled = logits / temperature
        probs = softmax(scaled)

        order = np.argsort(probs)[::-1]

        # top-k mask
        topk_keep = set(order[:top_k].tolist()) if top_k > 0 else set(order.tolist())
        probs_k = np.array([p if i in topk_keep else 0.0 for i, p in enumerate(probs)])

        # top-p (nucleus) mask, applied over what top-k kept
        order_k = np.argsort(probs_k)[::-1]
        cum = 0.0
        topp_keep = set()
        total = probs_k.sum()
        for i in order_k:
            if probs_k[i] <= 0:
                break
            topp_keep.add(int(i))
            cum += probs_k[i] / total if total > 0 else 0
            if cum >= top_p:
                break
        probs_kp = np.array([p if i in topp_keep else 0.0 for i, p in enumerate(probs_k)])

        final = probs_kp / probs_kp.sum() if probs_kp.sum() > 0 else probs
        chosen = int(sample_rng.choice(len(final), p=final))

        return {
            "logits": _round(logits),
            "scaled_logits": _round(scaled),
            "probs": _round(probs, 4),
            "kept_topk": sorted(topk_keep),
            "kept_topp": sorted(topp_keep),
            "final_probs": _round(final, 4),
            "chosen_id": chosen,
            "chosen_token": model.vocab[chosen],
            "params": {"temperature": round(temperature, 3),
                       "top_k": top_k, "top_p": round(top_p, 3)},
        }

    # ---- 1) Tokenization step ----
    steps.append({
        "kind": "tokenize",
        "title": "Tokenization",
        "tokens": tokens,
    })

    # ---- 2) Prefill: process every prompt token, build the KV cache ----
    last_logits = None
    last_records = None
    for pos, tok in enumerate(tokens):
        last_logits, layer_r, emb_r = process_position(pos, tok["id"], tok["text"])
        last_records = (layer_r, emb_r)

    sampling = sample(last_logits)
    layer_r, emb_r = last_records
    steps.append({
        "kind": "generate",
        "phase": "prefill",
        "title": "Prefill + first token",
        "embed": emb_r,
        "layers": layer_r,
        "sampling": sampling,
    })

    # ---- 3) Decode loop ----
    generated = [{"text": sampling["chosen_token"], "id": sampling["chosen_id"]}]
    next_id = sampling["chosen_id"]
    pos = len(tokens)

    for step_i in range(max_tokens - 1):
        if next_id == eos:
            break
        logits, layer_r, emb_r = process_position(
            pos, next_id, model.vocab[next_id])
        sampling = sample(logits)
        steps.append({
            "kind": "generate",
            "phase": "decode",
            "title": f"Decode token #{step_i + 2}",
            "embed": emb_r,
            "layers": layer_r,
            "sampling": sampling,
        })
        generated.append({"text": sampling["chosen_token"], "id": sampling["chosen_id"]})
        next_id = sampling["chosen_id"]
        pos += 1

    return {
        "model": model.describe(),
        "trained": model.trained,
        "prompt": prompt,
        "params": {"temperature": round(temperature, 3), "top_k": top_k,
                   "top_p": round(top_p, 3), "max_tokens": max_tokens, "seed": seed},
        "tokens": tokens,
        "generated": generated,
        "steps": steps,
    }
