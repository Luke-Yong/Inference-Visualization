"""Flask server for the LLM inference + training visualizer."""

import json
import os

from flask import (Flask, render_template, request, jsonify, Response,
                   stream_with_context)

import engine
from engine import run_inference
import trainer

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/train")
def train_page():
    return render_template("train.html")


@app.route("/api/status")
def status():
    has = os.path.exists(engine.WEIGHTS_PATH)
    info = None
    if has and os.path.exists(engine.WEIGHTS_META):
        import json
        with open(engine.WEIGHTS_META, "r", encoding="utf-8") as f:
            meta = json.load(f)
        info = {"dataset": meta.get("dataset"), "mode": meta.get("mode"),
                "vocab_size": meta.get("config", {}).get("vocab_size"),
                "d_model": meta.get("config", {}).get("d_model"),
                "num_layers": meta.get("config", {}).get("num_layers", 1)}
    return jsonify({"has_trained_weights": has, "trained_info": info})


@app.route("/api/infer", methods=["POST"])
def infer():
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "the cat sat on the").strip()
    if not prompt:
        prompt = "the cat sat on the"

    result = run_inference(
        prompt=prompt,
        temperature=float(data.get("temperature", 0.9)),
        top_k=int(data.get("top_k", 5)),
        top_p=float(data.get("top_p", 0.9)),
        max_tokens=int(data.get("max_tokens", 6)),
        seed=int(data.get("seed", 0)),
        use_trained=bool(data.get("use_trained", False)),
    )
    return jsonify(result)


@app.route("/api/train_stream", methods=["POST"])
def train_stream():
    """Stream training progress as Server-Sent Events so the client can show
    real, live progress (loss per epoch) instead of waiting for the whole run."""
    data = request.get_json(force=True) or {}
    kwargs = dict(
        dataset_name=str(data.get("dataset", "shakespeare")),
        epochs=int(data.get("epochs", 60)),
        lr=float(data.get("lr", 0.05)),
        optimizer=str(data.get("optimizer", "adam")),
        seed=int(data.get("seed", 42)),
        d_model=data.get("d_model") or None,
        num_layers=data.get("num_layers") or None,
        slice_chars=int(data.get("slice_chars", 3000)),
        block_size=int(data.get("block_size", 32)),
        balance_alpha=float(data.get("balance_alpha", 0.0)),
    )

    def generate():
        try:
            for event in trainer.train_stream(**kwargs):
                yield "data: " + json.dumps(event) + "\n\n"
        except Exception as exc:  # surface server-side errors to the client
            yield "data: " + json.dumps({"type": "error", "message": str(exc)}) + "\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # disable proxy buffering if present
    }
    return Response(stream_with_context(generate()), headers=headers)


@app.route("/api/reset_weights", methods=["POST"])
def reset_weights():
    for p in (engine.WEIGHTS_PATH, engine.WEIGHTS_META):
        if os.path.exists(p):
            os.remove(p)
    return jsonify({"has_trained_weights": False})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
