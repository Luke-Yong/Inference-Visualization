"""
Datasets + tokenizers for the MoE visualizer.

Two datasets share one small interface so the trainer/engine stay generic:

  WordCorpusDataset  -- the original tiny templated corpus (word-level,
                        25-word fixed vocab). Great for maximally legible
                        weight tables.

  ShakespeareCharDataset -- a slice of Andrej Karpathy's "Tiny Shakespeare"
                        (~1MB of real English). CHARACTER-LEVEL, so the
                        vocabulary is just the unique characters (~40-70),
                        which keeps the embedding table legible while
                        training on genuine English. The file is downloaded
                        on first run and cached on disk.

Common interface (duck-typed):
    .name           short id
    .mode           "word" | "char"
    .vocab          list[str]  (token id -> token string)
    .token_to_id    dict[str,int]
    .encode(text)   -> list[int]
    .decode(ids)    -> str
    .training_sequences() -> list[list[int]]   (each is one training example)
    .seeds()        -> list[str]   (prompts used to show generation quality)
    .info()         -> dict        (sent to the UI to describe the dataset)
    .block_len      max sequence length (for positional-encoding sizing)
"""

import os
import urllib.request

import numpy as np

import engine  # reuse the word VOCAB / tokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAKESPEARE_PATH = os.path.join(DATA_DIR, "tinyshakespeare.txt")
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# Offline fallback so the app still works without internet.
FALLBACK_TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
    "All:\nSpeak, speak.\n\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
    "All:\nResolved. resolved.\n\n"
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"
) * 4


# --------------------------------------------------------------------------- #
# Word-level dataset (the original demo corpus)
# --------------------------------------------------------------------------- #
class WordCorpusDataset:
    name = "word"
    mode = "word"

    SENTENCES = [
        "the cat sat on the mat",
        "a dog ran on the mat",
        "the sky is blue",
        "the sun rises in the east",
        "the ai model learns from data very fast",
    ]
    EVAL = [
        ("the cat sat on the", "mat"),
        ("the sky is", "blue"),
        ("the sun rises in the", "east"),
        ("the ai model learns from", "data"),
    ]

    def __init__(self, **_):
        self.vocab = list(engine.VOCAB)
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.block_len = 16

    def encode(self, text):
        ids = [self.token_to_id["<bos>"]]
        for w in text.lower().split():
            ids.append(self.token_to_id.get(w, self.token_to_id["<unk>"]))
        ids.append(self.token_to_id["<eos>"])
        return ids

    def decode(self, ids):
        return " ".join(self.vocab[i] for i in ids)

    def training_sequences(self):
        return [self.encode(s) for s in self.SENTENCES]

    def seeds(self):
        return [p for p, _ in self.EVAL]

    def eval_pairs(self):
        return self.EVAL

    def info(self):
        return {
            "name": "Toy word corpus",
            "mode": "word",
            "vocab_size": len(self.vocab),
            "vocab_preview": self.vocab,
            "num_examples": len(self.SENTENCES),
            "examples": [{"text": s, "tokens": [self.vocab[i] for i in self.encode(s)]}
                         for s in self.SENTENCES],
            "description": "5 templated sentences from a fixed 25-word vocabulary.",
        }


# --------------------------------------------------------------------------- #
# Character-level Tiny Shakespeare
# --------------------------------------------------------------------------- #
def _ensure_shakespeare():
    """Download Tiny Shakespeare once, cache it, return the full text.

    Falls back to a small embedded snippet if the download fails (offline).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(SHAKESPEARE_PATH) and os.path.getsize(SHAKESPEARE_PATH) > 1000:
        with open(SHAKESPEARE_PATH, "r", encoding="utf-8") as f:
            return f.read(), "cache"
    try:
        req = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "moe-viz"})
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode("utf-8")
        with open(SHAKESPEARE_PATH, "w", encoding="utf-8") as f:
            f.write(text)
        return text, "downloaded"
    except Exception:
        return FALLBACK_TEXT, "fallback"


class ShakespeareCharDataset:
    name = "shakespeare"
    mode = "char"

    def __init__(self, slice_chars=3000, block_size=32, stride=16, **_):
        self.block_len = int(block_size)
        self.stride = int(stride)
        full, self.source = _ensure_shakespeare()
        self.full_len = len(full)
        # Use a contiguous slice for training (full file is too big to
        # full-batch in numpy on every epoch).
        self.text = full[: int(slice_chars)]

        # Vocab = the unique characters that appear in the slice, sorted so the
        # table ordering is stable. A leading space id is handy as a fallback.
        chars = sorted(set(self.text))
        if " " not in chars:
            chars = [" "] + chars
        self.vocab = chars
        self.token_to_id = {c: i for i, c in enumerate(self.vocab)}
        self._space = self.token_to_id[" "]

    def encode(self, text):
        return [self.token_to_id.get(c, self._space) for c in text]

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def training_sequences(self):
        ids = self.encode(self.text)
        seqs = []
        n = len(ids)
        i = 0
        while i + self.block_len + 1 <= n:
            seqs.append(ids[i:i + self.block_len + 1])  # +1 so we have a target
            i += self.stride
        if not seqs:                       # very short slice
            seqs = [ids]
        return seqs

    def seeds(self):
        # Short prefixes that exist in normal English/Shakespeare text.
        return ["The ", "And ", "My lord", "To be"]

    def info(self):
        preview = self.text[:400].replace("\r", "")
        return {
            "name": "Tiny Shakespeare (char-level)",
            "mode": "char",
            "vocab_size": len(self.vocab),
            "vocab_preview": [c if c not in (" ", "\n", "\t") else
                              {" ": "␠", "\n": "↵", "\t": "⇥"}[c] for c in self.vocab],
            "num_examples": len(self.training_sequences()),
            "block_size": self.block_len,
            "slice_chars": len(self.text),
            "full_chars": self.full_len,
            "source": self.source,           # downloaded / cache / fallback
            "preview": preview,
            "description": (
                f"A {len(self.text)}-character slice of real English "
                f"(of {self.full_len} total), tokenized per character. "
                f"Trained on {len(self.training_sequences())} overlapping "
                f"windows of {self.block_len} characters."
            ),
        }


def get_dataset(name, **kwargs):
    if name == "shakespeare":
        return ShakespeareCharDataset(**kwargs)
    return WordCorpusDataset(**kwargs)
