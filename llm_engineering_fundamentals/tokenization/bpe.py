from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def bytes_to_symbols(text: str) -> list[str]:
    """
    Convert text to a list of *byte* symbols.

    Using bytes keeps the tokenizer closed over arbitrary unicode input.
    """
    b = text.encode("utf-8")
    return [f"b{byte:02x}" for byte in b]


def symbols_to_bytes(symbols: list[str]) -> bytes:
    out = bytearray()
    for s in symbols:
        if s == "b__":  # Special case for space
            out.append(0x20)
        elif len(s) == 3 and s[0] == "b":
            out.append(int(s[1:], 16))
        else:
            raise ValueError(f"Invalid byte symbol: {s}")
    return bytes(out)


def pair_counts(words: list[list[str]], freqs: list[int]) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for w, f in zip(words, freqs, strict=True):
        for i in range(len(w) - 1):
            counts[(w[i], w[i + 1])] += f
    return counts


def merge_pair(word: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


@dataclass(frozen=True)
class BPEModel:
    """
    Minimal byte-level BPE model (educational, correctness-first).

    Notes:
    - We treat whitespace-separated chunks as “words” purely for simplicity.
    - We add an end marker (`b__`) to make merges stable across boundaries.
    """

    merges: list[tuple[str, str]]
    vocab: dict[str, int]
    id_to_symbol: dict[int, str]

    @staticmethod
    def train(corpus: Iterable[str], merges: int) -> "BPEModel":
        word_freq: Counter[tuple[str, ...]] = Counter()
        for line in corpus:
            for chunk in line.strip().split():
                syms = tuple(bytes_to_symbols(chunk) + ["b__"])
                word_freq[syms] += 1

        words = [list(w) for w in word_freq.keys()]
        freqs = [word_freq[tuple(w)] for w in words]

        learned_merges: list[tuple[str, str]] = []

        for _ in range(merges):
            pc = pair_counts(words, freqs)
            if not pc:
                break
            (a, b), _count = pc.most_common(1)[0]
            new_sym = f"{a}{b}"
            words = [merge_pair(w, (a, b), new_sym) for w in words]
            learned_merges.append((a, b))

        symbols: set[str] = set()
        for w in words:
            symbols.update(w)

        base = {f"b{byte:02x}" for byte in range(256)}
        base.add("b__")
        symbols |= base

        vocab = {sym: i for i, sym in enumerate(sorted(symbols))}
        id_to_symbol = {i: s for s, i in vocab.items()}
        return BPEModel(merges=learned_merges, vocab=vocab, id_to_symbol=id_to_symbol)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in text.split():
            syms = bytes_to_symbols(chunk) + ["b__"]
            for a, b in self.merges:
                merged = f"{a}{b}"
                syms = merge_pair(syms, (a, b), merged)
            ids.extend([self.vocab[s] for s in syms])
        return ids

    def decode(self, ids: list[int]) -> str:
        syms = [self.id_to_symbol[i] for i in ids]

        chunks: list[list[str]] = []
        cur: list[str] = []
        for s in syms:
            if s == "b__":
                if cur:
                    chunks.append(cur)
                cur = []
                continue

            if s.startswith("b") and len(s) == 3:
                cur.append(s)
                continue

            # merged symbols are concatenations of byte symbols; expand them back.
            expanded: list[str] = []
            i = 0
            while i < len(s):
                if s[i] != "b":
                    raise ValueError(f"Cannot decode symbol: {s}")
                expanded.append(s[i : i + 3])
                i += 3
            
            # Process expanded symbols, handling b__ as word boundaries
            for exp_sym in expanded:
                if exp_sym == "b__":
                    if cur:
                        chunks.append(cur)
                    cur = []
                else:
                    cur.append(exp_sym)

        if cur:
            chunks.append(cur)

        decoded_chunks: list[str] = []
        for c in chunks:
            if not c:
                continue
            decoded_chunks.append(symbols_to_bytes(c).decode("utf-8", errors="replace"))
        return " ".join(decoded_chunks)

    def to_json(self) -> dict:
        return {"merges": self.merges, "vocab": self.vocab}

    @staticmethod
    def from_json(data: dict) -> "BPEModel":
        merges = [tuple(x) for x in data["merges"]]
        vocab = {str(k): int(v) for k, v in data["vocab"].items()}
        id_to_symbol = {i: s for s, i in vocab.items()}
        return BPEModel(merges=merges, vocab=vocab, id_to_symbol=id_to_symbol)


def cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix for row vectors in x."""
    eps = 1e-12
    norms = np.linalg.norm(x, axis=1, keepdims=True) + eps
    xn = x / norms
    return xn @ xn.T


def gaussian_entropy(mu: float, sigma: float) -> float:
    """Crude entropy estimate by fitting a Gaussian (educational)."""
    sigma = max(float(sigma), 1e-12)
    return 0.5 * math.log(2 * math.pi * math.e * sigma * sigma)


def demo_onehot_vs_learned(vocab_size: int = 64, seed: int = 7) -> dict[str, float]:
    """Return summary stats comparing one-hot vs learned embedding cosine similarities."""
    rng = np.random.default_rng(seed)
    onehot = np.eye(vocab_size, dtype=np.float32)
    learned = rng.normal(0, 1, size=(vocab_size, 64)).astype(np.float32)

    sim_onehot = cosine_similarity_matrix(onehot)
    sim_learned = cosine_similarity_matrix(learned)

    mask = ~np.eye(vocab_size, dtype=bool)
    o = sim_onehot[mask]
    l = sim_learned[mask]
    return {
        "onehot_offdiag_mean": float(o.mean()),
        "onehot_offdiag_std": float(o.std()),
        "learned_offdiag_mean": float(l.mean()),
        "learned_offdiag_std": float(l.std()),
        "learned_offdiag_entropy_est": float(gaussian_entropy(l.mean(), l.std())),
    }


