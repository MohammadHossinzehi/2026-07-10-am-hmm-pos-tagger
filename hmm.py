"""
Hidden Markov Model part-of-speech tagger.

Implements supervised training (maximum-likelihood estimation with
add-k smoothing) and Viterbi decoding in log-space, from scratch,
using numpy only for the dynamic-programming trellis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

Sentence = List[Tuple[str, str]]  # list of (word, tag) pairs

UNK = "<UNK>"


class HiddenMarkovModel:
    """A first-order HMM POS tagger trained by counting, decoded by Viterbi."""

    def __init__(self, smoothing: float = 0.1):
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0")
        self.smoothing = smoothing

        self.tags: List[str] = []
        self.tag_index: Dict[str, int] = {}
        self.vocab: set = set()

        self._pi = None      # initial tag log-probabilities, shape (n_tags,)
        self._trans = None   # log P(tag_j | tag_i), shape (n_tags, n_tags)
        self._emit: Dict[str, np.ndarray] = {}  # word -> log P(word | tag) per tag index

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, sentences: Sequence[Sentence]) -> "HiddenMarkovModel":
        tagset = sorted({tag for sent in sentences for _, tag in sent})
        self.tags = tagset
        self.tag_index = {t: i for i, t in enumerate(tagset)}
        n_tags = len(tagset)

        self.vocab = {word.lower() for sent in sentences for word, _ in sent}

        init_counts = np.zeros(n_tags)
        trans_counts = np.zeros((n_tags, n_tags))
        emit_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_tags))
        tag_totals = np.zeros(n_tags)

        for sent in sentences:
            if not sent:
                continue
            first_tag = sent[0][1]
            init_counts[self.tag_index[first_tag]] += 1

            prev_idx = None
            for word, tag in sent:
                idx = self.tag_index[tag]
                emit_counts[word.lower()][idx] += 1
                tag_totals[idx] += 1
                if prev_idx is not None:
                    trans_counts[prev_idx, idx] += 1
                prev_idx = idx

        k = self.smoothing

        # Initial tag distribution.
        self._pi = np.log((init_counts + k) / (init_counts.sum() + k * n_tags))

        # Transition distribution, row-normalized (row i = P(next | tag_i)).
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        self._trans = np.log((trans_counts + k) / (row_sums + k * n_tags))

        # Emission distribution per known word, plus a shared <UNK> fallback
        # built from the smoothing mass so unseen words never get probability
        # zero for every tag.
        vocab_size = len(self.vocab) + 1  # +1 reserves a slot for <UNK>
        self._emit = {}
        for word, counts in emit_counts.items():
            self._emit[word] = np.log((counts + k) / (tag_totals + k * vocab_size))
        self._emit[UNK] = np.log((np.zeros(n_tags) + k) / (tag_totals + k * vocab_size))

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _emission_logprobs(self, word: str) -> np.ndarray:
        if self._pi is None:
            raise RuntimeError("model has not been fit yet")
        return self._emit.get(word.lower(), self._emit[UNK])

    def viterbi(self, words: Sequence[str]) -> List[str]:
        """Return the most probable tag sequence for `words` via the Viterbi
        dynamic program, computed in log-space to avoid underflow."""
        if self._pi is None:
            raise RuntimeError("model has not been fit yet")

        n_tags = len(self.tags)
        T = len(words)
        if T == 0:
            return []

        log_trellis = np.full((T, n_tags), -math.inf)
        backptr = np.zeros((T, n_tags), dtype=int)

        log_trellis[0] = self._pi + self._emission_logprobs(words[0])

        for t in range(1, T):
            emit = self._emission_logprobs(words[t])
            # scores[i, j] = best log-prob of reaching tag j at time t via tag i
            scores = log_trellis[t - 1][:, None] + self._trans
            backptr[t] = np.argmax(scores, axis=0)
            log_trellis[t] = np.max(scores, axis=0) + emit

        best_path = np.zeros(T, dtype=int)
        best_path[-1] = int(np.argmax(log_trellis[-1]))
        for t in range(T - 2, -1, -1):
            best_path[t] = backptr[t + 1, best_path[t + 1]]

        return [self.tags[i] for i in best_path]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, sentences: Sequence[Sentence]) -> Dict[str, object]:
        """Tag every sentence and report token-level accuracy plus a
        gold-vs-predicted confusion count for misclassified tokens."""
        total = 0
        correct = 0
        confusion: Dict[Tuple[str, str], int] = defaultdict(int)

        for sent in sentences:
            words = [w for w, _ in sent]
            gold = [t for _, t in sent]
            pred = self.viterbi(words)
            for g, p in zip(gold, pred):
                total += 1
                correct += int(g == p)
                confusion[(g, p)] += 1

        accuracy = correct / total if total else 0.0
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "confusion": dict(confusion),
        }
