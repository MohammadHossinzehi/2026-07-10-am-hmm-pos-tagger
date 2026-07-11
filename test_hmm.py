import math
import random
from collections import Counter
from itertools import product

import numpy as np
import pytest

from corpus import TAGGED_SENTENCES
from hmm import UNK, HiddenMarkovModel


def brute_force_best_path(model: HiddenMarkovModel, words):
    """Exhaustively score every possible tag sequence and return the best
    one. Used to cross-check the Viterbi DP on short sentences, where
    enumeration is cheap -- this is the strongest correctness check
    available for a dynamic-programming decoder."""
    best_score = -math.inf
    best_seq = None
    for seq in product(model.tags, repeat=len(words)):
        idx0 = model.tag_index[seq[0]]
        score = model._pi[idx0] + model._emission_logprobs(words[0])[idx0]
        for t in range(1, len(words)):
            i = model.tag_index[seq[t - 1]]
            j = model.tag_index[seq[t]]
            score += model._trans[i, j]
            score += model._emission_logprobs(words[t])[j]
        if score > best_score:
            best_score = score
            best_seq = list(seq)
    return best_seq


@pytest.fixture(scope="module")
def trained_model():
    model = HiddenMarkovModel(smoothing=0.1)
    model.fit(TAGGED_SENTENCES)
    return model


def test_probabilities_normalize(trained_model):
    trans_probs = np.exp(trained_model._trans)
    row_sums = trans_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)

    init_probs = np.exp(trained_model._pi)
    assert math.isclose(init_probs.sum(), 1.0, abs_tol=1e-6)


def test_viterbi_matches_brute_force_on_short_sentences(trained_model):
    random.seed(0)
    sample_sentences = random.sample(TAGGED_SENTENCES, 5)
    for sent in sample_sentences:
        words = [w for w, _ in sent][:4]  # keep brute force tractable
        if len(words) < 2:
            continue
        expected = brute_force_best_path(trained_model, words)
        actual = trained_model.viterbi(words)
        assert actual == expected


def test_unknown_word_falls_back_to_unk_distribution(trained_model):
    probs = trained_model._emission_logprobs("zzznotaword")
    assert np.array_equal(probs, trained_model._emit[UNK])


def test_viterbi_output_length_matches_input(trained_model):
    words = ["The", "dog", "barked", "."]
    tags = trained_model.viterbi(words)
    assert len(tags) == len(words)
    assert all(t in trained_model.tags for t in tags)


def test_empty_sentence_returns_empty(trained_model):
    assert trained_model.viterbi([]) == []


def test_fit_rejects_non_positive_smoothing():
    with pytest.raises(ValueError):
        HiddenMarkovModel(smoothing=0.0)


def test_unfitted_model_raises_on_viterbi():
    model = HiddenMarkovModel()
    with pytest.raises(RuntimeError):
        model.viterbi(["hello"])


def test_accuracy_beats_most_common_tag_baseline():
    """A trained tagger should clearly beat a most-common-tag baseline on
    this small demo corpus -- this catches silent regressions where the
    model 'learns' nothing useful."""
    split = int(len(TAGGED_SENTENCES) * 0.8)
    train, test = TAGGED_SENTENCES[:split], TAGGED_SENTENCES[split:]

    model = HiddenMarkovModel(smoothing=0.1)
    model.fit(train)
    results = model.evaluate(test)

    tag_counts = Counter(tag for sent in train for _, tag in sent)
    most_common_tag = tag_counts.most_common(1)[0][0]
    baseline_total = sum(len(sent) for sent in test)
    baseline_correct = sum(1 for sent in test for _, tag in sent if tag == most_common_tag)
    baseline_acc = baseline_correct / baseline_total

    assert results["accuracy"] >= baseline_acc
