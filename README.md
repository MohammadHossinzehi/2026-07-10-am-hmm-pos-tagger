# HMM POS Tagger

A part-of-speech tagger built from scratch in Python: a first-order Hidden Markov Model trained by maximum-likelihood counting and decoded with the Viterbi algorithm in log-space, plus a small hand-labeled demo corpus, an evaluation script, and a test suite that checks Viterbi against brute-force search.

## What it does and why it's useful

Given a sentence, the tagger assigns each word a part-of-speech tag (noun, verb, adjective, ...) using a Hidden Markov Model: transition probabilities model how tags follow each other, emission probabilities model how likely a tag is to produce a given word, and the Viterbi algorithm finds the most probable tag sequence in O(T times N squared) time (T tokens, N tags) instead of enumerating every possible sequence.

POS tagging is a foundational building block for parsing, information extraction, and many NLP pipelines. This project implements the full pipeline, training, smoothing, decoding, and evaluation, with no NLP libraries, so every moving part is visible: counting, add-k smoothing, log-space arithmetic to avoid underflow, and out-of-vocabulary handling.

## How to run

```bash
pip install -r requirements.txt

# train on the bundled corpus, evaluate on a held-out split, print accuracy
python evaluate.py

# also tag a custom sentence (whitespace-tokenized, punctuation as its own token)
python evaluate.py "The dog chased a small cat ."

# run the test suite
pytest -v
```

Actual output from this repository's corpus and split:

```
Trained on 36 sentences, evaluated on 9 sentences.
Token-level accuracy: 88.14% (52/59)

Misclassifications (gold -> predicted: count):
  NOUN -> ADV: 2
  VERB -> ADP: 2
  ADP -> PUNCT: 1
  NUM -> NOUN: 1
  NOUN -> VERB: 1

Most-common-tag baseline ('NOUN'): 20.34% accuracy

Custom sentence tagging:
  The          DET
  dog          NOUN
  chased       VERB
  a            DET
  small        ADJ
  cat          NOUN
  .            PUNCT
```

The tagger clearly beats the most-common-tag baseline (88% vs 20%), and the custom sentence is tagged correctly end to end, including a word ("chased") that never appears in the training set.

## Design decisions

- Log-space Viterbi. Multiplying dozens of probabilities underflows to zero in double precision, so every probability is stored and combined as a log, with `np.max` and `np.argmax` over a vectorized `(prev_tag, next_tag)` score matrix at each time step instead of a nested Python loop.
- Add-k (Laplace) smoothing on the initial, transition, and emission distributions, so unseen tag transitions and unseen words never get probability zero and Viterbi never dead-ends into negative infinity for every path.
- An explicit `<UNK>` distribution. Words never seen during training fall back to a shared unknown-word emission distribution fit from the smoothing mass, rather than crashing or silently defaulting to a single tag.
- A lowercased vocabulary. Words are matched case-insensitively, which means capitalization is not available as a signal for proper nouns, a deliberate simplification named here rather than hidden, and a natural next feature (a capitalization-aware emission term) if extended.
- Correctness over corpus size. The bundled corpus is 45 hand-labeled sentences, intentionally small so the whole dataset is auditable in one file, not meant to compete with a tagger trained on a large treebank.

## Testing

`test_hmm.py` covers:

- Transition and initial probabilities sum to 1 after smoothing.
- Viterbi output matches an exhaustive brute-force search over all tag sequences on short sentences, the strongest correctness check available for a dynamic-programming decoder.
- Unknown words correctly fall back to the `<UNK>` emission distribution.
- Output length always matches input length, and the empty-sentence edge case returns an empty sequence.
- Invalid construction (non-positive smoothing) and use-before-fit both raise the expected errors.
- The trained model beats a most-common-tag baseline on held-out data, catching silent regressions where the model "learns" nothing.

All of the above were also verified manually against the actual numbers this repository's corpus produces before publishing.

## Files

- `hmm.py`, the `HiddenMarkovModel` class: `fit`, `viterbi`, `evaluate`.
- `corpus.py`, 45 hand-labeled demo sentences (`TAGGED_SENTENCES`).
- `evaluate.py`, CLI: trains, evaluates, prints a confusion summary, optionally tags a custom sentence.
- `test_hmm.py`, test suite, including a brute-force cross-check of Viterbi.
- `requirements.txt`, `numpy` and `pytest`.
