"""
Train and evaluate the HMM POS tagger on the bundled demo corpus, and
optionally tag a custom sentence passed on the command line.

Usage:
    python evaluate.py
    python evaluate.py "The dog chased a small cat ."
"""

import random
import sys
from collections import Counter

from corpus import TAGGED_SENTENCES
from hmm import HiddenMarkovModel


def train_test_split(sentences, test_ratio: float = 0.2, seed: int = 13):
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    return shuffled[n_test:], shuffled[:n_test]


def main() -> None:
    train, test = train_test_split(TAGGED_SENTENCES)

    model = HiddenMarkovModel(smoothing=0.1)
    model.fit(train)

    results = model.evaluate(test)
    print(f"Trained on {len(train)} sentences, evaluated on {len(test)} sentences.")
    print(
        f"Token-level accuracy: {results['accuracy']:.2%} "
        f"({results['correct']}/{results['total']})"
    )

    errors = {pair: n for pair, n in results["confusion"].items() if pair[0] != pair[1]}
    if errors:
        print("\nMisclassifications (gold -> predicted: count):")
        for (gold, pred), n in sorted(errors.items(), key=lambda kv: -kv[1]):
            print(f"  {gold} -> {pred}: {n}")
    else:
        print("\nNo misclassifications on the test set.")

    tag_counts = Counter(tag for sent in train for _, tag in sent)
    most_common_tag, most_common_n = tag_counts.most_common(1)[0]
    baseline_total = sum(len(sent) for sent in test)
    baseline_correct = sum(1 for sent in test for _, tag in sent if tag == most_common_tag)
    print(
        f"\nMost-common-tag baseline ('{most_common_tag}'): "
        f"{baseline_correct / baseline_total:.2%} accuracy"
    )

    if len(sys.argv) > 1:
        sentence = sys.argv[1]
        words = sentence.strip().split()
        tags = model.viterbi(words)
        print("\nCustom sentence tagging:")
        for word, tag in zip(words, tags):
            print(f"  {word:12s} {tag}")


if __name__ == "__main__":
    main()
