"""
A small hand-labeled part-of-speech corpus used to train and evaluate the
HMM tagger in this repository.

Tagset (simplified, 11 tags): DET, NOUN, PROPN, PRON, VERB, ADJ, ADV, ADP,
CONJ, NUM, PUNCT.

This is intentionally small (45 sentences) so the entire training set is
auditable in one file. It is a teaching/demo corpus, not a substitute for a
large treebank -- see the README for discussion.
"""

from typing import List, Tuple

Sentence = List[Tuple[str, str]]


def _s(text: str, tags: str) -> Sentence:
    words = text.split()
    tag_list = tags.split()
    assert len(words) == len(tag_list), (text, tags)
    return list(zip(words, tag_list))


TAGGED_SENTENCES: List[Sentence] = [
    _s("The dog barked loudly .", "DET NOUN VERB ADV PUNCT"),
    _s("The cat sat on the mat .", "DET NOUN VERB ADP DET NOUN PUNCT"),
    _s("A dog chased the cat .", "DET NOUN VERB DET NOUN PUNCT"),
    _s("The big dog ran fast .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("She walked to the store .", "PRON VERB ADP DET NOUN PUNCT"),
    _s("He runs every morning .", "PRON VERB DET NOUN PUNCT"),
    _s("They played in the park .", "PRON VERB ADP DET NOUN PUNCT"),
    _s("We watched the old movie .", "PRON VERB DET ADJ NOUN PUNCT"),
    _s("John bought a new car .", "PROPN VERB DET ADJ NOUN PUNCT"),
    _s("Mary reads books quietly .", "PROPN VERB NOUN ADV PUNCT"),
    _s("Mary and John walked home .", "PROPN CONJ PROPN VERB NOUN PUNCT"),
    _s("The teacher and the student talked .", "DET NOUN CONJ DET NOUN VERB PUNCT"),
    _s("I like small dogs and big cats .", "PRON VERB ADJ NOUN CONJ ADJ NOUN PUNCT"),
    _s("The three dogs barked loudly .", "DET NUM NOUN VERB ADV PUNCT"),
    _s("Two cats slept on the sofa .", "NUM NOUN VERB ADP DET NOUN PUNCT"),
    _s("The children played happily .", "DET NOUN VERB ADV PUNCT"),
    _s("The old man walked slowly .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("The young woman smiled warmly .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("He will arrive tomorrow .", "PRON VERB VERB ADV PUNCT"),
    _s("She can sing beautifully .", "PRON VERB VERB ADV PUNCT"),
    _s("The dog and the cat played together .", "DET NOUN CONJ DET NOUN VERB ADV PUNCT"),
    _s("John and Mary bought two books .", "PROPN CONJ PROPN VERB NUM NOUN PUNCT"),
    _s("The student wrote a long essay .", "DET NOUN VERB DET ADJ NOUN PUNCT"),
    _s("The teacher read the essay carefully .", "DET NOUN VERB DET NOUN ADV PUNCT"),
    _s("We visited the museum yesterday .", "PRON VERB DET NOUN ADV PUNCT"),
    _s("They will visit the museum soon .", "PRON VERB VERB DET NOUN ADV PUNCT"),
    _s("The small cat sat quietly .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("The big dog barked fiercely .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("I bought five new books .", "PRON VERB NUM ADJ NOUN PUNCT"),
    _s("She bought three red apples .", "PRON VERB NUM ADJ NOUN PUNCT"),
    _s("The apples were fresh and sweet .", "DET NOUN VERB ADJ CONJ ADJ PUNCT"),
    _s("The old car broke suddenly .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("John walked slowly to school .", "PROPN VERB ADV ADP NOUN PUNCT"),
    _s("Mary ran quickly to the store .", "PROPN VERB ADV ADP DET NOUN PUNCT"),
    _s("The dog and the man walked together .", "DET NOUN CONJ DET NOUN VERB ADV PUNCT"),
    _s("We like the new teacher .", "PRON VERB DET ADJ NOUN PUNCT"),
    _s("They like the old museum .", "PRON VERB DET ADJ NOUN PUNCT"),
    _s("The cat and the dog slept .", "DET NOUN CONJ DET NOUN VERB PUNCT"),
    _s("He wrote three short stories .", "PRON VERB NUM ADJ NOUN PUNCT"),
    _s("She reads two long novels .", "PRON VERB NUM ADJ NOUN PUNCT"),
    _s("The young student studied hard .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("The tired teacher rested quietly .", "DET ADJ NOUN VERB ADV PUNCT"),
    _s("John and the teacher talked briefly .", "PROPN CONJ DET NOUN VERB ADV PUNCT"),
    _s("Mary and the student studied together .", "PROPN CONJ DET NOUN VERB ADV PUNCT"),
    _s("The five students passed the exam .", "DET NUM NOUN VERB DET NOUN PUNCT"),
]
