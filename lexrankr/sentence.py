from collections import Counter

from typing import (
    List,
    Tuple,
)


__all__: Tuple[str, ...] = (
    'Sentence',
)


class Sentence:
    """
        The purpose of this class is as follows:

        1. In order to use the 'pagerank' function in the networkx library, you need a hashable object.
        2. Summaries should keep the sentence order from its original text to improve the verbosity.
    """

    def __init__(self, index: int, text: str, tokens: List[str]) -> None:
        self.index: int = index
        self.text: str = text
        self.tokens: List[str] = tokens
        self.counter: Counter = Counter(self.tokens)
        self.bow: List[Tuple[int, int]] = []
        self.tfidf_pairs: List[Tuple[int, float]] = []

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        summary: str = self.text[:10]
        if len(self.text) > 10:
            summary += '...'
        return f"Sentence(\"{summary}\")"

    def __hash__(self) -> int:
        return self.index
