from typing import (
    List,
    Tuple,
    Iterator,
)

from gensim.corpora import (
    Dictionary,
    TextCorpus,
)

from .sentence import Sentence


__all__: Tuple[str, ...] = (
    'SentenceCorpus',
)


class SentenceCorpus(TextCorpus):
    """
        Args:
            sentences: a list of `sentence.Sentence` instances.
            no_below: ignore unique tokens with inverse document count below this value (int).
            no_above: ignore unique tokens with inverse document frequency above this value (float).
            max_size: maximum vocabulary size.

        See `gensim.corpora.TextCorpus` for more details.
    """

    def __init__(self,
                 sentences: List[Sentence],
                 no_below: int = 3,
                 no_above: float = 0.8,
                 max_size: int = 20000
                 ):

        # preserver original sentences
        self.sentences: List[Sentence] = sentences

        # init dictionary
        self.dictionary: Dictionary = Dictionary(self.get_texts(), prune_at=max_size)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=max_size)
        self.dictionary.compactify()

        # precompute bows
        self.bows: List[List[Tuple[int, int]]] = []
        for tokens in self.get_texts():
            bow: List[Tuple[int, int]] = self.dictionary.doc2bow(tokens)
            self.bows.append(bow)

    def get_texts(self) -> Iterator[List[str]]:
        for sentence in self.sentences:
            yield sentence.tokens
