from re import split

from typing import (
    List,
    Tuple,
    Callable,
)

from .sentence import Sentence


__all__: Tuple[str, ...] = (
    'parse_text_into_sentences',
)


def parse_text_into_sentences(text: str,
                              tokenizer: Callable[[str], List[str]],
                              min_num_tokens: int = 2
                              ) -> List[Sentence]:
    """
        This function splits the given text into sentence candidates using a pre-defined splitter,
        then creates a list of `sentence.Sentence` instances, tokenized by the given tokenizer.
    """

    # init
    index: int = 0
    duplication_checker: set = set()
    sentences: List[Sentence] = []

    # parse text
    candidates: List[str] = split(r'(?:(?<=[^0-9])\.|\n|!|\?)', text)
    for candidate in candidates:

        # cleanse the candidate
        candidate_stripped: str = candidate.strip('. ')
        if not len(candidate_stripped):
            continue
        if candidate_stripped in duplication_checker:
            continue

        # tokenize the candidate
        tokens: List[str] = tokenizer(candidate_stripped)
        if len(tokens) < min_num_tokens:
            continue
        duplication_checker.add(candidate_stripped)

        # create a sentence
        sentence = Sentence(index, candidate_stripped, tokens)
        sentences.append(sentence)
        index += 1

    # return
    return sentences
