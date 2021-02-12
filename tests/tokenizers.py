import json
import requests

from typing import (
    List,
    Tuple,
)

from konlpy.tag import Okt
from requests.models import Response


__all__: Tuple[str, ...] = (
    'OktTokenizer',
    'ApiTokenizer',
)


class OktTokenizer:
    """
        A POS-tagger based tokenizer functor. Note that these are just examples.

        Example:
            tokenizer: OktTokenizer = OktTokenizer()
            tokens: List[str] = tokenizer(your_text_here)
    """

    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens


class ApiTokenizer:
    """
        An API based tokenizer functor, assuming that the response body is a jsonifyable string with content of list of `str` tokens.

        Example:
            tokenizer: ApiTokenizer = ApiTokenizer()
            tokens: List[str] = tokenizer(your_text_here)
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint: str = endpoint

    def __call__(self, text: str) -> List[str]:
        body: bytes = text.encode('utf-8')
        res: Response = requests.post(self.endpoint, data=body)
        tokens: List[str] = json.loads(res.text)
        return tokens
