# -*- coding: utf-8 -*-

import unittest

from typing import List

from lexrankr import LexRank

from .tokenizers import OktTokenizer


class TestLexRank(unittest.TestCase):

    def setUp(self) -> None:
        tokenizer: OktTokenizer = OktTokenizer()
        self.lexrank: LexRank = LexRank(tokenizer)
        self.text: str = "사과 배 감 귤. 배 감 귤 수박. 감 귤 수박 딸기. 오이 참외 오징어. 참외 오징어 달팽이. 빨강 파랑 초록. 파랑 초록 노랑. 노랑 노랑 빨강. 검정 파랑 빨강 초록. /"

    def test_summarized(self) -> None:
        self.lexrank.summarize(self.text, no_below=0)
        summaries: List[str] = self.lexrank.probe()
        self.assertEqual(len(summaries), 3)
        self.assertEqual(summaries[0], "배 감 귤 수박")
        self.assertEqual(summaries[1], "오이 참외 오징어")
        self.assertEqual(summaries[2], "파랑 초록 노랑")


if __name__ == '__main__':
    unittest.main()
