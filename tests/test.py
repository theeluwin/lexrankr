#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import unittest

from lexrankr import LexRank


class TestLexRank(unittest.TestCase):

    def setUp(self):
        self.text = "사과 배 감 귤. 배 감 귤 수박. 감 귤 수박 딸기. 오이 참외 오징어. 참외 오징어 달팽이."
        self.lexrank = LexRank(min_keyword_length=0, no_below_word_count=0, min_cluster_size=1)

    def test_summarized(self):
        self.lexrank.summarize(self.text)
        summaries = self.lexrank.probe()
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0], "배 감 귤 수박")


if __name__ == '__main__':
    unittest.main()
