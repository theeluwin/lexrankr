# lexrankr

[![Build Status](https://travis-ci.org/theeluwin/lexrankr.svg?branch=main)](https://travis-ci.org/theeluwin/lexrankr)
[![Coverage Status](https://coveralls.io/repos/github/theeluwin/lexrankr/badge.svg?branch=main)](https://coveralls.io/github/theeluwin/lexrankr?branch=main)
[![PyPI version](https://badge.fury.io/py/lexrankr.svg)](https://badge.fury.io/py/lexrankr)

Clustering based multi-document selective text summarization using [LexRank](http://dl.acm.org/citation.cfm?id=1622501) algorithm.

This repository is a source code for the paper [설진석, 이상구. "lexrankr: LexRank 기반 한국어 다중 문서 요약." 한국정보과학회 학술발표논문집 (2016): 458-460](http://www.eiric.or.kr/community/post2.php?m=view&gubun=201612&num=6769).

* Mostly designed for Korean, but not limited to.
* Click [here](http://konlpy.org/en/latest/install/) to see how to install [KoNLPy](http://konlpy.org/) properly.
* Check out [textrankr](https://github.com/theeluwin/textrankr), which is a simpler summarizer using [TextRank](http://digital.library.unt.edu/ark:/67531/metadc30962/).

## Installation

```bash
pip install lexrankr
```

## Tokenizers

Tokenizers are not included. You have to implement one by yourself.

Example:

```python
from typing import List

class MyTokenizer:
    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = text.split()
        return tokens
```

한국어의 경우 [KoNLPy](http://konlpy.org)를 사용하는 방법이 있습니다.

```python
from typing import List
from konlpy.tag import Okt

class OktTokenizer:
    okt: Okt = Okt()

    def __call__(self, text: str) -> List[str]:
        tokens: List[str] = self.okt.pos(text, norm=True, stem=True, join=True)
        return tokens
```

## Usage

```python
from typing import List
from lexrankr import LexRank

# 1. init
mytokenizer: MyTokenizer = MyTokenizer()
lexrank: LexRank = LexRank(mytokenizer)

# 2. summarize (like, pre-computation)
lexrank.summarize(your_text_here)

# 3. probe (like, query-time)
summaries: List[str] = lexrank.probe()
for summary in summaries:
    print(summary)
```

## Test

Use docker.

```bash
docker build -t lexrankr -f Dockerfile .
docker run --rm -it lexrankr
```
