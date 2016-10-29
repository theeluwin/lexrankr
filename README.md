LexRank for Korean
==========

Text summarization using [LexRank][1] algorithm for Korean.
Click [here][2] to see how to install [KoNLPy][3] properly.
[older version][4] using [TextRank][5]


Installation
-----

```sh
pip install lexrankr
```

Usage
-----

```python
from __future__ import print_function
from lexrankr import LexRank

lexrank = LexRank()  # can init with various settings
lexrank.summarize(your_text_here)
summaries = lexrank.probe(num_summaries)  # `num_summaries` can be `None` (using auto-detected topics)
for summary in summaries:
    print(summary)
```


Test
-----

```bash
python -m tests.test
```

[1]: http://dl.acm.org/citation.cfm?id=1622501
[2]: http://konlpy.org/en/latest/install/
[3]: http://konlpy.org/
[4]: https://github.com/theeluwin/textrankr
[5]: http://digital.library.unt.edu/ark:/67531/metadc30962/
