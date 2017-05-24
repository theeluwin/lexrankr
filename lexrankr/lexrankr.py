# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import re
import math
import networkx
import numpy as np

from konlpy import tag as taggers
from collections import Counter
from gensim.corpora import Dictionary, TextCorpus
from gensim.models import TfidfModel
from sklearn.cluster import Birch, DBSCAN, AffinityPropagation
from sklearn.feature_extraction import DictVectorizer


stopwords_ko = ["저", "것", "동시에", "몇", "고려하면", "관련이", "놀라다", "무엇", "어느쪽", "오", "정도의", "더구나", "아무도", "줄은모른다", "참", "아니", "휘익", "향하다", "응당", "알겠는가", "인젠", "그래서", "자신", "해서는", "둘", "이었다", "임에", "하도록시키다", "누구", "이때", "삼", "제외하고", "쿵", "하면", "좀", "그렇지않으면", "아니었다면", "이라면", "팍", "일", "통하여", "무엇때문에", "보아", "하게하다", "하는", "이르다", "타다", "까지도", "오직", "도달하다", "잠깐", "외에", "심지어", "하려고하다", "게다가", "후", "알", "비하면", "헉헉", "근거로", "월", "따라서", "않는다면", "일지라도", "함께", "이유는", "흥", "혼자", "관하여", "붕붕", "하다", "진짜로", "의해", "바와같이", "대하면", "퍽", "보다더", "그렇게", "끼익", "댕그", "시초에", "당장", "하는것만", "누가", "만이", "만일", "이지만", "하마터면", "꽈당", "만은", "우선", "없다", "휴", "하도록하다", "그런데", "비로소", "하게될것이다", "만큼 어찌됏든", "오히려", "을", "더라도", "안", "왜냐하면", "습니다", "줄은", "그리하여", "하", "어떻게", "대로", "기대여", "끙끙", "예를", "와르르", "이리하여", "이", "조차", "하고", "이젠", "뒤이어", "할줄알다", "반대로", "시각", "펄렁", "잇따라", "공동으로", "비록", "가까스로", "여덟", "비슷하다", "이상", "차라리", "이어서", "모두", "툭", "조차도", "헉", "부터", "혹시", "않고", "우리", "삐걱", "여보시오", "허", "해요", "견지에서", "하기는한데", "토하다", "않으면", "이봐", "관계가", "한다면", "시작하여", "연이서", "이외에도", "그", "운운", "에게", "그럼에도", "예", "만약에", "했어요", "결과에", "제", "오자마자", "것들", "약간", "것과", "일때", "셋", "각종", "아이구", "같은", "향해서", "일것이다", "해야한다", "아이야", "로", "편이", "등등", "해도좋다", "하기에", "김에", "몰랏다", "같이", "하도다", "즉시", "갖고말하자면", "우에", "어느", "허허", "하자마자", "에서", "그래도", "하여야", "된이상", "까악", "한켠으로는", "많은", "그중에서", "사", "낼", "뿐만", "저쪽", "어쩔수", "어떤것들", "물론", "결론을", "이만큼", "이렇게되면", "소인", "바꾸어말하면", "들", "이렇구나", "하물며", "얼마간", "얼마든지", "한항목", "하는것도", "졸졸", "한마디", "말할것도", "만약", "남들", "총적으로", "허걱", "그리고", "따지지", "구체적으로", "못하다    하기보다는", "언제", "따르는", "구토하다", "앞에서", "대해서", "아", "앞의것", "비걱거리다", "헐떡헐떡", "어찌하든지", "입장에서", "의", "마저", "바로", "하기만", "않기", "또한", "쓰여", "위해서", "의거하여", "인", "아니면", "를", "사람들", "할수있다", "일곱", "근거하여", "한적이있다", "함으로써", "낫다", "어떤것", "방면으로", "중의하나", "어", "무릎쓰고", "저것만큼", "서술한바와같이", "그런즉", "들자면", "하지", "아이고", "불문하고", "만", "마저도", "얼마만큼", "예컨대", "이렇게말하자면", "연관되다", "않다면", "들면", "이쪽", "의지하여", "여섯", "그저", "아니다", "그렇지만", "기준으로", "되어", "가", "무렵", "즉", "말하면", "어찌", "그럼", "그위에", "그런", "조금", "매번", "혹은", "이천구", "중에서", "따름이다", "하기", "가령", "잠시", "아무거나", "하기보다는", "주저하지", "당신", "봐라", "그렇지", "자기집", "할지라도", "요만한걸", "우르르", "못하다", "왜", "이렇게", "퉤", "관계없이", "그래", "대해", "쪽으로", "저것", "자기", "아홉", "지만", "구", "하지마", "따위", "하지만", "나", "해도", "전자", "그만이다", "안된다", "까닭으로", "되다", "오르다", "딱", "다음에", "너희들", "점에서", "아이쿠", "쾅쾅", "종합한것과같이", "할수있어", "그치지", "비교적", "륙", "되는", "개의치않고", "엉엉", "하든지", "때가", "영차", "바꿔", "더불어", "주룩주룩", "따라", "이용하여", "우리들", "여기", "더욱이는", "하더라도", "입각하여", "여러분", "마치", "하느니", "너", "어디", "제각기", "밖에", "봐", "위하여", "팔", "요만큼", "가서", "아니라면", "지든지", "참나", "할만하다", "타인", "든간에", "하겠는가", "거바", "겨우", "다음", "이러한", "이럴정도로", "각자", "어때", "지말고", "형식으로", "그러한즉", "아니나다를가", "할", "불구하고", "지경이다", "어떠한", "기점으로", "할때", "등", "다시", "시키다", "답다", "소생", "라", "로써", "각", "부류의", "알았어", "훨씬", "위에서", "뿐이다", "시간", "그러나", "하곤하였다", "일단", "막론하고", "좋아", "솨", "이곳", "뿐만아니라", "아울러", "옆사람", "다수", "예하면", "령", "어떤", "어떻해", "할수록", "말하자면", "전후", "메쓰겁다", "에", "으로써", "이번", "하면된다", "이것", "딩동", "양자", "달려", "본대로", "탕탕", "마음대로", "쉿", "미치다", "다시말하면", "동안", "그러니까", "과연", "뚝뚝", "거의", "이천팔", "이로", "않도록", "또", "한하다", "아래윗", "수", "다소", "어느것", "까지", "남짓", "저기", "관한", "무슨", "그에", "년도", "삐걱거리다", "이러이러하다", "와", "넷", "쳇", "논하지", "습니까", "이천육", "기타", "오로지", "어느곳", "설령", "할지언정", "칠", "다만", "반드시", "한데", "곧", "의해서", "얼마나", "아니라", "상대적으로", "너희", "있다", "인하여", "다섯", "생각이다", "몰라도", "정도에", "버금", "까닭에", "얼마큼", "전부", "로부터", "힘입어", "틈타", "해도된다", "나머지는", "흐흐", "그때", "하여금", "모", "이런", "바꾸어서", "비추어", "각각", "설사", "이래", "비길수", "하지마라", "응", "다른", "듯하다", "보는데서", "어쨋든", "대하여", "좍좍", "으로", "여차", "틀림없다", "과", "고로", "요컨대", "일반적으로", "줄", "하는바", "그들", "요만한", "윙윙", "콸콸", "어기여차", "언젠가", "이와", "할망정", "이천칠", "네", "없고", "둥둥", "겸사겸사", "그러므로", "안다", "거니와", "년", "여부", "때문에", "된바에야", "향하여", "때", "하하", "및", "오호", "하면서", "더군다나", "한", "이유만으로", "어이", "하나", "저희", "더욱더", "두번째로", "바꾸어말하자면", "이와같다면", "이르기까지", "단지", "그러면", "야", "결국", "영", "뒤따라", "즈음하여", "도착하다", "와아", "다음으로", "같다", "자", "아하", "생각한대로", "외에도", "의해되다", "설마", "으로서", "보면", "할뿐", "첫번째로", "아야", "어째서", "하는것이", "하구나", "않다", "힘이", "육", "그러니", "여전히", "어찌됏어", "어찌하여", "어느해", "앗", "게우다", "보드득", "관해서는", "자마자", "매", "하고있었다", "어느때", "여", "실로", "해봐요", "얼마", "아이"]


class LexRankError(Exception):
    pass


class Sentence(object):

    def __init__(self, text, tokens=[], index=0):
        self.index = index
        self.text = text
        self.tokens = tokens
        self.counter = Counter(self.tokens)

    def __unicode__(self):
        return self.text

    def __str__(self):
        return str(self.index)

    def __repr__(self):
        try:
            return self.text.encode('utf-8')
        except:
            return self.text

    def __eq__(self, another):
        return hasattr(another, 'index') and self.index == another.index

    def __hash__(self):
        return self.index


class SentenceFactory(object):

    def __init__(self, tagger, useful_tags, delimiters, min_token_length, stopwords, **kwargs):
        if tagger == 'twitter':
            self.tagger = taggers.Twitter()
            self.tagger_options = {
                'norm': bool(kwargs.get('norm', True)),
                'stem': bool(kwargs.get('stem', True)),
            }
        elif tagger == 'komoran':
            self.tagger = taggers.Komoran()
            self.tagger_options = {
                'flatten': bool(kwargs.get('flatten', True)),
            }
        elif tagger == 'hannanum':
            self.tagger = taggers.Hannanum()
            self.tagger_options = {
                'ntags': int(kwargs.get('ntags', 9)),
                'flatten': bool(kwargs.get('flatten', True)),
            }
        elif tagger == 'kkma':
            self.tagger = taggers.Kkma()
            self.tagger_options = {
                'flatten': bool(kwargs.get('flatten', True)),
            }
        elif tagger == 'mecab':
            self.tagger = taggers.Mecab()
            self.tagger_options = {
                'flatten': bool(kwargs.get('flatten', True)),
            }
        else:
            raise LexRankError("available taggers are: twitter, komoran, hannanum, kkma, mecab")
        self.useful_tags = useful_tags
        self.delimiters = delimiters
        self.stopwords = stopwords
        self.min_token_length = min_token_length
        self.splitter = self.splitterer()
        self.pos = lambda text: self.tagger.pos(text, **self.tagger_options)

    def splitterer(self):
        escaped_delimiters = '|'.join([re.escape(delimiter) for delimiter in self.delimiters])
        return lambda value: re.split(escaped_delimiters, value)

    def text2tokens(self, text):
        tokens = []
        word_tag_pairs = self.pos(text)
        for word, tag in word_tag_pairs:
            if word in self.stopwords:
                continue
            if tag not in self.useful_tags:
                continue
            tokens.append("{}/{}".format(word, tag))
        return tokens

    def text2sentences(self, text):
        candidates = self.splitter(text.strip())
        sentences = []
        index = 0
        for candidate in candidates:
            while len(candidate) and (candidate[-1] == '.' or candidate[-1] == ' '):
                candidate = candidate.strip(' ').strip('.')
            if not candidate:
                continue
            tokens = self.text2tokens(candidate)
            if len(tokens) < self.min_token_length:
                continue
            sentence = Sentence(candidate, tokens, index)
            sentences.append(sentence)
            index += 1
        return sentences


class SentenceCorpus(TextCorpus):

    def __init__(self, sentences, no_below=3, no_above=0.8, max_size=None):
        self.metadata = False
        self.sentences = sentences
        self.dictionary = Dictionary(self.get_texts(), prune_at=max_size)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=max_size)
        self.dictionary.compactify()
        self.bows = [self.dictionary.doc2bow(tokens) for tokens in self.get_texts()]

    def get_texts(self):
        for sentence in self.sentences:
            yield sentence.tokens


class LexRank(object):

    def __init__(self, similarity='cosine', decay_window=20, decay_alpha=0.25, clustering='dbscan', tagger='twitter', useful_tags=['Noun', 'Verb', 'Adjective', 'Determiner', 'Adverb', 'Conjunction', 'Josa', 'PreEomi', 'Eomi', 'Suffix', 'Alpha', 'Number'], delimiters=['. ', '\n', '.\n'], min_token_length=2, stopwords=stopwords_ko, no_below_word_count=2, no_above_word_portion=0.85, max_dictionary_size=None, min_cluster_size=2, similarity_threshold=0.85, matrix_smoothing=False, n_clusters=None, compactify=True, **kwargs):
        self.decay_window = decay_window
        self.decay_alpha = decay_alpha
        if similarity == 'cosine':  # very, very slow :(
            self.vectorizer = DictVectorizer()
            self.uniform_sim = self._sim_cosine
        elif similarity == 'jaccard':
            self.uniform_sim = self._sim_jaccard
        elif similarity == 'normalized_cooccurrence':
            self.uniform_sim = self._sim_normalized_cooccurrence
        else:
            raise LexRankError("available similarity functions are: cosine, jaccard, normalized_cooccurrence")
        self.sim = lambda sentence1, sentence2: self.decay(sentence1, sentence2) * self.uniform_sim(sentence1, sentence2)
        self.factory = SentenceFactory(tagger=tagger, useful_tags=useful_tags, delimiters=delimiters, min_token_length=min_token_length, stopwords=stopwords, **kwargs)
        if clustering == 'birch':
            self._birch = Birch(threshold=0.99, n_clusters=n_clusters)
            self._clusterer = lambda matrix: self._birch.fit_predict(1 - matrix)
        elif clustering == 'dbscan':
            self._dbscan = DBSCAN()
            self._clusterer = lambda matrix: self._dbscan.fit_predict(1 - matrix)
        elif clustering == 'affinity':
            self._affinity = AffinityPropagation()
            self._clusterer = lambda matrix: self._affinity.fit_predict(1 - matrix)
        elif clustering is None:
            self._clusterer = lambda matrix: [0 for index in range(matrix.shape[0])]
        else:
            raise LexRankError("available clustering algorithms are: birch, markov, no-clustering(use `None`)")
        self.no_below_word_count = no_below_word_count
        self.no_above_word_portion = no_above_word_portion
        self.max_dictionary_size = max_dictionary_size
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.matrix_smoothing = matrix_smoothing
        self.compactify = compactify

    def summarize(self, text):
        self.sentences = self.factory.text2sentences(text)
        self.num_sentences = len(self.sentences)
        self.corpus = SentenceCorpus(self.sentences, self.no_below_word_count, self.no_above_word_portion, self.max_dictionary_size)
        self.model = TfidfModel(self.corpus.bows, id2word=self.corpus.dictionary, normalize=True)
        self.tfidfs = self.model[self.corpus.bows]
        self._inject_tfidfs()
        self._build_matrix()
        self._clustering()
        if self.compactify:
            self._compactify()
        self.graphs = []
        for i in range(self.num_clusters):
            graph = self.sentences2graph(self.clusters[i])
            pagerank = networkx.pagerank(graph, weight='weight')
            self.clusters[i] = sorted(pagerank, key=pagerank.get, reverse=True)
            self.graphs.append(graph)

    def _sim_jaccard(self, sentence1, sentence2):
        if sentence1 == sentence2:
            return 1
        p = sum((sentence1.counter & sentence2.counter).values())
        q = sum((sentence1.counter | sentence2.counter).values())
        return p / q if q else 0

    def _sim_cosine(self, sentence1, sentence2):
        if sentence1 == sentence2:
            return 1
        sentence1_tfidf = {word_id: tfidf for word_id, tfidf in sentence1.tfidf}
        sentence2_tfidf = {word_id: tfidf for word_id, tfidf in sentence2.tfidf}
        vector1, vector2 = self.vectorizer.fit_transform([sentence1_tfidf, sentence2_tfidf]).toarray()
        return vector1.dot(vector2)

    def _sim_normalized_cooccurrence(self, sentence1, sentence2):
        if sentence1 == sentence2:
            return 1
        return len(set(sentence1.tokens) & set(sentence2.tokens)) / (math.log(len(sentence1.tokens)) + math.log(len(sentence2.tokens)))

    def decay(self, sentence1, sentence2):
        distance = abs(sentence1.index - sentence2.index)
        closeness = max(self.decay_window - distance, 0) / self.decay_window
        return math.pow(closeness, self.decay_alpha)

    def _inject_tfidfs(self):
        for index in range(self.num_sentences):
            bow = self.corpus.bows[index]
            self.sentences[index].bow = bow
            self.sentences[index].tfidf = self.model[bow]

    def _build_matrix(self):
        self.matrix = np.zeros((self.num_sentences, self.num_sentences))
        for sentence1 in self.sentences:
            for sentence2 in self.sentences:
                self.matrix[sentence1.index, sentence2.index] = self.sim(sentence1, sentence2)
        if self.matrix_smoothing:
            for index in range(self.num_sentences):
                self.matrix[index, index] = 0
                self.matrix[index, index] = max(self.matrix[index])

    def sentences2graph(self, sentences):
        graph = networkx.Graph()
        graph.add_nodes_from(sentences)
        for sentence1 in sentences:
            for sentence2 in sentences:
                weight = self.matrix[sentence1.index, sentence2.index]
                if weight:
                    graph.add_edge(sentence1, sentence2, weight=weight)
        return graph

    def _clustered(self):
        self.clusters = [cluster for cluster in self.clusters if len(cluster) >= self.min_cluster_size]
        self.num_clusters = len(self.clusters)
        self.clusters = sorted(self.clusters, key=lambda cluster: len(cluster), reverse=True)

    def _clustering(self):
        cls = self._clusterer(self.matrix)
        bucket = {}
        for index in range(len(cls)):
            key = str(cls[index])
            if key not in bucket:
                bucket[key] = []
            bucket[key].append(self.sentences[index])
        self.clusters = bucket.values()
        self._clustered()

    def _compactify(self):
        clusters = []
        for cluster in self.clusters:
            compact_cluster = []
            cluster_size = len(cluster)
            for i in range(cluster_size):
                cluster[i].duplicated = False
            for i in range(cluster_size):
                if cluster[i].duplicated:
                    continue
                for j in range(i + 1, cluster_size):
                    if cluster[j].duplicated:
                        continue
                    if self.uniform_sim(cluster[i], cluster[j]) > self.similarity_threshold:
                        cluster[j].duplicated = True
                compact_cluster.append(cluster[i])
            clusters.append(compact_cluster)
        self.clusters = clusters
        self._clustered()

    def _verbose(self):
        summaries = sorted(self.summaries, key=lambda sentence: sentence.index)
        return [sentence.text for sentence in summaries]

    def probe(self, k=None):
        if not hasattr(self, 'clusters'):
            raise LexRankError("summarize it first")
        if not k:
            k = max(2, self.num_clusters)
        if k < 0:
            raise LexRankError("appropriate value for `k`: float(0 ~ 1) for compress rate, or natural number for exact number of sentences")
        if k > self.num_sentences:
            raise LexRankError("this will not give a summarization")
        if k < 1:
            k = int(self.num_sentences * k)
        self.summaries = []
        ends = np.array([len(cluster) for cluster in self.clusters])
        drones = np.zeros(ends.shape)
        for i in range(self.num_clusters):
            self.summaries.append(self.clusters[i][0])
            drones[i] += 1
            if len(self.summaries) == k:
                return self._verbose()
        while True:
            branch = np.array([drones + 1, ends]).min(axis=0) / ends
            leach = int(branch.argmin())
            drone = int(drones[leach])
            self.summaries.append(self.clusters[leach][drone])
            drones[leach] += 1
            if len(self.summaries) == k:
                return self._verbose()
