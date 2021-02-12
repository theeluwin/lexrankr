import numpy as np

from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional,
    Callable,
)

from gensim.models import TfidfModel
from sklearn.cluster import (
    Birch,
    DBSCAN,
    AffinityPropagation,
)
from sklearn.feature_extraction import DictVectorizer
from networkx import (
    pagerank,
    Graph,
)

from .sentence import Sentence
from .corpus import SentenceCorpus
from .utils import parse_text_into_sentences


__all__: Tuple[str, ...] = (
    'LexRankError',
    'LexRank',
)


def _get_reordered_summary_texts(summaries: List[Sentence]) -> List[str]:
    summaries = sorted(summaries, key=lambda sentence: sentence.index)
    reordered_summary_texts: List[str] = [sentence.text for sentence in summaries]
    return reordered_summary_texts


class LexRankError(Exception):
    pass


class LexRank:
    """
        Multi-document summarization. A key idea is to decay similarity values with sentence distance, then apply clustering.

        Args:
            tokenizer: A text tokenizer.
            similarity_method: One of 'tfidf_cosine', 'jaccard_index', 'normalized_cooccurrence'.
            similarity_decay_window: Any two sentences with further distance than this value will have 0 similarity.
            similarity_decay_alpha: An exponent for a sentence distance.
            similarity_matrix_diagonal_smoothing: If True, diagonal elements will have maximum value among each row except the respective diagonal value, which is 1.
            clustering_method: One of 'birch', 'dbscan', 'affinity_propagation', or None.
            birch_threshold: A threshold value for BIRCH, if selected.
            birch_n_clusters: The number of clusters for BIRCH, if selected.
            dbscan_eps: A threshold value for DBSCAN, if selected.
            min_cluster_size: Minimum number of sentences for each cluster.
            prune_duplicates: Prune duplicate sentences on the summarization result, if any.
            pruning_threshold: A threshold value for duplicate recognition (uses similarity values).
    """

    def __init__(self,
                 tokenizer: Callable[[str], List[str]],
                 similarity_method: str = 'tfidf_cosine',
                 similarity_decay_window: int = 20,
                 similarity_decay_alpha: float = 0.25,
                 similarity_matrix_diagonal_smoothing: bool = False,
                 clustering_method: Optional[str] = 'birch',
                 birch_threshold: float = 0.75,
                 birch_n_clusters: Optional[int] = None,
                 dbscan_eps: float = 1.0,
                 min_cluster_size: int = 1,
                 prune_duplicates: bool = True,
                 pruning_threshold: float = 0.85,
                 ) -> None:

        # save arguments
        self.tokenizer: Callable[[str], List[str]] = tokenizer
        self.similarity_method: str = similarity_method
        self.similarity_decay_window: int = similarity_decay_window
        self.similarity_decay_alpha: float = similarity_decay_alpha
        self.similarity_matrix_diagonal_smoothing: bool = similarity_matrix_diagonal_smoothing
        self.clustering_method: Optional[str] = clustering_method
        self.birch_threshold: float = birch_threshold
        self.birch_n_clusters: Optional[int] = birch_n_clusters
        self.dbscan_eps: float = dbscan_eps
        self.min_cluster_size: int = min_cluster_size
        self.prune_duplicates: bool = prune_duplicates
        self.pruning_threshold: float = pruning_threshold

        # sim func init
        self._dict_vectorizer: DictVectorizer = DictVectorizer()
        self._similarity_function: Callable[[Sentence, Sentence], float]
        if self.similarity_method == 'tfidf_cosine':
            self._similarity_function = self._similarity_function_tfidf_cosine
        elif self.similarity_method == 'jaccard_index':
            self._similarity_function = self._similarity_function_jaccard_index
        elif self.similarity_method == 'normalized_cooccurrence':
            self._similarity_function = self._similarity_function_normalized_cooccurrence
        else:
            raise LexRankError("Available similarity methods are: 'tfidf_cosine', 'jaccard_index', 'normalized_cooccurrence'.")

        # clustering model init
        self._clustering_model: Optional[Union[Birch, DBSCAN, AffinityPropagation]]
        if clustering_method == 'birch':
            self._clustering_model = Birch(threshold=self.birch_threshold, n_clusters=self.birch_n_clusters)
        elif clustering_method == 'dbscan':
            self._clustering_model = DBSCAN(eps=self.dbscan_eps, min_samples=2)
        elif clustering_method == 'affinity_propagation':
            self._clustering_model = AffinityPropagation()
        elif clustering_method is None:
            self._clustering_model = None
        else:
            raise LexRankError("Available clustering methods are: 'birch', 'dbscan', 'affinity_propagation', or None (no clustering).")

    def _similarity_function_jaccard_index(self, sentence1: Sentence, sentence2: Sentence) -> float:
        if sentence1.index == sentence2.index:
            return 1.0
        p: int = sum((sentence1.counter & sentence2.counter).values())
        q: int = sum((sentence1.counter | sentence2.counter).values())
        try:
            return p / q
        except ZeroDivisionError:
            return 0.0

    def _similarity_function_tfidf_cosine(self, sentence1: Sentence, sentence2: Sentence) -> float:
        if sentence1.index == sentence2.index:
            return 1.0
        sentence1_tfidf_dict: Dict[int, float] = {word_id: tfidf for word_id, tfidf in sentence1.tfidf_pairs}
        sentence2_tfidf_dict: Dict[int, float] = {word_id: tfidf for word_id, tfidf in sentence2.tfidf_pairs}
        tfidf_vec1, tfidf_vec2 = self._dict_vectorizer.fit_transform([sentence1_tfidf_dict, sentence2_tfidf_dict]).toarray()
        return tfidf_vec1 @ tfidf_vec2

    def _similarity_function_normalized_cooccurrence(self, sentence1: Sentence, sentence2: Sentence) -> float:
        if sentence1.index == sentence2.index:
            return 1.0
        p: int = len(set(sentence1.tokens) & set(sentence2.tokens))
        q: float = np.log(len(sentence1.tokens)) + np.log(len(sentence2.tokens))
        try:
            return p / q
        except ZeroDivisionError:
            return 0.0

    def summarize(self,
                  text: str,
                  no_below: int = 3,
                  no_above: float = 0.8,
                  max_size: int = 20000
                  ) -> None:
        """
            Args:
                text: A text to be summarized.
                no_below: See `corpus.SentenceCorpus`.
                no_above: See `corpus.SentenceCorpus`.
                max_size: See `corpus.SentenceCorpus`.
        """

        # save arguments
        self.text: str = text
        self.no_below: int = no_below
        self.no_above: float = no_above
        self.max_size: int = max_size

        # parse text into sentences
        self.sentences: List[Sentence] = parse_text_into_sentences(text, self.tokenizer)
        self.num_sentences: int = len(self.sentences)

        # preprocess sentences
        if self.similarity_method == 'tfidf_cosine':
            self.corpus: SentenceCorpus = SentenceCorpus(
                sentences=self.sentences,
                no_below=self.no_below,
                no_above=self.no_above,
                max_size=self.max_size
            )
            self.tfidf_model: TfidfModel = TfidfModel(self.corpus.bows, id2word=self.corpus.dictionary, normalize=True)
            for index in range(self.num_sentences):
                bow: List[Tuple[int, int]] = self.corpus.bows[index]
                self.sentences[index].bow = bow
                self.sentences[index].tfidf_pairs = self.tfidf_model[bow]

        # build matrix
        self.matrix: np.ndarray = np.zeros((self.num_sentences, self.num_sentences))
        for sentence1 in self.sentences:
            for sentence2 in self.sentences:
                distance: int = abs(sentence1.index - sentence2.index)
                closeness: float = max(self.similarity_decay_window - distance, 0) / self.similarity_decay_window
                decay: float = np.power(closeness, self.similarity_decay_alpha)
                similarity: float = self._similarity_function(sentence1, sentence2)
                self.matrix[sentence1.index, sentence2.index] = decay * similarity
        if self.similarity_matrix_diagonal_smoothing:
            for index in range(self.num_sentences):
                self.matrix[index, index] = 0.0
                self.matrix[index, index] = max(self.matrix[index])

        # run clustering
        index2cids: List[int]
        if self._clustering_model is None:
            index2cids = [0 for _ in range(self.matrix.shape[0])]
        else:
            index2cids = self._clustering_model.fit_predict(1 - self.matrix)
        cid2sentences: Dict[int, List[Sentence]] = {}
        for index, cid in enumerate(index2cids):
            if cid not in cid2sentences:
                cid2sentences[cid] = []
            cid2sentences[cid].append(self.sentences[index])
        self.clusters: List[List[Sentence]] = list(cid2sentences.values())

        # prune duplicate sentences
        if self.prune_duplicates:
            pruned_clusters: List[List[Sentence]] = []
            for cluster in self.clusters:
                cluster_size: int = len(cluster)
                pruned_cluster: List[Sentence] = []
                duplication_table: Dict[int, bool] = {i: False for i in range(cluster_size)}
                for i in range(cluster_size):  # short trick for cluster_size == 1 case
                    if duplication_table[i]:
                        continue
                    for j in range(i + 1, cluster_size):
                        if duplication_table[j]:
                            continue
                        if self._similarity_function(cluster[i], cluster[j]) > self.pruning_threshold:
                            duplication_table[j] = True
                    pruned_cluster.append(cluster[i])
                pruned_clusters.append(pruned_cluster)
            self.clusters = pruned_clusters

        # organize clusters
        self.clusters = [cluster for cluster in self.clusters if len(cluster) >= self.min_cluster_size]
        self.clusters = sorted(self.clusters, key=len, reverse=True)
        self.num_clusters: int = len(self.clusters)
        self.max_k = sum([len(cluster) for cluster in self.clusters])  # the number of effective sentences

        # build graphs and run pagerank
        self.graphs: List[Graph] = []
        for i, cluster in enumerate(self.clusters):
            graph: Graph = Graph()
            graph.add_nodes_from(cluster)
            for sentence1 in cluster:
                for sentence2 in cluster:
                    weight: float = self.matrix[sentence1.index, sentence2.index]
                    if weight:
                        graph.add_edge(sentence1, sentence2, weight=weight)
            sentence2prob: Dict[Sentence, float] = pagerank(graph, weight='weight')
            self.clusters[i] = sorted(sentence2prob, key=sentence2prob.get, reverse=True)
            self.graphs.append(graph)

    def probe(self, k: Union[int, float] = 0) -> List[str]:

        # what a stateful...
        if not hasattr(self, 'clusters'):
            raise LexRankError("You should call `summarize` first.")

        # validate argument
        effective_k: int
        if not k:
            effective_k = self.num_clusters
        elif k < 0:
            raise LexRankError("An appropriate value for `k` is float(0~1) for compress rate, or a int(>1) for the exact number of sentences.")
        elif k > self.max_k:
            effective_k = self.max_k
        elif k < 1:
            effective_k = max(1, int(self.max_k * k))

        # prepare probing
        summaries: List[Sentence] = []
        cid2sizes: np.ndarray = np.array([len(cluster) for cluster in self.clusters])
        cid2steps: np.ndarray = np.zeros(cid2sizes.shape)

        # first round: to ensure that at least one sentence comes from each cluster
        # (note that the clusters are sorted by cluster size)
        for cid, cluster in enumerate(self.clusters):
            summaries.append(cluster[0])
            cid2steps[cid] += 1
            if len(summaries) == effective_k:
                return _get_reordered_summary_texts(summaries)

        # after the first round: all clusters should contribute about equal amount of sentences in the final summary
        while True:
            cid2ahead: np.ndarray = np.array([cid2steps + 1, cid2sizes]).min(axis=0) / cid2sizes  # +1 denotes "the next step"
            selected_cid: int = int(cid2ahead.argmin())
            current_step: int = int(cid2steps[selected_cid])
            if current_step >= int(cid2sizes[selected_cid]):  # fail-safe (this should never happend)
                break
            summaries.append(self.clusters[selected_cid][current_step])
            cid2steps[selected_cid] += 1
            if len(summaries) == effective_k:
                return _get_reordered_summary_texts(summaries)

        # this should never be reached
        return []
