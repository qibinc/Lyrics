from attribute.extractor import Extractor
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

class KeywordExtractor(Extractor):
    """Base class of extracting keywords"""

    def get_keywords(self, n_keywords = 1):
        pass

class TFIDFKeywordExtractor(KeywordExtractor):
    """Use tf-idf to extract keywords from each bag of words

    :param bags: a list of numpy arrays representing bags of words.
    :type bags: list

    .. code-block:: python

        import itertools
        from utils import Doc
        Doc.load()
        bags = itertools.chain.from_iterable([doc.get_lines() for doc in Doc.get_corpus()])
        TFIDFKeywordExtractor(bags, len(Doc.get_vocab()))
    """
    def __init__(self, bags, vocab_size):
        rows = []
        cols = []
        data = []
        for rol, bag in enumerate(bags):
            unique, counts = np.unique(bag, return_counts=True)
            for idx, count in zip(unique, counts):
                rows.append(rol)
                cols.append(idx)
                data.append(count)

        mtx = sparse.csr_matrix((data, (rows, cols)), shape=(len(bags), vocab_size))

        self.__samples = len(bags)
        self.transformer = TfidfTransformer(smooth_idf=True)
        self.fit = sparse.coo_matrix(self.transformer.fit_transform(mtx))

    def get_keywords(self, n_keywords = 1):
        """Return the keywords of input bags.

        Warning: not all the return values have exact n_keywords
        :param n_keywords: number of keywords want to extract, defaults to 1
        :type n_keywords: number, optional
        """
        maps = [[] for i in range(self.__samples)]
        for (row, col, data) in zip(self.fit.row, self.fit.col, self.fit.data):
            maps[row].append((col, data))

        keywords = []
        for bag_idx in range(self.__samples):
            try:
                indices, data = zip(*(maps[bag_idx]))
                indices = np.array(indices)
                keywords.append(indices[np.argsort(data)[::-1][:n_keywords]])
            except ValueError:
                keywords.append(np.array([]))
        return keywords

