from attribute.extractor import Extractor
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
import itertools

class KeywordExtractor(Extractor):
    """Base class of extracting keywords"""

    def get_keywords(self):
        pass

class TFIDFKeywordExtractor(KeywordExtractor):
    """Use tf-idf to extract keywords from each bag of words

    :param list_of_list_of_lines: a list of list of numpy arrays representing list_of_list_of_lines of words.
    :type list_of_list_of_lines: list
    :param n_keywords: number of keywords want to extract, defaults to 1
    :type n_keywords: number, optional

    .. code-block:: python

        from utils import Doc
        Doc.load()
        list_of_list_of_lines = [doc.get_lines() for doc in Doc.get_corpus()]
        TFIDFKeywordExtractor(list_of_list_of_lines, len(Doc.get_vocab()))
    """
    def __init__(self, list_of_list_of_lines, vocab_size, n_keywords=1):
        self.__list_of_lines_sizes = [len(list_of_lines) for list_of_lines in list_of_list_of_lines]
        list_of_list_of_lines = list(itertools.chain.from_iterable(list_of_list_of_lines))
        rows = []
        cols = []
        data = []
        for rol, bag in enumerate(list_of_list_of_lines):
            unique, counts = np.unique(bag, return_counts=True)
            for idx, count in zip(unique, counts):
                rows.append(rol)
                cols.append(idx)
                data.append(count)

        mtx = sparse.csr_matrix((data, (rows, cols)), shape=(len(list_of_list_of_lines), vocab_size))

        samples = len(list_of_list_of_lines)
        transformer = TfidfTransformer(smooth_idf=True)
        fit = sparse.coo_matrix(transformer.fit_transform(mtx))

        maps = [[] for i in range(samples)]
        for (row, col, data) in zip(fit.row, fit.col, fit.data):
            maps[row].append((col, data))

        self.keywords = []
        for bag_idx in range(samples):
            try:
                indices, data = zip(*(maps[bag_idx]))
                indices = np.array(indices)
                self.keywords.append(indices[np.argsort(data)[::-1][:n_keywords]])
            except ValueError:
                self.keywords.append(np.array([]))

    def get_keywords(self):
        """Return the keywords of input list_of_list_of_lines.

        Warning: not all the return values have exact n_keywords

        :returns: a list looks like the input. Except that the sequence of sentence now become a list of keywords.
        """
        # Organize keywords to a list of list of keywords.
        keywords = []
        t = 0
        for i, size in enumerate(self.__list_of_lines_sizes):
            keywords.append([])
            for j in range(size):
                keywords[i].append(self.keywords[t + j])
            t += size
        return keywords

