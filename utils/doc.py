import numpy as np
import thulac
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import os
path_saved = os.path.join(os.path.dirname(__file__), '../saved')
path_word2vec_model = os.path.join(path_saved, 'word2vec_model')

class Doc():
    """Operate lyrics as documents.

    Use ``Doc.load_corpus(lyrics)`` to initialize the corpus.

    .. code-block:: python

        from utils import Doc, read
        dev_lyrcs = read()[:500]
        Doc.load_corpus(dev_lyrics)

    In case you have to create an instance, use Doc(lyric, depth, tokenizer).

    Hierarchy in the lyric will be parsed(recursively) and can be accessed by ``doc.get_children()``.

    :param lyric: a piece of lyric, list of sentence
    :param depth: used internally for hierarchy, default to 0
    :param tokenizer: the tokenizer(char/word), default to word

    .. code-block:: python

        Doc(lyric='Hello world\\nGoodbye world', tokenizer='word')
    """

    __corpus = []

    __model = Word2Vec.load(path_word2vec_model)
    __cut = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注\n"

    __filter_list = [' ', '\n']
    __hierarchy = ['\n']
    __vocab = None
    __idx2word = None
    __special_words = {
        'SOS_token':0,
        'EOS_token':1,
        'PAD_token':2
        }

    def __init__(self, lyric, depth=0, tokenizer='word'):
        """Tokenize a string and store as bag of words.

        Besides, it automatically parse it and create hierarchy.
        """
        self.origin = lyric

        self.children = [Doc(lyric=line, depth=depth+1)\
                     for line in self.origin.split(self.__hierarchy[depth])]\
                if depth < len(self.__hierarchy) else []
        if self.children == []:
            if tokenizer == 'char':
                self.bag = list(self.origin)
            if tokenizer == 'word':
                try:
                    self.bag = list(list(zip(*Doc.__cut.cut(self.origin)))[0])
                except Exception:
                    self.bag = []
            self.bag = [word for word in self.bag if word not in Doc.__filter_list]
        else:
            self.bag = []
            for child in self.children:
                self.bag += child.bag

    @classmethod
    def load_corpus(cls, lyrics, vocab_size=10000, unk_percentage=0.08, tokenizer='word'):
        """Load from lyrics read by utils.data.read()

        :param lyrics: a list of original lyrics texts
        :type lyrics: list
        :param vocab_size: the vocabulary size
        :type vocab_size: int
        :param unk_percentage: maximum percentage of unknown token in a doc
        :type unk_percentage: float
        :param tokenizer: char / word, defaults to 'word'
        :type tokenizer: str, optional
        """
        cls.__corpus = []
        for idx, lyric in enumerate(lyrics):
            new_doc = Doc(lyric=lyric, tokenizer=tokenizer)
            cls.__corpus.append(new_doc)

            if idx % (len(lyrics) / 10)== 0:
                print('%.2f%% Loaded.' % (100 * idx / len(lyrics)))

        print('Loading complete')
        cls.__filter_corpus(vocab_size, unk_percentage)

    @classmethod
    def get_corpus(cls):
        """:returns: The list of the Doc() instances created by load_corpus()"""
        return cls.__corpus

    @classmethod
    def get_vocab(cls):
        """:returns: the vocabulary dict of the corpus."""
        if cls.__vocab is None:
            print('Warning: no vocab please load corpus first.')
        else:
            return cls.__vocab

    @classmethod
    def idxs_to_text(cls, idxs):
        """Return the text of idxs.

        Convert a numpy array to a sentence.
        Or a list of numpy arrays to a piece of lyrics

        :param idxs: a numpy array or a list of numpy array

        .. code-block:: python

            doc = Doc.get_corpus()[0]
            print(Doc.idxs_to_text(doc.get_bag()))
            print(Doc.idxs_to_text(doc.get_lines()))
        """
        if type(idxs) is list:
            return [
                ''.join([cls.__idx2word[idx] for idx in sentence_idxs])
                for sentence_idxs in idxs
                ]
        else:
            return ''.join([cls.__idx2word[idx] for idx in idxs])

    def get_origin(self):
        """:returns: the original text of this doc"""
        return self.origin

    def get_bag(self):
        """:returns: a list(bag) of indexs(int) of this doc"""
        return self.bag

    def get_lines(self):
        """:returns: a list of list(bag) of indexs(int) in the lyrics."""
        return [child.get_bag() for child in self.children]

    def embedding(self):
        """:returns: embedding of the doc as bag of words using pre-trained model."""
        vec = np.zeros([512], dtype='float')
        for word in self.get_words():
            if word in Doc.__model.wv:
                vec += Doc.__model.wv[word]
            else:
                for char in word:
                    vec += Doc.__model.wv[char] if char in Doc.__model.wv else 0

        return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec

    def similarity(self, doc2):
        """:returns: the cosine distance between two lines."""
        return np.dot(self.embedding(), doc2.embedding())

    @classmethod
    def __filter_corpus(cls, vocab_size, unk_percentage):
        print('Filtering out rare words.')

        texts = [' '.join(document.bag) for document in cls.__corpus]

        vectorizer = CountVectorizer(token_pattern='\\b\\w+\\b', max_features=(vocab_size-len(cls.__special_words))
        vectorizer.fit_transform(texts)

        cls.__vocab = vectorizer.vocabulary_
        for k,v in cls.__vocab.items():
            cls.__vocab.update({k,(v+len(cls.__special_words))})
        cls.__vocab = dict(cls.__special_words,**cls.__vocab)
        cls.__idx2word = {v: k for k, v in cls.__vocab.items()}

        corpus_size = len(cls.__corpus)

        new_corpus = []
        for doc in cls.__corpus:
            n_word = len(doc.get_bag())
            doc.__filter()
            if len(doc.get_bag()) / n_word > 1 - unk_percentage:
                new_corpus.append(doc)

        cls.__corpus = [doc for doc in new_corpus if len(doc.get_lines()) > 15]

        print('Complete ')
        print('Vocab size %d' % len(vectorizer.vocabulary_))
        print('Filtered %.2f%% too short or too many unks.' % (100 * (1 - len(cls.__corpus) / corpus_size)))

    def __filter(self):
        if self.children != []:
            for child in self.children:
                child.__filter()
            self.bag = np.concatenate(self.get_lines())
        else:
            self.bag = np.array([self.__vocab[word] for word in self.bag if word in self.__vocab])

        self.children = [child for child in self.children]

    @classmethod
    def dump(cls):
        """``Save everything and next time use load() instead of load_corpus() to save time!!``"""
        with open(os.path.join(path_saved, 'Doc_corpus'), 'wb') as f:
            pickle.dump(cls.__corpus, f)
        with open(os.path.join(path_saved, 'Doc_vocab'), 'wb') as f:
            pickle.dump(cls.__vocab, f)

    @classmethod
    def load(cls):
        """``Load things back!!``"""
        if not os.path.exists(os.path.join(path_saved, 'Doc_corpus'))\
            or not os.path.exists(os.path.join(path_saved, 'Doc_vocab')):
            print('Doc.load() Error! Nothing saved!\
                You need to run Doc.load_corpus(..) at least once\
                and call Doc.save() aftere that.\
                Then you can use Doc.load() in the future')
            return

        with open(os.path.join(path_saved, 'Doc_corpus'), 'rb') as f:
            cls.__corpus = pickle.load(f)
        with open(os.path.join(path_saved, 'Doc_vocab'), 'rb') as f:
            cls.__vocab= pickle.load(f)
            cls.__idx2word = {v: k for k, v in cls.__vocab.items()}

    @classmethod
    def test(cls):
        """Unit test and usage"""
        cls.load_corpus(['天青色 等烟雨\n而我在等你' * 20, '故事的小黄花\n从出生那年就飘着' * 20])

        corpus = cls.get_corpus()

        try:
            assert corpus[0].get_lines() is not []
            assert corpus[1].get_bag() is not []
            assert corpus[0].similarity(corpus[1]) > 0.1
            assert len(cls.get_vocab()) > 10
        except Exception:
            print(corpus[0].get_lines())
            print(corpus[1].get_bag())
            print(corpus[0].similarity)
            print(cls.get_vocab())
