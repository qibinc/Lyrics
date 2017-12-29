import numpy as np
import thulac
from gensim.models import Word2Vec
import os
from utils import Doc

# path_saved = os.path.join(os.path.dirname(__file__), '../saved')
path_saved = '../saved'
Doc.load()
path_word2vec_model = os.path.join(path_saved, 'word2vec_model')
Word2Vec.load(path_word2vec_model)
model = Word2Vec.load(path_word2vec_model)
cut_text = thulac.thulac(seg_only=True)

def word_embedding(text):
    """Returns the embedding of given text based on word2vec model

        :param word: text in string format

        .. code-block:: python

            # try:
            word_embedding('理想是丰满的')
    """
    bag_of_words = list(list(zip(*cut_text.cut(text)))[0])
    vec = np.zeros([512], dtype='float')
    for word in bag_of_words:
        if word in model.wv:
            vec += model.wv[word]
        else:
            for char in word:
                vec += model.wv[char] if char in model.wv else 0
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec

def word_similarity(text1,text2):
    """:returns: the cosine distance between two texts."""
    return np.dot(word_embedding(text1),word_embedding(text2))

def sim_word(input_text, num):
    """Returns the top num words(in Doc.get_vocab()) most similar to given text

        :param input_text: text in string format
        :param num: int

        .. code-block:: python

            # try:
            sim_word('河流'， 10)
    """
    vocab = list(Doc.get_vocab())
    sim_array = np.array([word_similarity(input_text,x) for x in vocab])
    total_index = np.argsort(-sim_array)
    total_sim = -np.sort(-sim_array)
    return [np.array(vocab)[total_index[:num]],total_sim[:num]]

if __name__ == '__main__':
    sim_word('我看到了我的爱恋',60)