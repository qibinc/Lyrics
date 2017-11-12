import numpy as np
from gensim.models import Word2Vec
import thulac
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import os
path_saved = os.path.join(os.path.dirname(__file__), '../saved')
path_word2vec_model = os.path.join(path_saved, 'word2vec_model')

class Doc():
    '''Operate lyrics.

    Use Doc.load_corpus(lyrics) to initialize the corpus
    '''
    model = Word2Vec.load(path_word2vec_model)
    cut = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注\n"

    corpus = []
    filter_list = [' ', '\n']
    hierarchy = ['\n']

    def __init__(self, lyric, depth=0, tokenizer='word'):
        '''Tokenize a string and store as bag of words.

        Keyword arguments:
        lyric -- A piece of lyric, list of sentence
        depth -- The hierarchy of the doc
        tokenizer -- the tokenizer(char/word)
        '''
        self.origin = lyric

        self.children = [Doc(lyric=line, depth=depth+1)\
                     for line in self.origin.split(Doc.hierarchy[depth])]\
                if depth < len(Doc.hierarchy) else []
        if self.children == []:
            if tokenizer == 'char':
                self.bag = list(self.origin)
            if tokenizer == 'word':
                self.bag = list(list(zip(*Doc.cut.cut(self.origin)))[0])
            self.bag = [word for word in self.bag if word not in Doc.filter_list]
        else:
            self.bag = []
            for child in self.children:
                self.bag += child.bag

        self.text = ' '.join(self.bag)


    def load_corpus(lyrics, tokenizer='word'):
        '''Load from lyrics read by data.read_filtered()

        tokenizer: char / word
        '''
        Doc.corpus = []
        for idx, lyric in enumerate(lyrics):
            new_doc = Doc(lyric=lyric, tokenizer=tokenizer)
            Doc.corpus.append(new_doc)

            if idx % (len(lyrics) / 10)== 0:
                print('Loading %.2f%%' % (100 * idx / len(lyrics)))

        print('Loading complete')
        Doc.filter_corpus()

    def filter_corpus():
        print('Filtering out frequent and rare words.')

        texts = [' '.join(document.bag) for document in Doc.corpus]
        vectorizer = CountVectorizer(max_df=0.05, min_df=0.0005)
        vectorizer.fit_transform(texts)

        for doc in Doc.corpus:
            doc.filter(vectorizer.vocabulary_)

        corpus_size = len(Doc.corpus)
        Doc.corpus = [doc for doc in Doc.corpus if len(doc.bag) > 20]
        print('Complete ')
        print('Vocab size %d' % len(vectorizer.vocabulary_))
        print('Filtered %.2f%% too short.' % (100 * (1 - len(Doc.corpus) / corpus_size)))

    def filter(self, vocab):
        self.bag = [word for word in self.bag\
                    if word in vocab]
        self.text = ' '.join(self.bag)

        for child in self.children:
            child.filter(vocab)

        self.children = [child for child in self.children if child.bag != []]


    def to_vec(self):
        '''Use pretrained model to get the vector[512] of bag'''
        vec = np.zeros([512], dtype='float')
        for word in self.bag:
            if word in Doc.model.wv:
                vec += Doc.model.wv[word]
            else:
                for char in word:
                    vec += Doc.model.wv[char] if char in Doc.model.wv else 0

        return vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec

    def similarity(self, doc2):
        '''Return the cosine distance between two lines.'''
        return np.dot(self.to_vec(), doc2.to_vec())

    def most_similar(self):
        '''Find the most similar line in the corpus.

        Similar defined as cosine distance.
        '''

        most_simi, winner = 0, None

        for i, candidate in enumerate(Doc.corpus):
            simi = Doc.similarity(self, candidate)
            if simi > most_simi and candidate.bag != self.bag:
                most_simi, winner = simi, candidate

        if winner is None:
            print('Warning: Nothing in corpus.')
            return ''
        else:
            return winner.origin

    def test():
        '''Unit test & usage'''
        line1 = '天青色等烟雨'
        doc1 = Doc(line1)
        print('Tokenized and word vec[:10] of %s:' % line1)
        print(doc1.bag)
        print(doc1.to_vec()[:10])
        print('')
        print('Most similar word to 河流')
        print(Doc.model.most_similar('河流'))
        print('')
        line2 = '而我在等你'
        doc2 = Doc(line2)
        print('Similarity between %s, %s' % (line1, line2))
        print(Doc.similarity(doc1, doc2))
        print('')
        print('Most similar to %s in corpus' % line1)
        print(doc1.most_similar())
        print('')
