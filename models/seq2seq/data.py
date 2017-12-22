from utils import Doc, read
import logging
import random


def pad_seq(seq, max_length, pad_token):
    '''Pad a with the PAD symbol'''
    seq += [pad_token for i in range(max_length - len(seq))]
    return seq


class PairGenerator(object):

    def __init__(self):
        # Load the corpus
        Doc.load()

        symbols = ['<pad>', '<sos>', '<eos>']
        Doc.extend_vocab(symbols)
        vocab = Doc.get_vocab()

        # Get sentences
        assert len(vocab) == 10001 + len(symbols)
        list_of_list_of_lines = [doc.get_lines() for doc in Doc.get_corpus()]

        # Form pairs
        q = [first
             for line in list_of_list_of_lines for first in line[:-1]]

        a = [second
             for line in list_of_list_of_lines for second in line[1:]]

        assert len(q) == len(a)
        logging.info('%d pairs generated' % len(q))

        # shuffle them
        idxs = list(range(len(q)))
        random.shuffle(idxs)
        self.pairs = {
            'q': [q[idxs[i]] for i in range(len(q))],
            'a': [a[idxs[i]] for i in range(len(a))],
        }
        del q, a

    def view(self, idx):
        '''View the text of a pair'''
        return Doc.idxs_to_text(self.pairs['q'][idx]),\
            Doc.idxs_to_text(self.pairs['a'][idx])

    def get(self):
        return self.pairs
