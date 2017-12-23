import random
import numpy as np
import torch
from torch.autograd import Variable
from utils import Doc, read
from models.seq2seq.config import USE_CUDA


class SimplePairGenerator(object):

    def __init__(self):
        # Load the corpus
        Doc.load()

        symbols = ['<PAD>', '<SOS>', '<EOS>']
        Doc.extend_vocab(symbols)
        vocab = Doc.get_vocab()
        Doc.PAD_token = vocab['<PAD>']
        Doc.SOS_token = vocab['<SOS>']
        Doc.EOS_token = vocab['<EOS>']
        self.pad_token = vocab['<PAD>']

        # Get sentences
        assert len(vocab) == 10001 + len(symbols)
        list_of_list_of_lines = [doc.get_lines() for doc in Doc.get_corpus()]

        # Form pairs
        q = [first
             for line in list_of_list_of_lines for first in line[:-1]]

        a = [second
             for line in list_of_list_of_lines for second in line[1:]]

        assert len(q) == len(a)
        print('%d pairs generated' % len(q))

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

    def trim(self, min_length=1, max_length=10):
        q = self.pairs['q']
        a = self.pairs['a']
        unfiltered = len(q)
        self.pairs['q'] = []
        self.pairs['a'] = []
        for i in range(len(q)):
            if min_length <= len(q[i]) and len(q[i]) < max_length and\
               min_length <= len(a[i]) and len(a[i]) < max_length:
                self.pairs['q'].append(q[i])
                self.pairs['a'].append(a[i])
        print('Filtered %f' % (1 - len(self.pairs['q']) / unfiltered))

    def get(self):
        return self.pairs

    def pad_seq(self, seq, max_length):
        '''Pad a with the PAD symbol'''
        return seq + [self.pad_token] * (max_length - len(seq))

    def random_batch(self, batch_size):
        input_seqs = []
        target_seqs = []

        # Choose random pairs
        for i in range(batch_size):
            pair = random.choice(range(len(self.pairs['q'])))
            input_seqs.append(self.pairs['q'][pair].astype(int).tolist() + [Doc.EOS_token])
            target_seqs.append(self.pairs['a'][pair].astype(int).tolist() + [Doc.EOS_token])

        # Zip into pairs, sort by length (descending), unzip
        seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
        input_seqs, target_seqs = zip(*seq_pairs)

        # For input and target sequences, get array of lengths and pad with 0s to max length
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [self.pad_seq(s, max(input_lengths))
                        for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [self.pad_seq(s, max(target_lengths))
                         for s in target_seqs]

        # Turn padded arrays into (batch_size x max_len) tensors,
        # transpose into (max_len x batch_size)
        input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

        if USE_CUDA:
            input_var = input_var.cuda()
            target_var = target_var.cuda()

        return input_var, input_lengths, target_var, target_lengths
