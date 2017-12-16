###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################
# %%
import argparse

import numpy as np
import torch
from torch.autograd import Variable

import data
from models.beam_search import beam_search
from utils import Doc
Doc.load()

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='60',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--pre', type=str, default='pre.txt',
                    help='previous input for rnnlm')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

# %%
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

def greedy_decode(input, hidden):
    with open(args.outf, 'w') as outf:
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ' ')

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

def generate_step_fn(sequences, states, scores):
    new_sequences, new_states, new_scores = [], [], []
    for sequence, state, score in zip(sequences, states, scores):
        output, state = model(sequence[-1].view(-1, 1), state)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]

        new_token = Variable(torch.LongTensor([word_idx]).view(-1, 1))
        if args.cuda:
            new_token.data = new_token.data.cuda()
        new_sequences.append(torch.cat([sequence, new_token]))
        new_states.append(state)
        new_scores.append(score + np.log(word_weights[word_idx] / torch.sum(word_weights)))

    return new_sequences, new_states, new_scores

def beam_decode(num_steps=100, beam_size=1, branch_factor=1, steps_per_iteration=1):
    with open(args.pre, 'r') as pre:
        s = pre.read()
        d = Doc(s)
        d.filter()
        pre_tokens = []
        for line in d.get_lines():
            for idx in line:
                pre_tokens.append(Doc.idxs_to_text(idx))
            pre_tokens.append('\n')
        pre_tokens = pre_tokens[:-1]
        print(pre_tokens)

        pre_tokens = [corpus.dictionary.word2idx[token]\
                if token in corpus.dictionary.word2idx\
                else corpus.dictionary.word2idx['<unk>']\
                for token in pre_tokens]
        # print(pre_tokens)

    hidden = model.init_hidden(1)
    input = Variable(torch.LongTensor(pre_tokens).view(-1, 1), volatile=True)
    if args.cuda:
        input.data = input.data.cuda()

    with open(args.outf, 'w') as outf:
        # for i in range(len(input)- 1):
            # print(i)
            # _, hidden = model(input[i].view(-1, 1), hidden)
        selected = beam_search(input, hidden, generate_step_fn, num_steps,\
                beam_size, branch_factor, steps_per_iteration)
        sequence, state, score = selected
        # print(sequence.data)
        for word_idx in sequence.data:
            word = corpus.dictionary.idx2word[word_idx[0]]
            outf.write(word + ' ')
            print(word, end='')

# beam_decode(input, hidden)
