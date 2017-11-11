from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np
from gensim.models import Word2Vec

import torch
from torch.autograd import Variable
from torch import optim, nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()


# Loading data files
# ==================
# 
# The data for this project is a set of many thousands of English to
# French translation pairs.
# 
# The English to French pairs are too big to include in the repo, so
# download to ``data/eng-fra.txt`` before continuing. The file is a tab
# separated list of translation pairs:
# 
# ::
# 
#     I am cold.    Je suis froid.

# Similar to the character encoding used in the character-level RNN
# tutorials, we will be representing each word in a language as a one-hot
# vector, or giant vector of zeros except for a single one (at the index
# of the word). Compared to the dozens of characters that might exist in a
# language, there are many many more words, so the encoding vector is much
# larger. We will however cheat a bit and trim the data to only use a few
# thousand words per language.
# 

# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.
# 
# 
# 

# In[3]:


model = Word2Vec.load('../../seq2seq/saved/word2vec_model')


# In[4]:


SOS_token = 0
EOS_token = 1
PAD_token = 2

class Vocab:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "PAD"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
# 
# 
# 

# In[5]:


MAX_LENGTH = 40
def filterPair(p):
    return len(p[0]) < MAX_LENGTH and         len(p[1]) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# In[6]:


def readVocab():
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('../../seq2seq/saved/dataset.txt', encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[s for s in l.split('\t')] for l in lines]

    # make Vocab instances

    return Vocab(), pairs


# The full process for preparing the data is:
# 
# -  Read text file and split into lines, split lines into pairs
# -  Normalize text, filter by length and content
# -  Make word lists from sentences in pairs
# 
# 
# 

# In[7]:


def prepareData():
    vocab, pairs = readVocab()
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        vocab.addSentence(pair[0])
        vocab.addSentence(pair[1])
    print("Counted words:")
    print(vocab.n_words)
    return vocab, pairs


vocab, pairs = prepareData()
#print(random.choice(pairs))


# The Seq2Seq Model
# =================
# 
# A Recurrent Neural Network, or RNN, is a network that operates on a
# sequence and uses its own output as input for subsequent steps.
# 
# A `Sequence to Sequence network <http://arxiv.org/abs/1409.3215>`__, or
# seq2seq network, or `Encoder Decoder
# network <https://arxiv.org/pdf/1406.1078v3.pdf>`__, is a model
# consisting of two RNNs called the encoder and decoder. The encoder reads
# an input sequence and outputs a single vector, and the decoder reads
# that vector to produce an output sequence.
# 
# Unlike sequence prediction with a single RNN, where every input
# corresponds to an output, the seq2seq model frees us from sequence
# length and order, which makes it ideal for translation between two
# languages.
# 
# Consider the sentence "Je ne suis pas le chat noir" → "I am not the
# black cat". Most of the words in the input sentence have a direct
# translation in the output sentence, but are in slightly different
# orders, e.g. "chat noir" and "black cat". Because of the "ne/pas"
# construction there is also one more word in the input sentence. It would
# be difficult to produce a correct translation directly from the sequence
# of input words.
# 
# With a seq2seq model the encoder creates a single vector which, in the
# ideal case, encodes the "meaning" of the input sequence into a single
# vector — a single point in some N dimensional space of sentences.
# 
# 
# 

# In[8]:


BATCH_SIZE = 128
hidden_size = 512


# In[9]:


init_embeddings = np.zeros([vocab.n_words, hidden_size], dtype=np.float32)
for idx in vocab.index2word:
    if vocab.index2word[idx] in model.wv:
        init_embeddings[idx] = model.wv[vocab.index2word[idx]]
init_embeddings = torch.from_numpy(init_embeddings)


# The Encoder
# -----------
# 
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
# 
# ![](http://pytorch.org/tutorials/_images/encoder-network.png)
# 

# In[10]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        # self.embedding.weight = torch.nn.Parameter(init_embeddings)
        
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# ##### The Decoder
# -----------
# 
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
# 
# 
# 

# ##### Simple Decoder
# ^^^^^^^^^^^^^^
# 
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
# 
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
# 
# ![](http://pytorch.org/tutorials/_images/decoder-network.png)
# 
# 

# In[11]:


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #self.embedding.weight = torch.nn.Parameter(init_embeddings)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, placeholder):
        output = self.embedding(input).view(1, -1, self.hidden_size)
#       TODO: Why relu here?
#         output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, None

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# I encourage you to train and observe the results of this model, but to
# save space we'll be going straight for the gold and introducing the
# Attention Mechanism.
# 
# 
# 

# Attention Decoder
# ^^^^^^^^^^^^^^^^^
# 
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
# 
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
# 
# 
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
# 
# ![](https://i.imgur.com/1152PYf.png)
# 
# ![](http://pytorch.org/tutorials/_images/attention-decoder-network.png)

# In[12]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #self.embedding.weight = torch.nn.Parameter(init_embeddings)
        
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).transpose(0, 1)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


# <div class="alert alert-info"><h4>Note</h4><p>There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.</p></div>
# 
# Training
# ========
# 
# Preparing Training Data
# -----------------------
# 
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
# 
# 
# 

# In[13]:


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index[word] for word in sentence]


def variableFromSentence(vocab, sentence, reverse=False):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    indexes = [PAD_token] * (MAX_LENGTH - len(indexes)) + indexes
    if reverse:
        indexes = indexes[::-1]
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(pair):
    input_variable = variableFromSentence(vocab, pair[0])
    target_variable = variableFromSentence(vocab, pair[1], True)
    return (input_variable, target_variable)


# Training the Model
# ------------------
# 
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
# 
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability 
# 
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
# 
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
# 
# 
# 

# In[14]:


teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden(BATCH_SIZE)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = MAX_LENGTH
    target_length = MAX_LENGTH
    
    encoder_outputs = Variable(torch.zeros(BATCH_SIZE, max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    
    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[:, ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]] * BATCH_SIZE))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing


    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(k=1, dim=1)
            
            decoder_input = Variable(torch.LongTensor([[ni] for ni in topi[:, 0]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])

    loss.backward()

    debug_func()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
# 
# 
# 

# In[15]:


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# The whole training process looks like this:
# 
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
# 
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
# 
# 
# 

# In[16]:


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(BATCH_SIZE)]
        input_variable = torch.cat([pair[0] for pair in training_pairs], 1)
        target_variable = torch.cat([pair[1] for pair in training_pairs], 1)

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            evaluateRandomly(encoder, decoder)
            showPlot(plot_losses)


# Plotting results
# ----------------
# 
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
# 
# 
# 

# In[17]:


#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


# Evaluation
# ==========
# 
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
# 
# 
# 

# In[18]:


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH, beam_width=30):
    input_variable = variableFromSentence(vocab, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden(beam_width)

    input_variable = torch.cat([input_variable for i in range(beam_width)], 1)

    encoder_outputs = Variable(torch.zeros(beam_width, max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[:, ei] = encoder_output[0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]] * beam_width))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoder_hidden_copy = Variable(torch.zeros(encoder1.n_layers, beam_width, hidden_size))
    decoder_hidden_copy = decoder_hidden_copy.cuda() if use_cuda else decoder_hidden_copy
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    probability = torch.zeros(beam_width).view(-1, 1)
    probability = probability.cuda() if use_cuda else probability
    
    prev = [[]] * max_length
    idxs = [[]] * max_length
    
    cands = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)

#       LogSoftmax
        topv, topi = decoder_output.data.topk(beam_width)

        if di == 0:
            decoder_input = Variable(torch.LongTensor([[ni] for ni in topi[0]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            probability = topv[0].view(-1, 1)
        else:
#           Get beam_width candidate for each beam.
            topv = topv + probability
            topv, topi = topv.view(-1), topi.view(-1)

#           Select beam_width from beam_width*beam_width.
            _, topt = topv.topk(beam_width)

#           Update adn prepare for the next step.
            probability = topv[topt]
            decoder_input = Variable(torch.LongTensor([[topi[topt[k]]] for k in range(beam_width)]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
            prev[di] = [topt[k] // beam_width for k in range(beam_width)]
            
#           Don't forget to prepare the corresponding hidden state.
            for i in range(beam_width):
                decoder_hidden_copy[:, i] = decoder_hidden[:, prev[di][i]]
            decoder_hidden = decoder_hidden_copy.clone()
            
#           If some beam meets its end.

            for i in range(beam_width):
                if decoder_input.data[i, 0] == PAD_token and probability[i] > -np.inf:
                    cands.append((probability[i], di, i))
                    probability[i] = -np.inf

#       Record each step's input
        idxs[di] = decoder_input.data
        
    def full_sentence(start, di):
        decoded_words = []
        for loc in range(1, di+1)[::-1]:
            decoded_words = [vocab.index2word[idxs[loc][start][0]]] + decoded_words
            start = prev[loc][start]
        return ''.join(decoded_words).replace('PAD', '')[::-1]

    cands = sorted(cands)[::-1]

    cands = list(map(lambda x: full_sentence(x[2], x[1]), cands))
    
    answers = []
    for cand in cands:
        if cand not in answers:
            answers.append(cand)

    return answers, decoder_attentions[:di + 1]


# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
# 
# 
# 

# In[19]:


def evaluateRandomly(encoder, decoder, n=5, beam_width=30, show=5):
    for i in range(n):
        pair = random.choice(pairs)
        #print('>', pair[0])
        #print('=', pair[1])
        output_sentences, attentions = evaluate(encoder, decoder, pair[0], beam_width=beam_width)
        #for j in range(min(len(output_sentences), show)):
        #    print('<', output_sentences[j])
        #print()


# Training and Evaluating
# =======================
# 
# With all these helper functions in place (it looks like extra work, but
# it's easier to run multiple experiments easier) we can actually
# initialize a network and start training.
# 
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
# 
# .. Note:: 
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
# 
# 
# 

# In[20]:


def debug_func():
    pass
#     print(torch.sum(encoder1.embedding.weight.grad.data), torch.sum(encoder1.embedding.weight.grad.data[1]))
#     print(list(encoder1.gru.parameters())[0].grad )
#     print(torch.sum(encoder1.embedding.weight.grad.data), torch.sum(encoder1.embedding.weight.grad.data[1]))


# In[33]:


encoder1 = EncoderRNN(vocab.n_words, hidden_size, n_layers=2)
# decoder1 = DecoderRNN(hidden_size, vocab.n_words, n_layers=2)
attn_decoder1 = AttnDecoderRNN(hidden_size, vocab.n_words, n_layers=2, dropout_p=0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
#     decoder1 = decoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()

encoder1.load_state_dict(torch.load('../../seq2seq/saved/encoder.params', map_location=lambda storage, loc: storage))
attn_decoder1.load_state_dict(torch.load('../../seq2seq/saved/attn_decoder.params', map_location=lambda storage, loc: storage))
# trainIters(encoder1, attn_decoder1, 75000, print_every=10, plot_every=200, learning_rate=1e-3)
# torch.save(encoder1.state_dict(), 'encoder.params')
# torch.save(attn_decoder1.state_dict(), 'attn_decoder1.params')


# In[31]:


#evaluateRandomly(encoder1, attn_decoder1, beam_width=30)


# In[ ]:


def predict(sentence):
    answers, _ = evaluate(encoder1, attn_decoder1, sentence, max_length=MAX_LENGTH, beam_width=30)
    return answers

