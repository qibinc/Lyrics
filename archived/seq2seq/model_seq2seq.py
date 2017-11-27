from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim, nn
import torch.nn.functional as F
# from gensim.models import Word2Vec


hidden_size = 512
teacher_forcing_ratio = 0.5
MIN_LENGTH = 0
MAX_LENGTH = 40
BATCH_SIZE = 32
SOS_token = 0
EOS_token = 1
PAD_token = 2
#use_cuda = False
use_cuda = torch.cuda.is_available()
print('use_cuda ',use_cuda)

def debug_func():
    pass
#     print(torch.sum(encoder1.embedding.weight.grad.data), torch.sum(encoder1.embedding.weight.grad.data[1]))
#     print(list(encoder1.gru.parameters())[0].grad )
#     print(torch.sum(encoder1.embedding.weight.grad.data), torch.sum(encoder1.embedding.weight.grad.data[1]))        
        
     

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        print(use_cuda)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return hidden, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
        
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)

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
        
        
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, -1, self.hidden_size)
#         embedded = self.dropout(embedded)

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

def evaluate(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
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

    #loss.backward()

    debug_func()
    
    #encoder_optimizer.step()
    #decoder_optimizer.step()

    return loss.data[0] / target_length
        
        

        
  
        
        
        
        
        
        
        
        
        
        
        
        
        
        