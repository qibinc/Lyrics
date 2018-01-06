import math
import time
import random
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from utils import Doc
from models.seq2seq.masked_cross_entropy import masked_cross_entropy

# Choose input pairs
from models.seq2seq.data import SimplePairGenerator as PairGenerator
from models.seq2seq.config import USE_CUDA, MAX_LENGTH, MIN_LENGTH,TEMPRETURE
# Choose models
from models.seq2seq.model import EncoderRNN as Encoder
from models.seq2seq.model import LuongAttnDecoderRNN as Decoder
# Choose model sizes
from models.seq2seq.config import attn_model, hidden_size, n_layers, dropout, batch_size
# Choose training parameters
from models.seq2seq.config import learning_rate, decoder_learning_ratio, n_steps, plot_every, print_every, evaluate_every

from tensorboardX import SummaryWriter

writer = SummaryWriter()

pg = PairGenerator()
pg.trim(MIN_LENGTH, MAX_LENGTH)
pg.separate()
random_batch = pg.random_batch
n_words = len(Doc.get_vocab())

# # Training
#
# ## Defining a training iteration
#
# To train we first run the input sentence through the encoder word by word, and keep track of every output and the latest hidden state. Next the decoder is given the last hidden state of the decoder as its first hidden state, and the `<SOS>` token as its first input. From there we iterate to predict a next token from the decoder.
#
# ### Teacher Forcing vs. Scheduled Sampling
#
# "Teacher Forcing", or maximum likelihood sampling, means using the real target outputs as each next input when training. The alternative is using the decoder's own guess as the next input. Using teacher forcing may cause the network to converge faster, but [when the trained network is exploited, it may exhibit instability](http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf).
#
# You can observe outputs of teacher-forced networks that read with coherent grammar but wander far from the correct translation - you could think of it as having learned how to listen to the teacher's instructions, without learning how to venture out on its own.
#
# The solution to the teacher-forcing "problem" is known as [Scheduled Sampling](https://arxiv.org/abs/1506.03099), which simply alternates between using the target values and predicted values when training. We will randomly choose to use teacher forcing with an if statement while training - sometimes we'll feed use real target as the input (ignoring the decoder's output), sometimes we'll use the decoder's output.

# In[22]:


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([Doc.SOS_token] * batch_size))
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]




# Keep track of time elapsed and running averages
start = time.time()
plot_losses = []
print_loss_total = 0 # Reset every print_every
plot_loss_total = 0 # Reset every plot_every

# Plus helper functions to print time elapsed and estimated time remaining, given the current time and progress.

# In[24]:


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


# # Evaluating the network
#
# Evaluation is mostly the same as training, but there are no targets. Instead we always feed the decoder's predictions back to itself. Every time it predicts a word, we add it to the output string. If it predicts the EOS token we stop there. We also store the decoder's attention outputs for each step to display later.

# In[25]:

def evaluate(input_seq, max_length=MAX_LENGTH):
    input_seqs = [Doc.text_to_idxs(input_seq)[0].astype(int).tolist() + [Doc.EOS_token]]
    input_lengths = [len(input_seq[0])]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if USE_CUDA:
        input_batches = input_batches.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([Doc.SOS_token]), volatile=True) # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    decoded_words = []
    
    #Run through beam search decoder
    #print('Choose word by beam search')
    from models.beam_search import beam_search
    def generate_step_fn(sequences, states, scores):
        new_sequences, new_states, new_scores = [], [], []
        for sequence, state, score in zip(sequences, states, scores):
            #for attn model, we add encoder_outputs
            output, state = decoder(sequence[-1].view(-1, 1), state, encoder_outputs)
            word_weights = output.squeeze().data.div(TEMPRETURE).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]

            new_token = Variable(torch.LongTensor([word_idx]).view(-1, 1))
            if USE_CUDA:
                new_token.data = new_token.data.cuda()
            new_sequences.append(torch.cat([sequence, new_token]))
            new_states.append(state)
            new_scores.append(score + np.log(word_weights[word_idx] / torch.sum(word_weights)))
        return new_sequences, new_states, new_scores

    selected = beam_search(initial_sequence=decoder_input, initial_state=decoder_hidden,\
                           generate_step_fn = generate_step_fn, num_steps = 100,\
                           beam_size=5, branch_factor=1, steps_per_iteration=1)

    sequence, state, score = selected
    # print(sequence.data)
    for word_idx in sequence.data:
        if word_idx[0] == Doc.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(Doc.idxs_to_text(word_idx))
       
    '''
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        # Choose word by sampling
        print('Choose word by sampling with tempreture ',TEMPRETURE)
        word_weights = decoder_output.data.div(TEMPRETURE).exp().cpu()
        ni = torch.multinomial(word_weights, 1)[0][0]
        # Choose top word from output
        #print('Choose top word')
        #topv, topi = decoder_output.data.topk(1)
        #ni = topi[0][0]
        if ni == Doc.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(Doc.idxs_to_text(ni))

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    '''
    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return decoded_words


def evaluate_and_show(input_sentence, target_sentence=None):
    output_words = evaluate(input_sentence)
    output_sentence = ' '.join(output_words)
    print('>', input_sentence)
    if target_sentence is not None:
        print('=', target_sentence)
    print('<', output_sentence)

def evaluate_lines(num=5):
    print('Training sets:')
    idx = random.choice(range(len(pg.train_pairs['q'])))
    input_sentence = Doc.idxs_to_text(pg.train_pairs['q'][idx])
    print(input_sentence)
    for i in range(num):
        output_words = evaluate(input_sentence)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        input_sentence = output_sentence


# We can evaluate random sentences from the training set and print out the input, target, and output to make some subjective quality judgements:
def evaluate_randomly():
    print('Training sets:')
    idx = random.choice(range(len(pg.train_pairs['q'])))
    input_sentence = Doc.idxs_to_text(pg.train_pairs['q'][idx])
    target_sentence = Doc.idxs_to_text(pg.train_pairs['a'][idx])
    evaluate_and_show(input_sentence, target_sentence)
    
    print('Test sets:')
    idx = random.choice(range(len(pg.test_pairs['q'])))
    input_sentence = Doc.idxs_to_text(pg.test_pairs['q'][idx])
    target_sentence = Doc.idxs_to_text(pg.test_pairs['a'][idx])
    evaluate_and_show(input_sentence, target_sentence)


# ## Running training
#
# With everything in place we can actually initialize a network and start training.
#
# To start, we initialize models, optimizers, a loss function (criterion), and set up variables for plotting and tracking progress:

# In[23]:


# Initialize models
encoder = Encoder(n_words, hidden_size, n_layers, dropout=dropout)
decoder = Decoder(attn_model, hidden_size, n_words, n_layers, dropout=dropout)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    decoder.cuda()

# # Putting it all together
#
# To actually train, we call the train function many times, printing a summary as we go.
#
# *Note:* If you're running this notebook you can **train, interrupt, evaluate, and come back to continue training**. Simply run the notebook starting from the following cell (running from the previous cell will reset the models).


# Begin!


visualize_num = 1000
visualize_list = random.sample(range(n_words), visualize_num)
step = 0
while step < n_steps:
    step += 1

    # Get training data for this cycle
    input_batches, input_lengths, target_batches, target_lengths = random_batch(batch_size,'train')

    # Run the train function
    loss = train(
        input_batches, input_lengths, target_batches, target_lengths,
        encoder, decoder,
        encoder_optimizer, decoder_optimizer, criterion
    )

    # Keep track of loss
    print_loss_total += loss
    plot_loss_total += loss

    if step % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_since(start, step / n_steps), step, step / n_steps * 100, print_loss_avg)
        print(print_summary)

    if step % evaluate_every == 0:
        evaluate_randomly()
        writer.add_embedding(list(encoder.embedding.parameters())[0].data[visualize_list],
                             Doc.idxs_to_text([[idx] for idx in visualize_list]),
                             global_step=step, tag='Word Embeddings')

    if step % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
        writer.add_scalar('loss', plot_loss_avg, step)

import datetime
now = datetime.datetime.now()
now_time = now.strftime('%Y-%m-%d %H:%M:%S')

torch.save(encoder, './enc_params'+now_time+'.pkl')
#encoder = torch.load('./enc_params.pkl'))

torch.save(decoder, './dec_params'+now_time+'.pkl')
#decoder = torch.load('./dec_params.pkl'))
print('Model saved')
        
writer.close()

