from models.seq2seq import SimplePairGenerator as PairGenerator
from models.seq2seq import EncoderRNN as Encoder
from models.seq2seq import LuongAttnDecoderRNN as Decoder

# Configure input data
MIN_LENGTH = 2
MAX_LENGTH = 10
USE_CUDA = True

# Configure models
attn_model = 'dot' # dot / general / concat
hidden_size = 200
n_layers = 2
dropout = 0.1
batch_size = 64

# Configure training/optimization
learning_rate = 1e-3
decoder_learning_ratio = 5.0
# teacher_forcing_ratio = 0.5
n_epochs = 500
plot_every = 20
print_every = 1
evaluate_every = 10
