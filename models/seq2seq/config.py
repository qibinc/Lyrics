# Configure input data
MIN_LENGTH = 2
MAX_LENGTH = 10
USE_CUDA = True
TEMPRETURE = 1

# Configure models
attn_model = 'dot' # align models: dot / general / concat. Only dot is vectorized.
hidden_size = 512
n_layers = 2
dropout = 0.1
batch_size = 64

# Configure training/optimization
learning_rate = 2e-4
decoder_learning_ratio = 5.0
# teacher_forcing_ratio = 0.5
n_steps = 10000
plot_every = 50
print_every = 100
evaluate_every = 500

