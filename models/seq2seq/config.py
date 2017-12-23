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
learning_rate = 5e-4
decoder_learning_ratio = 5.0
# teacher_forcing_ratio = 0.5
n_epochs = 500
plot_every = 1
print_every = 5
evaluate_every = 10
