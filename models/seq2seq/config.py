# Configure input data
MIN_LENGTH = 2
MAX_LENGTH = 10
USE_CUDA = True

# Configure models
attn_model = 'concat' # dot / general / concat
hidden_size = 200
n_layers = 1
dropout = 0.1
batch_size = 32

# Configure training/optimization
learning_rate = 1e-3
decoder_learning_ratio = 5.0
# teacher_forcing_ratio = 0.5
n_epochs = 10000
plot_every = 50
print_every = 100
evaluate_every = 500
