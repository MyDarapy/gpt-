data = '/content/drive/MyDrive/bookcorpus.txt'
vocab_file_path = '/content/drive/MyDrive/BPEtokenizer/vocab.json'
data_files = 
saving_filepath = 

context_length = 2048
embed_dim = 576
query_heads = 9
kv_matrix_heads = 3
dropout_probability = 0.1
hidden_size = 1536
batch_size = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_blocks = 8
eval_iters = 200
evaluation_intervals = 200
vocab_size = 15_000

