#Script for extracting and cleaning a subset of bookcorpus dataset
!pip install datasets
import string
import re
from datasets import load_dataset



load data from hugging face 
clean it 
save it to a file path 
dataset = load_dataset("bookcorpus", split="train",trust_remote_code=True)


def load_data(name: str = "bookcorpus"):
  load_dataset(name, split="train", trust_remote_code=True)



def clean_data(sentence):
  sentence = sentence.translate(str.maketrans('', '', string.punctuation))
  sentence = ''.join(char for char in sentence if not char.isdigit())
  return sentence

def process_dataset(dataset):
    return [clean_data(sentence['text']) for sentence in dataset]

dataset = dataset.shuffle(seed=65).select(range(10_000_000))
cl_data = process_dataset(dataset)

file_path = '/content/drive/MyDrive/bookcorpus.txt'
def write_list_to_file(file_path, cl_data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in cl_data:
            f.write(f"{item}\n")  # Write each item on a new line

text = write_list_to_file(file_path, cl_data)

!pip install wandb
import wandb

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from google.colab import drive
drive.mount('/content/drive')

# Model Hyperparameters
context_length = 1024
pad_index = 0
EOS_token = 1
embed_dim = 768  # also called model dimension d_model (512 in the GPT 1 paper)
num_of_heads = 12 # Attention heads
dropout = 0.2  #regularization
batch_size = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_layers = 12 #Transformer block
eval_iters = 400
evaluation_intervals = 400
#learning_rate = 2.5e-2
vocab_size = 15_000
T_max = 200

#Read in the file path
file_path = '/content/drive/MyDrive/bookcorpus.txt'

# read it in to inspect it
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", .format(len(text)))

type(text)

# let's look at the first 1000 words
print(text[:1000])

from tokenizers import ByteLevelBPETokenizer
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer on your dataset
tokenizer.train(files =['/content/drive/MyDrive/data.txt'], vocab_size=15_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Save the trained tokenizer
tokenizer.save_model('/content/drive/MyDrive/BPEtokenizer')

#Get the vocabulary created by the tokenizer
import json
vocab_file_path = '/content/drive/MyDrive/BPEtokenizer/vocab.json' #get all the strings in the dataset to be integers that can be used as for the lookup. reason for tokenization
# Open and read the Vocab JSON file
with open(vocab_file_path, 'r') as file:
    vocab = json.load(file)

print(len(vocab))

vocab_size = len(vocab)

tokens = tokenizer.encode(text[:100]).tokens #convert the strings into tokens
  print(tokens[:100])

def tokenize_map_ids(example, tokenizer, vocab):
  tokens = tokenizer.encode(example).tokens #convert the strings into tokens
  token_ids = [vocab.get(token) for token in tokens] #convert the tokens into ids
  token_ids = torch.tensor(token_ids, dtype=torch.long)
  return token_ids

data = tokenize_map_ids(text[:5_000_000], tokenizer, vocab)

data = torch.tensor(data, dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

def split_data(data, train_data_size):
  n = int(0.8*len(data))
  train_dataset = data[:n]
  test_dataset = data[n:]
  return train_dataset, test_dataset

train_dataset, test_dataset = split_data(data, 0.8)

def load_dataset(split):
  data = train_dataset if split == 'train' else test_dataset
  idx = torch.randint((len(data)-context_length), (batch_size,)) #idx has size of {batch_size}
  x = torch.stack([data[i:i+context_length] for i in idx])
  y = torch.stack([data[i+1:i+context_length+1] for i in idx]) #The target this includes the next token we are trying to predict
  x,y = x.to(device), y.to(device)
  return x, y

xb,yb = load_dataset('train')
print(xb,yb)

print(xb.shape, yb.shape)

#Write a dataclass to do the above

import math

class GPTLearningRateScheduler:
    def __init__(self, max_lr: float, min_lr: float, warm_up_iters: int, max_iters: int, start_lr: float = 0.0):
        self.start_lr: float = start_lr
        self.min_lr: float = min_lr
        self.max_lr: float = max_lr
        self.warm_up_iters: int = warm_up_iters
        self.max_iters: int = max_iters

    def linear_warm_up(self, current_iter: int):
        return self.max_lr * (current_iter / self.warm_up_iters)

    def cosine_annealing(self, current_iter: int):
        decay_ratio = (current_iter - self.warm_up_iters) / (self.max_iters - self.warm_up_iters)
        assert 0 <= decay_ratio <= 1, f"decay ratio is not between 0 and 1"
        coeff = 1 + math.cos(math.pi * decay_ratio)
        return self.min_lr + (0.5 * coeff  * (self.max_lr - self.min_lr))

    def get_lr(self, current_iter: int):
        if current_iter <= self.warm_up_iters:
            lr = self.linear_warm_up(current_iter)
            return lr

        return self.cosine_annealing(current_iter)

#Implement a simple attention head
class SimpleSelfAttentionHead(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(embed_dim, head_size, bias=False)
    self.query = nn.Linear(embed_dim, head_size, bias=False)
    self.value = nn.Linear(embed_dim, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
    self.head_size = head_size
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape #Batch, Timesteps, Channels
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)
    d_k = k.size(-1)

    # calculate the attention scores
    attention_scores = torch.matmul(q, k.transpose(-2,-1))  #Dot product attention scores (B T C) @ (B, C, T) --> (B T T)

    #scale down the scores so as to have gradient trainable parameters
    scaled_scores = attention_scores / self.head_size**0.5
    #mask out the future tokens
    scaled_scores = scaled_scores.masked_fill(self.tril[:T, :T]==0, float('-inf'))

    #Pass the scores/logits through softmax to normalize the sceores to attention weights
    attention_weights = F.softmax(scaled_scores, dim=-1)
    attention_weights = self.dropout(attention_weights)

    #Find the weighted sum
    output = attention_weights @ v
    return output 

#Implement multiple attention head to work in parallel 8 heads per block in our case
class MultiHeadAttention(nn.Module):
  def __init__(self, num_of_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([SimpleSelfAttentionHead(head_size) for _ in range(num_of_heads)])
    self.projection = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.projection(out))
    return out

class Feedforward(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()
    self.linear1 = nn.Linear(embed_dim, 4*embed_dim) #(512, 2048)
    self.linear2 = nn.Linear(4*embed_dim, embed_dim) #(2048, 512)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = self.linear1(x)
    out = self.gelu(out)
    out = self.linear2(out)
    out = self.dropout(x)
    return out

class Block(nn.Module):
  def __init__(self, embed_dim, num_of_heads):
    super().__init__()
    head_size = embed_dim // num_of_heads
    self.self_attention = MultiHeadAttention(num_of_heads, head_size)
    self.feedforward = Feedforward(embed_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    x = x + self.self_attention(self.ln1(x))
    x = x + self.feedforward(self.ln2(x))
    return x

class GPTLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding = nn.Embedding(len(vocab), embed_dim)
    self.position_embedding = nn.Embedding(context_length, embed_dim)
    self.transformer_block = nn.Sequential(*[Block(embed_dim=embed_dim, num_of_heads=num_of_heads) for _ in range(n_layers)])
    self.dropout = nn.Dropout(dropout)
    self.ln = nn.LayerNorm(embed_dim)
    self.lm_head = nn.Linear(embed_dim, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    token_embed = self.token_embedding(idx)
    pos_embed = self.position_embedding(torch.arange(T, device=device))
    x = token_embed + pos_embed
    x = self.dropout(x)
    x = self.transformer_block(x)
    x = self.ln(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate_token(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, context_length:]
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim =-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Model evaluation code
@torch.no_grad()
def calculate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = load_dataset(split)
      logits, loss = model(X,Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print(f"=> Saving checkpoint at {iter}")
    torch.save(state, filename)

model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

max_iter = int(len(train_dataset) / batch_size)

# Used the parameters as stated in GPT-1 paper
lr_scheduler = GPTLearningRateScheduler(max_lr=2.4e-4, min_lr=0.0, warm_up_iters=2000, max_iters=max_iter)
learning_rate = lr_scheduler.get_lr(0)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
wandb.init(project='tinyGPT')


for iter in range(max_iter):
  if iter % evaluation_intervals == 0 or iter == max_iter - 1:
    losses = calculate_loss()
    checkpoint = {
        'iter': iter,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': losses}
    save_checkpoint(checkpoint, filename=f"checkpoint_epoch_{iter}.pth.tar")
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, Learning Rate: {lr_scheduler.get_lr(current_iter=iter)}")
    wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "learning_rate": lr_scheduler.get_lr(current_iter=iter)})

  # Sample a batch of data
  x,y = load_dataset(split='train')

  #evaluate the loss, calculate gradient, update weight
  logits, loss = model(x, y)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()


  for params in optimizer.param_groups:
    params["lr"] = lr_scheduler.get_lr(current_iter=iter)

#Get some generations from our TinyGPT Language Model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(tokenizer.decode(model.generate_token(context, max_new_tokens=1000)[0].tolist()))

