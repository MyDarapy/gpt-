import torch 
import torch.nn as nn
from feedforward import Feedforward
from multihead import MultiHeadAttention

class Block(nn.Module):
  def __init__(self, embed_dim, num_of_heads):
    super().__init__()
    assert embed_dim % num_of_heads == 0
    head_size = int(embed_dim // num_of_heads)
    self.self_attention = MultiHeadAttention(num_of_heads, head_size)
    self.feedforward = Feedforward(embed_dim)
    self.ln1 = nn.LayerNorm(embed_dim)
    self.ln2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    x = x + self.self_attention(self.ln1(x))
    x = x + self.feedforward(self.ln2(x))
    return x


class GPTLanguageModel(nn.Module):
  def __init__(self, embed_dim, context_length, num_of_heads, vocab_size, n_layers, dropout):
    super().__init__()
    self.token_embedding = nn.Embedding(vocab_size, embed_dim)
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
