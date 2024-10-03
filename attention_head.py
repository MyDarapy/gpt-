import torch 
import torch.nn as nn 

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