import torch 
import torch.nn as nn 
from attention_head import SimpleSelfAttentionHead

#Implement multiple attention head to work in parallel 12 heads per block in our case
class MultiHeadAttention(nn.Module):
  def __init__(self, num_of_heads, embed_dim, head_size, dropout_probability):
    super().__init__()
    assert embed_dim % num_of_heads == 0
    head_size = int(embed_dim // num_of_heads)
    self.heads = nn.ModuleList([SimpleSelfAttentionHead(head_size) for _ in range(num_of_heads)])
    self.projection = nn.Linear(embed_dim, embed_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.projection(out))
    return out
