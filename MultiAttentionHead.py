import torch 
import torch.nn as nn 
import SimpleSelfAttentionHead 
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
