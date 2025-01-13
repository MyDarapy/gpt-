import torch 
import torch.nn aas nn

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

