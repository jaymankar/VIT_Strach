import torch
import torch.nn as nn
from Config import config



class LayerNorm(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.eps = 1e-5
    self.dim_out = config["embedding_dim"]
    self.scale = nn.Parameter(torch.ones(config["embedding_dim"]))
    self.shift = nn.Parameter(torch.zeros(config["embedding_dim"]))

  def forward(self,x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return self.scale * norm_x + self.shift
