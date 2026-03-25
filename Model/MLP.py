import torch
import torch.nn as nn
from Config import config



class FeedForward(nn.Module):
  def __init__(self,config):
    super().__init__()

    self.layer = nn.Sequential(
        nn.Linear(config["embedding_dim"], 4 * config["embedding_dim"]),
        nn.GELU(),
        nn.Linear(4 * config["embedding_dim"], config["embedding_dim"])
    )
    self.dropout = nn.Dropout(config["dropout"])

  def forward(self,x):
    x = self.layer(x)
    x = self.dropout(x)
    return x

