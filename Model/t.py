import torch
import torch.nn as nn
from Config import config
from Model.Embeddings import Embeddings
from Model.LayerNorm import LayerNorm
from Model.MSa import Multi_Head_Attion
from Model.MLP import FeedForward


class Transformer_Block(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.layer_norm1 = LayerNorm(config)
    self.layer_norm2 = LayerNorm(config)
    self.Multi_Head_Attion = Multi_Head_Attion(config)
    self.MLP = FeedForward(config)

  def forward(self,x):
    shortcut = x
    x = self.layer_norm1(x)
    x = self.Multi_Head_Attion(x)
    x = shortcut + x
    shortcut = x

    x = self.layer_norm2(x)
    x = self.MLP(x)
    x = shortcut + x

    return x


class Transformer_Encoder(nn.Module):
  def __init__(self,config):
    super().__init__()
    self.block = nn.ModuleList([Transformer_Block(config) for _ in range(config["num_layer"])])

    # self.block = nn.ModuleList([])
    # for _ in range(config["num_layer"]):
    #   self.block.append(Transformer_Block(config))

  def forward(self,x):
    for block in self.block:
      x = block(x)

    return x


