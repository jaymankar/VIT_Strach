import torch
import torch.nn as nn
from Config import config
from Model.Embeddings import Embeddings
from Model.LayerNorm import LayerNorm
from Model.MSA import Multi_Head_Attion
from Model.MLP import FeedForward
from Model.Transformer_Encoder_Block import Transformer_Encoder


class VisionTranformer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.Embeddings = Embeddings(config)
    # self.Transformer_Block = nn.Sequential(*[Transformer_Block(config) for _ in range(config["num_layer"])])
    self.Transformer_Block = Transformer_Encoder(config)
    self.Norm = LayerNorm(config)
    self.Final_Layer = nn.Linear(config["embedding_dim"], config["num_classes"])

  def forward(self,x):
    x = self.Embeddings(x)
    x = self.Transformer_Block(x)
    x = self.Norm(x)
    x = x[:,0,:]
    logits = self.Final_Layer(x)
    return logits
