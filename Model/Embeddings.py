import torch
import torch.nn as nn
from Config import config



class Patch_Embeddings(nn.Module):
  def __init__(self,config):
    super().__init__()

    self.image_size = config["img_size"]
    self.patch_size = config["patch_size"]
    self.num_channle = config["num_channle"]
    self.embedding_dim = config["embedding_dim"]
    self.num_patches = (self.image_size // self.patch_size) ** 2

    self.patch = nn.Conv2d(self.num_channle, self.embedding_dim, kernel_size=self.patch_size, stride=self.patch_size )

  def forward(self,x):

    x = self.patch(x)
    x = x.flatten(2)
    x = x.transpose(1,2)
    return x




class Embeddings(nn.Module):
  def __init__(self,config):
    super().__init__()

    self.Patch_Embeddings = Patch_Embeddings(config)
    self.cls_token = nn.Parameter(torch.randn(1,1,config["embedding_dim"]))
    self.pos_embedding = nn.Parameter(torch.randn(1,1+config["num_patch"],config["embedding_dim"]))
    self.dropout = nn.Dropout(config["dropout"])

  def forward(self,x):

    x = self.Patch_Embeddings(x)
    batch_size,_,_ = x.size()
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embedding
    x = self.dropout(x)
    return x



