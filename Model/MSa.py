import torch
import torch.nn as nn
from Config import config



class Multi_Head_Attion(nn.Module):
     def __init__(self,config):
      super().__init__()

      self.dim_out = config["embedding_dim"]
      self.n_head = config["Att_head"]
      self.dim_head = self.dim_out // self.n_head
      self.W_query = nn.Linear(self.dim_out, self.dim_out, bias = config["qkv_bias"])
      self.W_key = nn.Linear(self.dim_out, self.dim_out, bias = config["qkv_bias"])
      self.W_values = nn.Linear(self.dim_out, self.dim_out, bias = config["qkv_bias"])
      self.dropout = nn.Dropout(config["dropout"])
      self.W_O = nn.Linear(self.dim_out, self.dim_out)

     def forward(self,x):
      batch_size, patch, d_in = x.shape
      query = self.W_query(x)
      key = self.W_key(x)
      values = self.W_values(x)

      query = query.view(batch_size, patch, self.n_head, self.dim_head)
      key = key.view(batch_size, patch, self.n_head, self.dim_head)
      values = values.view(batch_size, patch, self.n_head, self.dim_head)

      query = query.transpose(1,2)
      key = key.transpose(1,2)
      values = values.transpose(1,2)

      scores = torch.matmul(query, key.transpose(2,3))
      scores = scores / (self.dim_head ** 0.5)

      scores = torch.softmax(scores, dim=-1)
      scores = self.dropout(scores)

      out = torch.matmul(scores, values).transpose(1,2).contiguous().view(batch_size, patch, self.dim_out)
      out = self.W_O(out)
      return out



