from typing import List
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):

    def __init__(self, d_model: int, att_dim: int):
        super().__init__()

        self.d_model = d_model
        self.att_dim = att_dim

        self.query = nn.Linear(d_model, att_dim)
        self.key = nn.Linear(d_model, att_dim)
        self.value = nn.Linear(d_model, att_dim)

    def forward(self, input: Tensor) -> Tensor:
        Q: Tensor = self.query(input)
        K: Tensor = self.query(input)
        V: Tensor = self.query(input)

        QK = torch.matmul(Q, K.transpose(0, 1)) / torch.sqrt(torch.Tensor(self.att_dim))
        score = F.softmax(QK)

        return torch.matmul(score, V)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, att_dim: int, num_heads: int):
        super().__init__()

        self.d_model, self.att_dim, self.num_heads = d_model, att_dim, num_heads

        self.heads = nn.ModuleList([AttentionHead(d_model, att_dim) for _ in range(num_heads)])

        self.O = nn.Linear(att_dim * num_heads, d_model)

    def forward(self, input: Tensor) -> Tensor:
        Zs: List[Tensor] = []
        for head in self.heads:
            Zs.append(head(input))
        Zs_c = torch.cat(Zs, dim=1)
        return self.O(Zs_c)
    

class MultiQueryAttention(nn.Module):

    def __init__(self, d_model: int, att_dim: int, num_queries: int, use_att_dim: bool = False):
        super().__init__()

        self.Qs = nn.ModuleList([nn.Linear(d_model, att_dim) for _ in range(num_queries)])
        self.K = nn.Linear(d_model, att_dim)
        self.V = nn.Linear(d_model, att_dim)

        self.O = nn.Linear(num_queries * att_dim, d_model if not use_att_dim else att_dim)

    def forward(self, input: Tensor) -> Tensor:
        key: Tensor = self.K(input)
        value: Tensor = self.V(input)

        QK = torch.cat([torch.matmul(Q, key.transpose(0, 1)) / torch.sqrt(torch.Tensor(self.att_dim)) for Q in self.Qs], dim=1)

        return self.O(QK)
    

class GroupQueryAttention(nn.Module):

    def __init__(self, d_model: int, att_dim: int, num_queries: int, num_groups: int):
        super().__init__()

        self.mqa = nn.ModuleList([MultiQueryAttention(d_model, att_dim, num_queries, use_att_dim=True) for _ in range(num_groups)])
        self.O = nn.Linear(num_groups*att_dim, d_model)

    def forward(self, input: Tensor) -> Tensor:
        Vs = torch.cat([mqa(input) for mqa in self.mqa], 1)
        return self.O(Vs)



