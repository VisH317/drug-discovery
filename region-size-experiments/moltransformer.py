# control version
import rdkit
from typing import Any
from collections import namedtuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.attention.base import MultiHeadAttention
from modules.attention.topological import Topological
from modules.preprocessing.preprocess import Preprocess
from modules.preprocessing.utils import smiles_to_graph
from pydantic import BaseModel


Datum = namedtuple("Datum", ['graph', 'mol', 'psi', 'smiles'])


class TopologicalAttention(nn.Module):
    def __init__(self, d_attn: int, d_model: int):
        super().__init__()

        self.topological = Topological(d_attn)
        self.top = nn.Linear(16, 1)

        self.Q = nn.Linear(d_model, d_attn)
        self.K = nn.Linear(d_model, d_attn)
        self.V = nn.Linear(d_model, d_attn)

    def forward(self, input: Tensor, top: Tensor):
        # dim: batch, seq_len, d_model
        Q: Tensor = self.Q(input)
        K: Tensor = self.K(input)
        V: Tensor = self.V(input)

        total, top = self.topological.get_topological()

        QK = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.Tensor(self.att_dim))
        
        score = F.softmax(QK + self.top(top))

        return torch.matmul(score, V)

class MultiHeadTopologicalAttention(nn.Module):
    def __init__(self, n_head: int, d_attn: int, d_model: int):
        super().__init__()
        self.heads = nn.ModuleList([TopologicalAttention(d_attn, d_model) for head in range(n_head)])
        self.O = nn.Linear(n_head * d_attn, d_model)

    def forward(self, input: Tensor) -> Tensor:
        outputs = [head(input) for head in self.heads]
        concated = torch.concat(outputs, dim=-1)
        return self.O(concated)

class Encoder(nn.Module):
    def __init__(self, d_attn: int = 16, d_model: int = 32, n_head: int = 8):
        super().__init__()

        # add config
        self.d_attn = d_attn
        self.d_model = d_model
        self.n_head = n_head
        self.ff_dim = d_model * 4

        self.attn = MultiHeadTopologicalAttention(n_head, d_attn, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, input: Tensor, m, psi) -> Tensor:
        attn_out = self.attn(input, m, psi)
        res1 = attn_out + input
        out = self.ln1(res1)
        out = self.ff2(self.ff(out))
        out = out + res1
        return self.ln2(out)
    

class MolTransformer(nn.Module):
    def __init__(self, d_model: int, n_encoders: int, n_heads: int, d_attn: int, d_embed: int, use_pre: bool = False) -> None:
        super().__init__()

        self.use_pre = use_pre
        if use_pre: self.pre = Preprocess()

        self.initmod = nn.Sequential(
            nn.Linear(d_embed, d_model),
            nn.LayerNorm(d_model)
        )

        self.encoders = nn.ModuleList([Encoder(d_attn, d_model, n_heads) for enc in range(n_encoders)])

    def forward(self, input: Datum) -> Tensor:
        init = self.initmod(input.graph)
        for encoder in self.encoders:
            out = encoder(init, input.mol, input.psi)
        
        return out
        
    
