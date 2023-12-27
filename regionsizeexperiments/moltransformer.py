# control version
import os
import configparser
import rdkit
from typing import Any
from collections import namedtuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from modules.attention.topological import Topological
from modules.preprocessing.preprocess import Preprocess
from modules.preprocessing.utils import smiles_to_graph
from pydantic import BaseModel


Datum = namedtuple("Datum", ['graph', 'mol', 'psi', 'smiles'])


class TopologicalAttention(nn.Module):
    def __init__(self, d_attn: int, d_model: int):
        super().__init__()
        self.d_attn, self.d_model = d_attn, d_model
        self.softmax = nn.Softmax(-1)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, top: Tensor):
        QK = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_attn ** 0.5)
        # need to dim this
        score = self.softmax(QK + top)

        return torch.matmul(score, V)

class MultiHeadTopologicalAttention(nn.Module):
    def __init__(self, n_head: int, d_attn: int, d_model: int):
        super().__init__()
        self.att = TopologicalAttention(d_attn, d_model)

        self.n_head = n_head
        self.d_attn = d_attn

        self.top = nn.Linear(16, 1)

        self.w_O = nn.Linear(n_head * d_attn, d_model)

        self.w_Q = nn.Linear(d_model, d_attn * n_head)
        self.w_K = nn.Linear(d_model, d_attn * n_head)
        self.w_V = nn.Linear(d_model, d_attn * n_head)

    def forward(self, input: Tensor, top: Tensor) -> Tensor:
        batch_size, seq_len, d_attn = input.size()

        top_out = self.top(top)
        top_out = top_out.squeeze().unsqueeze(1).repeat(1, self.n_head, 1, 1)

        Q, K, V = self.w_Q(input).reshape(batch_size, seq_len, self.n_head, self.d_attn).permute(0, 2, 1, 3), \
            self.w_K(input).reshape(batch_size, seq_len, self.n_head, self.d_attn).permute(0, 2, 1, 3), \
            self.w_V(input).reshape(batch_size, seq_len, self.n_head, self.d_attn).permute(0, 2, 1, 3)
        
        O = self.att(Q, K, V, top_out).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_head * self.d_attn) # (batch_size, n_head, seq_len, d_model)
        return self.w_O(O)

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

    def forward(self, input: Tensor, top: Tensor) -> Tensor:
        attn_out = self.attn(input, top)
        res1 = attn_out + input
        out = self.ln1(res1)
        out = self.ff2(self.ff(out))
        out = F.silu(out + res1)
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

        # self.bn = nn.BatchNorm2d(d_model) batchnorm for the topological encodings?

        self.encoders = nn.ModuleList([Encoder(d_attn, d_model, n_heads) for enc in range(n_encoders)])

    @staticmethod
    def from_config(file_path: str):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), file_path))
        
        d_model = int(config.get("transformer", "d_model"))
        n_encoders = int(config.get("transformer", "n_encoders"))
        n_heads = int(config.get("transformer", "n_heads"))
        d_attn = int(config.get("transformer", "d_attn"))
        d_embed = int(config.get("transformer", "d_embed"))
        use_pre = int(config.get("transformer", "use_pre"))

        return MolTransformer(d_model, n_encoders, n_heads, d_attn, d_embed, use_pre)
        


    def forward(self, graph: Tensor, top: Tensor) -> Tensor:
        
        init = self.initmod(graph)
        for encoder in self.encoders:
            out = encoder(init, top)
        return out
        
    