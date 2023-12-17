# TODO: setup region selecting for attention
# TODO: setup additional positional encoding per region on encoder

from typing import Tuple
import psi4
import torch
from torch import nn, Tensor
import rdkit
from torch_geometric.data import Data
from rdkit.Chem.rdmolops import GetShortestPath
from ..preprocessing.utils import mol_to_xyz

def get_topological(m, psi, start_ix: int, end_ix: int):
    path: Tuple[int] = GetShortestPath(m, start_ix, end_ix)

    # possibly computing length based on a combination of length and order
    total_bond_length = 0

    for i in range(len(path)-1):
        total_bond_length += psi.bond_length(i, i+1)
    
    return total_bond_length

def get_edge(m, data: Data):
    pass
    # we dont need this right? embedding takes in the nearest edges


class Topological():
    def __init__(self, d_attn: int = 32):
        self.d_attn = d_attn

    def get_topological(self, m, psi, start_ix: int, end_ix: int) -> float:
        path: Tuple[int] = GetShortestPath(m, start_ix, end_ix)

        # possibly computing length based on a combination of length and order
        total_bond_length = 0
        bond_lengths = []

        for i in range(len(path)-1):
            le = psi.bond_length(i, i+1)
            bond_lengths.append(le)
            total_bond_length += le

        fin_arr = [total_bond_length] # TODO: try switching this to coords instead

        length = max(len(bond_lengths), 15)

        for i in range(length): fin_arr.append(bond_lengths[i])
        for i in range(15-length): fin_arr.append(0)
        
        return total_bond_length, Tensor(fin_arr, dtype=torch.float16)
