# TODO: setup region selecting for attention
# TODO: setup additional positional encoding per region on encoder

from typing import Tuple
import psi4
from torch import nn
import rdkit
from torch_geometric.data import Data
from rdkit.Chem import Mole
from rdkit.Chem.rdmolops import GetShortestPath
from ..preprocessing.utils import mol_to_xyz

def get_topological(m: Mole, psi, start_ix: int, end_ix: int):
    path: Tuple[int] = GetShortestPath(m, start_ix, end_ix)

    # possibly computing length based on a combination of length and order
    total_bond_length = 0

    for i in range(len(path)-1):
        total_bond_length += psi.bond_length(i, i+1)
    
    return total_bond_length

def get_edge(m: Mole, data: Data):
    pass
    # we dont need this right? embedding takes in the nearest edges


class Topological(nn.Module):
    def __init__(self, m: Mole, psi, d_attn: int = 32):
        
