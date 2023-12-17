import pandas as pd
from torch_geometric.data import Data
from typing import Any, List, Dict
import torch
from collections import namedtuple
from torch import Tensor
from torch.utils.data import Dataset
from.utils import smiles_to_graph
from ..attention.topological import Topological


FEATURES = ["NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "SR-ARE", "SR-ATADS", "SR-HSE", "SR-MMP", "SR-p53"]


Mole = namedtuple("Mole", ['graph', 'top', 'features'])

class Tox21(Dataset):
    def __init__(self):
        super().__init__()

        df = pd.read_csv("./data/tox21.csv")
        self.data: List[Mole] = []

        for ix in len(df):
            graph, mol, psi = smiles_to_graph(df["smiles"][ix])
            top = torch.empty((graph.x.size()[1], graph.x.size()[1], 16))
            for atom1 in mol.GetAtoms():
                for atom2 in mol.GetAtoms():
                    x, y = atom1.GetIdx(), atom2.GetIdx()
                    total, vec1 = Topological().get_topological(mol, psi, x, y)
                    total, vec2 = Topological().get_topological(mol, psi, y, x)
                    top[x][y] = vec1
                    top[y][x] = vec2

            features: List[float] = []
            for feature in FEATURES:
                try: features.append(df[feature][ix])
                except: features.append(-1)
            self.data.append(Mole(graph, top, features))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Mole:
        return self.data[idx]
    

def tox21_collate(mols: List[Mole]):
    graph: Tensor = torch.stack([mol.graph.x for mol in mols], 0)
    top: Tensor = torch.stack([mol.top for mol in mols], 0)
    features: List[List[float]] = [mol.features for mol in mols]

    return graph, top, features