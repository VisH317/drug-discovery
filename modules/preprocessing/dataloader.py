import pandas as pd
import os
import pickle
from torch_geometric.data import Data
from typing import Any, List, Dict
import torch
from collections import namedtuple
from torch import Tensor
from torch.utils.data import Dataset
from.utils import smiles_to_graph
from ..attention.topological import Topological
import pickle
from tqdm import tqdm
from random import randint


FEATURES = ["NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "SR-ARE", "SR-ATADS", "SR-HSE", "SR-MMP", "SR-p53"]


Mole = namedtuple("Mole", ['graph', 'top', 'features', 'active_features'])

def parse_data():
    df = pd.read_csv("./data/tox21.csv")
    data: List[Mole] = []

    for ix in tqdm(range(len(df)), total=len(df), desc="doing data stuff", ): 
        graph, mol, success = smiles_to_graph(df["smiles"][ix])
        if not success: continue
        top = torch.empty((graph.x.size()[0], graph.x.size()[0], 16))
        for atom1 in mol.GetAtoms():
            for atom2 in mol.GetAtoms():
                x, y = atom1.GetIdx(), atom2.GetIdx()
                vec1 = Topological().get_topological(mol, x, y)
                vec2 = Topological().get_topological(mol, y, x)
                top[x][y] = vec1
                top[y][x] = vec2

        features: List[float] = []
        active_features: list[float] = []
        for feature in FEATURES:
            try: 
                features.append(int(df[feature][ix]))
                active_features.append(ix)
            except: features.append(-1)
        data.append(Mole(graph, top, features))
    
    with open("./data/tox21_parsed.pkl", "wb") as pkl:
        pickle.dump(data, pkl)

class Tox21(Dataset):
    def __init__(self, file: str):
        super().__init__()
        self.file_path = file
        with open(os.path.join(os.getcwd(), file), "rb") as f:
            self.data = pickle.load(f)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Mole:
        return self.data[idx]
    
MAX_LEN = 300

def pad_graph_x(ten: Tensor) -> Tensor:
    l, dim = ten.size()
    if l >= MAX_LEN: return ten[:MAX_LEN]
    else:
        pad_ten = torch.zeros((MAX_LEN-l, dim))
        return torch.concat((ten, pad_ten), 0)

def pad_top(ten: Tensor) -> Tensor:
    l, l2, dim = ten.size()
    if l >= MAX_LEN: return ten[:MAX_LEN, :MAX_LEN]
    else:
        pad_1 = torch.zeros((MAX_LEN-l, l2, dim))
        first_pad = torch.concat((ten, pad_1), dim=0)
        pad_2 = torch.zeros((MAX_LEN, MAX_LEN - l2, dim))
        second_pad = torch.concat((first_pad, pad_2), dim=1)
        return second_pad

def tox21_collate(mols: List[Mole]):
    graph: Tensor = torch.stack([pad_graph_x(mol.graph.x.to(torch.float32)) for mol in mols], 0)
    top: Tensor = torch.stack([pad_top(mol.top.to(torch.float32)) for mol in mols], 0)
    feature_idx, feature = [], []
    for mol in mols:
        ix = mol.active_features[randint(0, len(mol.active_features)-1)]
        feature_idx.append(ix)
        feature.append(mol.features[ix])

    return graph, top, feature_idx, feature