import pandas as pd
from torch_geometric.data import Data
from typing import Any, List
from pydantic import BaseModel
import torch
from torch.utils.data import Dataset
from.utils import smiles_to_graph
from ..attention.topological import Topological


FEATURES = ["NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "SR-ARE", "SR-ATADS", "SR-HSE", "SR-MMP", "SR-p53"]

class Mole(BaseModel):
    graph: Data
    mol: Any
    psi: Any
    top: Tensor
    features: List[float]

class Tox21(Dataset):
    def __init__(self):
        super().__init__()

        df = pd.read_csv("./data/tox21.csv")
        self.data: List[Mole] = []

        for ix in len(df):
            graph, mol, psi = smiles_to_graph(df["smiles"][ix])
            top = torch.empty((graph.x.size()[1], graph.x.size()[1], 16))
            features: List[float] = []
            for feature in FEATURES:
                try: features.append(df[feature][ix])
                except: features.append(-1)
            self.data.append(Mole(graph, mol, psi, top, features))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int):
        return self.data[idx]