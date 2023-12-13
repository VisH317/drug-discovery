from torch_geometric.data import Data
from .utils import get_atom_info, get_edge_info, get_mol_info, smiles_to_graph
from torch_geometric.nn import GATConv



def preprocess(smiles: str):
    graph: Data = smiles_to_graph(smiles)

class Preprocess():
    def __init__(self):
        self.gat = GATConv(-1, 64, 2, dropout=0.05, edge_dim=7)
    
    def forward(self, graph: Data, mol, psi):
        preprocessed = self.gat(graph.x, graph.edge_index, edge_attr=graph.edge_attr)
        return preprocessed.x, mol, psi
    

