from torch_geometric.nn import GATConv

class GAT(GATConv):
    def __init__(self):
        super().__init__(-1, 64, 2, dropout=0.05, edge_dim=7)
        
