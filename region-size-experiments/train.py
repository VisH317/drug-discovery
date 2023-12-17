from .control import MolTransformer
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from modules.preprocessing.dataloader import FEATURES, Tox21

# CONFIG
d_model = 32
d_attn = 16

classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(d_model, 1), nn.Softmax()) for _ in FEATURES ])


def train(model: nn.Module, dataset: Tox21):
    losses = []
    tasks = []

    # create data
    train_data, val_data = random_split(dataset, [0.7, 0.3])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=16)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=16)

    for i, data
    
    

