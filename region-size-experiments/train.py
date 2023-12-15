from .control import MolTransformer
from torch import nn, Tensor
from modules.preprocessing.dataloader import FEATURES, Tox21

# CONFIG
d_model = 32
d_attn = 16

classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(d_model, 1), nn.Softmax()) for _ in FEATURES ])

def train():


