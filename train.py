from regionsizeexperiments.moltransformer import MolTransformer
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
from modules.preprocessing.dataloader import FEATURES, Tox21, tox21_collate

# CONFIG
d_model = 32
d_attn = 16

classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(d_model, 1), nn.Softmax()) for _ in FEATURES ])


def train(model: nn.Module, dataset: Tox21):
    losses = []
    tasks = []

    # create data
    train_data, val_data = random_split(dataset, [0.7, 0.3])
    train_loader = DataLoader(train_data, shuffle=True, batch_size=4, collate_fn=tox21_collate)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=1, collate_fn=tox21_collate)

    opt = optim.AdamW(model.parameters(), 3e-4)

    for i, data in enumerate(train_loader):
        graph, top, features = data

        opt.zero_grad()
        

model = MolTransformer(32, 2, 8, 16, 11, False)
item = Tox21()[0]
model.forward(item.graph, item.top)