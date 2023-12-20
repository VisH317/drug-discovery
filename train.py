import os
import configparser
from regionsizeexperiments.moltransformer import MolTransformer
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
from modules.preprocessing.dataloader import FEATURES, Tox21, tox21_collate

# config
CFG_PATH = "config/rse.ini"

classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(d_model, 1), nn.Softmax()) for _ in FEATURES ])
model = MolTransformer.from_config(CFG_PATH)

# train config
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), CFG_PATH))

batch_size = config.get("training", "batch_size")
n_epochs = config.get("training", "n_epochs")
val_size = config.get("training", "val_size")
val_step = config.get("training", "val_step")


def train(model: nn.Module, dataset: Tox21):
    losses = []
    tasks = []

    opt = optim.AdamW(model.parameters(), 3e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(gamma=0.9)

    for epoch in range(n_epochs):

        # create data
        train_data, val_data = random_split(dataset, [0.7, 0.3])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=tox21_collate)
        val_loader = DataLoader(val_data, shuffle=True, batch_size=val_size, collate_fn=tox21_collate)

        epoch_losses = []
        epoch_val = []

        for i, data in enumerate(train_loader):
            graph, top, features = data

            opt.zero_grad()
        

model = MolTransformer(32, 2, 8, 16, 11, False)
item = Tox21()[0]
print(item)