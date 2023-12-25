import os
import torch
import configparser
from collections import namedtuple
from regionsizeexperiments.moltransformer import MolTransformer
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, random_split
from modules.preprocessing.dataloader import FEATURES, Tox21, tox21_collate, MAX_LEN
from tqdm import tqdm

# config
CFG_PATH = "config/rse.ini"

Mole = namedtuple("Mole", ['graph', 'top', 'features', 'active_features'])


def train(cfg_path: str = CFG_PATH):

    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), CFG_PATH))

    d_model = int(config.get("transformer", "d_model"))

    classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(d_model * MAX_LEN, 2), nn.Softmax(-1)).to(dtype=torch.float32) for _ in FEATURES ])
    model = MolTransformer.from_config(CFG_PATH).to(dtype=torch.float32)

    batch_size = int(config.get("training", "batch_size"))
    n_epochs = int(config.get("training", "n_epochs"))
    val_size = int(config.get("training", "val_size"))
    val_step = int(config.get("training", "val_step"))

    losses = []

    dataset = Tox21("data/tox21_parsed.pkl")

    opt = optim.AdamW(model.parameters(), 3e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.9)

    loss_fn = nn.MSELoss().to(dtype=torch.float32) # test with BCE later as well

    for epoch in range(n_epochs):

        # create data
        train_data, val_data = random_split(dataset, [0.7, 0.3])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=tox21_collate)
        val_loader = DataLoader(val_data, shuffle=True, batch_size=val_size, collate_fn=tox21_collate)

        epoch_losses = []
        epoch_val = []

        for i, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch} Loss: {epoch_losses[-1] if len(epoch_losses) > 0 else 0}", total=len(train_loader)):
            graph, top, feature_idx, feature = data

            opt.zero_grad()

            out = model(graph.to(dtype=torch.float32), top.to(dtype=torch.float32)).reshape(batch_size, d_model * MAX_LEN).to(dtype=torch.float32)

            loss = 0

            for ix in range(out.size()[0]):
                final = classifiers[feature_idx[ix]](out)
                loss += loss_fn(final[ix, 0].unsqueeze(0), torch.tensor([feature[ix]], dtype=torch.float32))
            
            epoch_losses.append(loss)
            loss.backward()
            opt.step()

            if i % val_step == 0:
                with torch.no_grad():
                    pass
                    # setup val here later
        
        losses.append(epoch_losses)
        scheduler.step()    

dataset = Tox21("data/tox21_parsed.pkl")
train()