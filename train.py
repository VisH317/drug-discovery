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


def train(cfg_path: str = CFG_PATH, dtype: torch.dtype = torch.float32, cuda: bool = False):

    config = configparser.ConfigParser()
    config.read(os.path.join(os.getcwd(), CFG_PATH))

    d_model = int(config.get("transformer", "d_model"))

    classifiers = nn.ModuleList([ nn.Sequential(nn.Linear(d_model * MAX_LEN, 2), nn.Softmax(-1)).to(dtype=dtype) for _ in FEATURES ])
    if cuda: classifiers.cuda()

    model = MolTransformer.from_config(CFG_PATH).to(dtype=dtype)
    if cuda: model.cuda()

    batch_size = int(config.get("training", "batch_size"))
    n_epochs = int(config.get("training", "n_epochs"))
    val_size = int(config.get("training", "val_size"))
    val_step = int(config.get("training", "val_step"))
    lr = float(config.get("training", "lr"))
    grad_clip = float(config.get("training", "grad_clip"))

    losses = []
    val_losses = []

    dataset = Tox21("data/tox21_parsed.pkl")
    print(lr)
    opt = optim.AdamW(model.parameters(), lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.9)

    loss_fn = nn.CrossEntropyLoss().to(dtype=dtype) # test with BCE later as well
    if cuda: loss_fn.cuda()

    for epoch in range(n_epochs):

        # create data
        train_data, val_data = random_split(dataset, [0.7, 0.3])
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=tox21_collate)
        val_loader = iter(DataLoader(val_data, shuffle=True, batch_size=val_size, collate_fn=tox21_collate))

        epoch_losses = []
        epoch_val = []

        tr = tqdm(enumerate(train_loader), desc=f"Epoch {epoch} Loss: {epoch_losses[-1] if len(epoch_losses) > 0 else 0}", total=len(train_loader))

        for i, data in tr:
            graph, top, feature_idx, feature = data

            opt.zero_grad()

            out = model(graph.to(dtype=dtype) if not cuda else graph.to(dtype=dtype).cuda(), top.to(dtype=dtype) if not cuda else top.to(dtype=dtype).cuda()).reshape(graph.size()[0], d_model * MAX_LEN).to(dtype=dtype)

            loss = torch.zeros([1], device=torch.device("cuda" if cuda else "cpu"))

            for ix in range(out.size()[0]):
                final = classifiers[feature_idx[ix]](out)[ix]
                if cuda: final.cuda()
                features = torch.tensor([feature[ix], 0 if feature[ix] == 1 else 1], dtype=dtype, device=torch.device("cuda" if cuda else "cpu"))
                # print(features, final)
                
                cur_loss = loss_fn(final, features)
                # print("final: ", out, ", features: ", features)
                loss += cur_loss

            tr.set_description(f"Epoch {epoch} Loss: {str(epoch_losses[-1] if len(epoch_losses) > 0 else 0)}, Valid: {str(epoch_val[-1] if len(epoch_val) > 0 else 0)}")
            tr.refresh()
            print(loss)
            epoch_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

            if i % val_step == 0:
                with torch.no_grad():
                    try:
                        graph, top, feature_idx, feature = next(val_loader)
                    except:
                        val_loader = iter(DataLoader(val_data, shuffle=True, batch_size=val_size, collate_fn=tox21_collate))
                        graph, top, feature_idx, feature = next(val_loader)
                    
                    out = model(graph.to(dtype=dtype).cuda() if cuda else graph.to(dtype=dtype), top.to(dtype=dtype).cuda() if cuda else top.to(dtype=dtype)).reshape(val_size, d_model * MAX_LEN).to(dtype=dtype)
                    if cuda: out.cuda()

                    loss = 0

                    for ix in range(out.size()[0]):
                        final = classifiers[feature_idx[ix]](out)[ix, 0].unsqueeze(0)
                        if cuda: final.cuda()
                        features = torch.tensor([feature[ix]], dtype=dtype, device=torch.device("cuda" if cuda else "cpu"))

                        loss += loss_fn(final, features)
                    
                    epoch_val.append(loss.item())
        
        losses.extend(epoch_losses)
        val_losses.extend(epoch_val)
        scheduler.step()    

    return model, losses

dataset = Tox21("data/tox21_parsed.pkl")
train(dtype=torch.float16, cuda=True)