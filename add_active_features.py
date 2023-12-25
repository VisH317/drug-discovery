import pickle
from tqdm import tqdm
from collections import namedtuple

Mole = namedtuple("Mole", ['graph', 'top', 'features', 'active_features'])

with open("./data/tox21_parsed.pkl", "rb") as f:
    new_dt: list[Mole] = []
    dt = pickle.load(f)
    for mol in tqdm(dt, desc="parsing active features"):
        active_features = []
        for ix, feat in enumerate(mol.features):
            if feat == 0.0 or feat == 1.0: active_features.append(ix)
        
        new_dt.append(Mole(mol.graph, mol.top, mol.features, active_features))
        
    with open("./data/tox21_parsed_updated.pkl", "wb") as f:
        pickle.dump(new_dt, f)