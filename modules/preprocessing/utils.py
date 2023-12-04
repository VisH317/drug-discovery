from torch_geometric.data import Data
import torch
from rdkit import Atom, Bond, Chem
from data import x_map, e_map
import psi4

def get_atom_info(atom: Atom):
    x = []
    x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
    x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
    x.append(x_map['degree'].index(atom.GetTotalDegree()))
    x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
    x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
    x.append(x_map['num_radical_electrons'].index(
        atom.GetNumRadicalElectrons()))
    x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
    x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
    x.append(x_map['is_in_ring'].index(atom.IsInRing()))
    
def get_edge_info(bond: Bond, m):

    idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    atom1_coords = m.GetConformer().GetAtomPosition(idx1)
    atom2_coords = m.GetConformer().GetAtomPosition(idx2)

    # convert to psi4 molecule for quantum estimations
    psi_bond = psi4.geometry(
        f"""
        2
        Atom1 {atom1_coords.x} {atom1_coords.y} {atom1_coords.z}
        Atom2 {atom2_coords.x} {atom2_coords.y} {atom2_coords.z}
        """
    )

    e = []
    e.append(e_map['bond_type'].index(str(bond.GetBondType())))
    e.append(e_map['stereo'].index(str(bond.GetStereo())))
    e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

    bond_length = psi_bond.bond_length(0, 1) # radial representation of the quantum number
    # TODO: angular representation requires a custom attention layer, I'll set this up later
    energy, wavefunction = psi4.energy("scf/cc-pvdz")
    # bond_order = wavefunction.bond_order(0, 1) # bond type and strength, based on quantum calculations, but rdkit provides this info already
    alpha, beta = wavefunction.nalpha(), wavefunction.nbeta()
    # oe_a, oe_b = wavefunction.epsilon_a_subset("A0"), wavefunction.epsilon_b_subset("AO") # we don't need these, these are eigenvalues - calculated on the hamiltonian to represent different energy states of the molecule
    # voe_a, voe_b = wavefunction.epsilon_a_subset("A0", "VIRTUAL"), wavefunction.epsilon_b_subset("AO", "VIRTUAL")
    # nmr = psi4.nmr_shielding(psi_bond, atoms=[0,1]) # magnetic info, important because its one of the quantum numbers that aren't considered in the above (radial, angular, magnetic)
    # TODO: ADD IN NMR LATER for magnetic information (it returns a 3x3 tensor per nucleus)

    e.append(bond_length)
    e.append(energy)
    e.append(alpha)
    e.append(beta)

    return [[idx1, idx2], [idx2, idx1]], [e, e]


def get_mol_info(mol):
    atoms = []

    for atom in mol.GetAtoms():
        atoms.append(get_atom_info(atom))
    
    x = torch.tensor(atoms, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []

    for bond in mol.GetBonds():
        edge_idx, edge_attr = get_edge_info(bond, mol)
        edge_indices += edge_idx
        edge_attrs += edge_attr

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 7)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return x, edge_index, edge_attr


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    x, edge_index, edge_attr = get_mol_info(mol)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
