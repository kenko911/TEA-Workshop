import argparse, json
from collections import namedtuple
from timeit import default_timer
from typing import Callable
from typing import Dict, List, Tuple, Union

import os
import numpy as np
import torch
import pytorch_lightning as pl

from dgl.data import DGLDataset
from dgl.data.utils import split_dataset
from pathlib import Path
from pymatgen.core import Element, Structure, Molecule
from matgl.graph.data import MGLDataLoader, collate_fn_efs, M3GNetDataset
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.utils.training import PotentialLightningModule, xavier_init
from matgl.models import M3GNet
from matgl.apps.pes import Potential
from matgl.layers import AtomRef
from dgl.data.utils import load_graphs
from pytorch_lightning.loggers import CSVLogger
from matgl.config import DEFAULT_ELEMENT_TYPES

from torch.optim.lr_scheduler import CosineAnnealingLR

import pickle
from tqdm import tqdm, trange

module_dir = os.path.dirname(os.path.abspath(__file__))
# torch.set_default_device("cuda")
torch.set_float32_matmul_precision("high")


# torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=8)
## import structures

## Import megnet related modules
from monty.serialization import loadfn

from sklearn.model_selection import train_test_split


unit_kcal_per_mol2ev = 0.0433633636
lattice_const = [[17.25837925, 8.62899692, 0.0], [0.0, 14.9458635, 0.0], [0.0, 0.0, 38.70524185]]


def extract(filename: str):
    file = np.load(filename, allow_pickle=True)
    pos = file["R"]
    z = file["z"]
    e = np.concatenate(file["E"]) * unit_kcal_per_mol2ev
    f = file["F"] * unit_kcal_per_mol2ev
    count = 0
    structures = []
    for i in range(len(e)):
        structure = Structure(lattice_const, z, pos[i], coords_are_cartesian=True)
        structures.append(structure)
    return structures, e.tolist(), f.tolist()


train_structures, train_energies, train_forces = extract("train.npz")
element_types = get_element_list(train_structures)
cry_graph = Structure2Graph(element_types=element_types, cutoff=5.0)
import matgl
from matgl.ext.ase import M3GNetCalculator
from pymatgen.io.ase import AseAtomsAdaptor

ff = matgl.load_model(os.getcwd())
ff.calc_stress = False
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

rmse = MeanSquaredError(squared=False)
mae = MeanAbsoluteError()


def calc_error(structs: list, target_e: list, target_f: list, pot):
    p_energies = []
    p_forces = []
    for struct in tqdm(structs):
        g, st = cry_graph.get_graph(struct)
        e, f, s, h = pot(g=g, state_attr=st)
        p_energies.append(e.detach().numpy() / g.num_nodes())
        p_forces.append(f.detach().numpy())
        del g, st, e, f, s, h
    p_energies = torch.tensor(p_energies)
    p_forces = torch.tensor(np.concatenate(p_forces[:]))
    t_energies = torch.tensor(np.array(target_e[:]) / 114.0)
    t_forces = torch.tensor(np.concatenate(target_f[:]))
    MAE_E = mae(t_energies, p_energies)
    MAE_F = mae(t_forces, p_forces)
    RMSE_E = rmse(t_energies, p_energies)
    RMSE_F = rmse(t_forces, p_forces)
    print("Test mae E, mae F, rmse E, rmse F", MAE_E, MAE_F, RMSE_E, RMSE_F)
    string = (
        str(MAE_E.detach().numpy())
        + " "
        + str(MAE_F.detach().numpy())
        + " "
        + str(RMSE_E.detach().numpy())
        + " "
        + str(RMSE_F.detach().numpy())
        + "\n"
    )
    file_error.write(string)
    return None

g,st = cry_graph.get_graph(train_structures[0])
e, f, s, h = ff(g=g, state_attr=st)
print(e)
print(train_energies[0])

