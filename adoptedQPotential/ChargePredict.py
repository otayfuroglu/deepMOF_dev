import os
import sys
import torch
import numpy as np
import pickle as pk
from ase.io import read
import torch.nn as nn
from ase.data import covalent_radii as cv
from ase.neighborlist import NeighborList
from torch.utils.data import DataLoader, TensorDataset
from dscribe.descriptors import ACSF
from NN import EnergyPredictor
from acs_params import g2_params, g4_params


class ChargeCalculator:
    def __init__(self, param_file='paramFile.dat', model_file='best_model.pth', hidden_size1=None, hidden_size2=None):
        self.hidden_size1=hidden_size1
        self.hidden_size2=hidden_size2
        self.param_file = param_file
        self.model_file = model_file
        self.up = self._grep_keyword('u=')
        self.lw = self._grep_keyword('l=')
        self.inpNormMax, self.inpNormMin = self._get_input_norm_params()
        self.trgNormMax, self.trgNormMin = self._get_target_norm_params()
        self.model = self._load_model()

    def _grep_keyword(self, keyword):
        with open(self.param_file, 'r') as file:
            lines = file.readlines()
        matching_lines = [line.strip() for line in lines if keyword in line]
        return float(matching_lines[0].split('=')[-1]) if matching_lines else None

    def _get_input_norm_params(self):
        input_norm_lines = self._grep_keyword_lines('fpNormParam')
        inpNormMax, inpNormMin = [], []
        for line in input_norm_lines:
            inpNormMin.append(float(line.split()[4]))
            inpNormMax.append(float(line.split()[6]))
        return torch.tensor(inpNormMax), torch.tensor(inpNormMin)

    def _grep_keyword_lines(self, keyword):
        with open(self.param_file, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines if keyword in line]

    def _get_target_norm_params(self):
        target_norm_lines = self._grep_keyword_lines('trgNormParam')
        trgNormMin = float(target_norm_lines[0].split()[2])
        trgNormMax = float(target_norm_lines[0].split()[4])
        return trgNormMax, trgNormMin

    def _load_model(self):
        input_size = len(self.inpNormMax)
        hidden_size1, hidden_size2 = self.hidden_size1, self.hidden_size2
        model = EnergyPredictor(input_size, hidden_size1, hidden_size2)
        model.load_state_dict(torch.load(self.model_file))
        model.eval()
        return model

    def calculate_acsf(self, atoms):
        calculator = ACSF(
            species=["H", "Mg", "C", "O"],
            r_cut=8.0,
            g2_params=g2_params,
            g4_params=g4_params
        )
        sf = calculator.create(atoms, n_jobs=32)
        sf[np.isnan(sf)]=0
        return sf

    def create_dataset(self, structure):
        inputs = [self.calculate_acsf(structure)]
        #  energies_elc = [structure.get_array('DDECPQ')]
        return torch.Tensor(np.array(inputs))#, torch.Tensor(np.array(energies_elc))

    def get_charge(self, struct):
        #  valid_inputs, valid_target_elc = self.create_dataset(struct)
        valid_inputs = self.create_dataset(struct)

        # Convert lists to tensors
        inpNormMin_tensor = torch.tensor(self.inpNormMin, dtype=torch.float32)
        inpNormMax_tensor = torch.tensor(self.inpNormMax, dtype=torch.float32)

        # Vectorized normalization
        norm_range = self.up - self.lw
        inp_norm_range = inpNormMax_tensor - inpNormMin_tensor
        valid_inputs = (valid_inputs - inpNormMin_tensor) / inp_norm_range * norm_range + self.lw
        # Use the model to predict charges
        with torch.no_grad():
            NNP_Ener_elc_normalized = self.model(valid_inputs)[0].squeeze()
        # Vectorized denormalization
        trg_norm_range = self.trgNormMax - self.trgNormMin
        NNP_Ener_elc = (NNP_Ener_elc_normalized - self.lw) / norm_range * trg_norm_range + self.trgNormMin
        return NNP_Ener_elc

# Usage
if __name__ == "__main__":
    nn1=200
    nn2=160
    calculator = ChargeCalculator(hidden_size1=nn1,hidden_size2=nn2)
    struct = read(sys.argv[1])
    charges = calculator.get_charge(struct)
    print(charges)

