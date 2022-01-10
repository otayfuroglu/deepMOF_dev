#! /truba/home/yzorlu/miniconda3/bin/python
import time
import os
import pandas as pd
import numpy as np
import schnetpack as spk

from schnetpack.datasets import AtomsData
import torch
from model_config_param import config

from multiprocessing import Pool
import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda"

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-fragName", "--fragName", type=str,
                    required=True, help="give molecule of fragment base name")
parser.add_argument("-dbBase", "--dbBase", type=str,
                    required=True, help="give db base name")
args = parser.parse_args()

# basic settings
# dbBase = "nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScalingAndGuestCO2"
dbBase = args.dbBase
fragName = args.fragName
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP"
dataPath = "%s/prepare_data/workingOnDataBase/%s.db" % (BASE_DIR, dbBase)
minMaxSF_csvName = "%s_MinMaxValues.csv" % fragName

dataset = AtomsData(dataPath)
_, properties = dataset.get_properties(0)

properties = [propert for propert in properties.keys() if "_" not in propert]

# save to current work directory.
fragDataPath = "%s_%s.db" % (fragName, dbBase)

frag_db = AtomsData(
    fragDataPath,
    available_properties=properties,
)


def checkPath(path):
    return os.path.exists(path)

def _getFragDataBase(idx):
    i = idx
    file_base = dataset.get_name(i)
    if fragName in file_base and "co2" not in file_base:
        property_values = []
        for propert in properties:

            mol = dataset.get_atoms(i)
            target_propert = dataset[i][propert]
            target_propert = np.array(target_propert, dtype=np.float32)
            property_values.append(target_propert)

        # combine two lists into a dictionary
        property_dict = dict(zip(properties, property_values))
        return mol, file_base, property_dict # order: atom_list, name_list, property list

def getFragDataBaseP(num_processes):

    n_sample = len(dataset)
    print("Fragment data will obtain ...")
    print("Number of Samples: ", n_sample)

    pool = Pool(processes=num_processes)
    #  result_list_tqdm = []
    # implementation of  multiprocessor in tqdm. Ref.https://leimao.github.io/blog/Python-tqdm-Multiprocessing/
    idxs = range(n_sample)

    atoms_list = []
    name_list = []
    property_list = []
    for result in tqdm.tqdm(pool.imap_unordered(func=_getFragDataBase, iterable=idxs),
                            total=n_sample):
        #  result_list_tqdm.append(result)
        if result:
            atom, name, property_dict = result
            atoms_list.append(atom)
            name_list.append(name)
            property_list.append(property_dict)

    #  property_list, atoms_list, name_list = result_list_tqdm[0]
    frag_db.add_systems(atoms_list, name_list, property_list)


def calculateSF(config, sample):

    representation = spk.representation.BehlerSFBlock(
        config["n_radial"],
        config["n_angular"],
        zetas={1},
        cutoff_radius=config["cutoff_radius"],
        elements=frozenset((1, 6, 8, 30)),
        centered=False,
        crossterms=False,
        mode=config["mode"],  # "weighted" for wACSF "Behler" for ACSF

    ).to(device) # cuda device for accelration

    # spk.representation.StandardizeSF(
    #     representation,
    #     train_loader,
    #     cuda=device,
    # )
    if device == "cuda":
        sample = {k: v.to(device) for k, v in sample.items()}

    return representation.forward(sample)


def getMinMaxSF(config):

    if not checkPath(fragDataPath):
        getFragDataBaseP(num_processes=10)

    dataset = AtomsData(fragDataPath, collect_triples=True)
    n_sample = len(dataset)
    print("Max and Min SF values will calculate for each atom in mol")
    print("Number of Fragmnet Samples: ", n_sample)

    num_workers = 40
    data_loader = spk.AtomsLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    max_symfunc_values = []
    min_symfunc_values = []

    for sample in tqdm.tqdm(data_loader):
        symfunc_valuesForMol = calculateSF(config, sample)
        max_symfunc_values.append(torch.max(symfunc_valuesForMol.squeeze(0).to(device),
                                            dim=1)[0].unsqueeze(0))
        min_symfunc_values.append(torch.min(symfunc_valuesForMol.squeeze(0).to(device),
                                            dim=1)[0].unsqueeze(0))

    max_valuesOverAll = torch.max(torch.cat(max_symfunc_values, 0), dim=0)[0].to("cpu")
    min_valuesOverAll = torch.min(torch.cat(min_symfunc_values, 0), dim=0)[0].to("cpu")

    df = pd.DataFrame()

    df["MaxValues"] = max_valuesOverAll
    df["MinValues"] = min_valuesOverAll
    df.to_csv(minMaxSF_csvName)

if __name__ == "__main__":
    st = time.time()
    if not os.path.exists(minMaxSF_csvName):
        getMinMaxSF(config)
        print("Spend time: %f min." % ((time.time() - st) / 60.0))
