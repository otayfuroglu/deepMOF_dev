#! /truba/home/yzorlu/miniconda3/bin/python
import os
import logging
from torch.optim import Adam
import schnetpack as spk
import schnetpack.atomistic.model

from schnetpack.train import Trainer, CSVHook, ReduceLROnPlateauHook
from schnetpack.train.metrics import RootMeanSquaredError, MeanAbsoluteError
from schnetpack.train import build_mse_loss
from schnetpack.datasets import AtomsData
from schnetpack.environment import TorchEnvironmentProvider, AseEnvironmentProvider
from schnetpack.nn.cutoff import HardCutoff
from trainingHelper import get_loss_fn, get_metrics

import torch

import get_atomrefs

# from functools import partial
# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

# set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
n_gpus = torch.cuda.device_count()
mof_number = "1_4_6_7_10"


def createModelName(config, descriptive_word, UNIT):
    return "schnet_l{}_basis{}_filter{}_interact{}_gaussian{}_rho{}_lr{}_bs{}_cutoff_{}_{}_{}".format(
        config["n_layers"],
        config["n_atom_basis"],
        config["n_filters"],
        config["n_interactions"],
        config["n_gaussians"],
        str(config["rho"]).replace(".", ""),
        str(config["lr"]).replace(".", ""),
        config["batch_size"],
        str(config["cutoff_radius"]).replace(".", ""),
        descriptive_word,
        UNIT)


config = {
          "n_atom_basis": 96,
          "n_filters": 64,
          "n_interactions": 3,
          "n_gaussians": 20,
          "n_layers": 3,
          "max_z": 100,
          "lr": 0.0001,
          "batch_size": 1,
          "cutoff_radius": 6.0,
          "rho": 0.01,
          "stress": None,
          "negative_dr": True,
          "derivative": "forces",
          "property": "energy",
}


# set cuda visible devices
#  if config["batch_size"] > 1:
#      if n_gpus == 2:
#          os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#      if n_gpus == 4:
#          os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# basic settings
UNIT = "ev"
BASE_DIR = "/truba_scratch/yzorlu/deepMOF/HDNNP"

DATA_DIR = ("%s/prepare_data/workingOnDataBase"
            "/nonEquGeometriesEnergyForcesWithORCA_TZVP_fromScaling_IRMOFseries%s_merged_50000_%s.db" % (BASE_DIR, mof_number, UNIT))

# data preparation
properties = ["energy", "forces"]
dataset = AtomsData(DATA_DIR,
                    # available_properties=properties,
                    load_only=properties,
                    collect_triples=False,
                    environment_provider=AseEnvironmentProvider(config["cutoff_radius"]),
                    #  environment_provider=TorchEnvironmentProvider(config["cutoff_radius"], "cpu"),
                   )

descriptive_word = "withoutStress_aseEnv_IRMOFseries%s_merged_%s" %(mof_number, len(dataset))
trainingName = createModelName(config, descriptive_word, UNIT)
MODEL_DIR = os.path.join(os.getcwd(), trainingName)

check_flag = 0
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    check_flag = 1

# logging to logFile
logFile = os.path.join(MODEL_DIR, "%s.log" % trainingName)
if os.path.exists(logFile):
    os.remove(logFile)
logging.basicConfig(
    filename=logFile,
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=os.environ.get("LOGLEVEL", "INFO"),
)

if check_flag == 0:
    logging.info("Warning: model will be restored from"
                 + "checkpiont in the %s directory!" % MODEL_DIR)

logging.info("Job started %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
logging.info("Job ID: %d" % os.getpid())
logging.info("device type --> %s" % device)
logging.info("Number of cuda devices --> %s" % n_gpus,)

#  _, properties = dataset.get_properties(0)
#  properties = [item for item in properties.keys() if "_" != item[0]]
# del properties[1]

logging.info("available properties --> %s" % properties)

n_sample = len(dataset)
logging.info("Number of sample --> %s" % n_sample)


def run_train(config,  checkpoint_dir=None):

    train, val, test = spk.train_test_split(
        data=dataset,
        num_train=int(n_sample * 0.8),
        num_val=int(n_sample * 0.1), # 0.1 remain data for test
        split_file=os.path.join(MODEL_DIR, "split.npz"),
    )
    num_workers = 40
    train_loader = spk.AtomsLoader(train,
                                   batch_size=config["batch_size"],
                                   shuffle=True, num_workers=num_workers,
                                   pin_memory=True)
    val_loader = spk.AtomsLoader(val,
                                 batch_size=config["batch_size"],
                                 num_workers=num_workers,
                                 pin_memory=True)

    # get statistics
    atomrefs = get_atomrefs.atomrefs_energy0(properties[0], UNIT)
    #  per_atom = {properties[0]: True, properties[1]: False}
    means, stddevs = train_loader.get_statistics(
        [properties[0]],
        single_atom_ref=atomrefs,
        divide_by_atoms=True,
    )

    # model build
    logging.info("build model")
    representation = spk.representation.SchNet(
        n_atom_basis=config["n_atom_basis"],
        n_filters=config["n_filters"],
        n_interactions=config["n_interactions"],
        cutoff=config["cutoff_radius"],
        n_gaussians=config["n_gaussians"],
        normalize_filter=False,
        coupled_interactions=False,
        return_intermediate=False,
        max_z=config["max_z"],
        cutoff_network=HardCutoff,
        trainable_gaussians=False,
        distance_expansion=None,
        charged_systems=False,
    )


    output_modules = [
        schnetpack.atomistic.Atomwise(
            n_in=representation.n_atom_basis,
            n_out=1,
            n_layers=config["n_layers"],
            aggregation_mode="sum",
            n_neurons=None,
            activation=schnetpack.nn.activations.shifted_softplus,
            property=properties[0],
            mean=means[properties[0]],
            derivative=config["derivative"],
            stddev=stddevs[properties[0]],
            atomref=atomrefs[properties[0]],
            stress=config["stress"],
            negative_dr=config["negative_dr"],
            create_graph=False,
            outnet=None,
        )
    ]


    model = schnetpack.atomistic.model.AtomisticModel(representation,
                                                      output_modules)
    # for multi GPU
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    # build optimizer
    optimizer = Adam(params=model.parameters(), lr=config["lr"])

    # hooks
    logging.info("build trainer")

    if len(properties) == 1:
        metrics = [MeanAbsoluteError(p, p) for p in properties]
        metrics += [RootMeanSquaredError(p, p) for p in properties]
        loss = build_mse_loss(properties, loss_tradeoff=[0.1]) # for ["energy"]
    elif len(properties) == 2:
        metrics = [MeanAbsoluteError(p, p) for p in properties]
        metrics += [RootMeanSquaredError(p, p) for p in properties]
        rho = config["rho"]
        loss = build_mse_loss(properties,
                              loss_tradeoff=[rho, 1 - rho])
    else:
        # get loss with trade off functions
        loss = get_loss_fn(argsDict=config)
        metrics = get_metrics(argsDict=config)

    hooks = [CSVHook(log_path=MODEL_DIR, metrics=metrics),
             ReduceLROnPlateauHook(
                 optimizer,
                 patience=5,
                 factor=0.5,
                 min_lr=1e-6,
                 # window_length=1,
                 stop_after_min=False)]

    # trainer
    trainer = Trainer(
        MODEL_DIR,
        # train_type="train",
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )

    # run training
    logging.info("training")
    trainer.train(device=device, n_epochs=1000)

logging.info("Train configuration:")
for key, val in config.items():
    logging.info("%s --> %s" % (key, val))

run_train(config)
logging.info("training was done")
