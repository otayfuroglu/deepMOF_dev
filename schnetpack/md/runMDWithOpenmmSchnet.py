#! /truba/home/otayfuroglu/miniconda3/envs/python38/bin/python -u
import os
import sys
import openmm_py
import numpy as np
import torch
import torch.nn as nn
import simtk.openmm as mm

from sys import stdout, exit
from time import sleep
from simtk.openmm import app, Platform
from simtk.openmm import CustomNonbondedForce
from simtk import unit as u

from simtk.openmm import *

from ase.io import read

from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from schnetpack.environment import AseEnvironmentProvider

from computeEnergyAndForcesForSchnet import EnergyComputer

BASEDIR = "/truba_scratch/otayfuroglu/deepMOF/HDNNP/schnetpack/"
MODELDIR = os.path.join(BASEDIR, "runTraining/schnet_l3_basis96_filter64_interact3_gaussian20_rho001_lr00001_bs1_cutoff_60_withoutStress_aseEnv_IRMOFseries6_merged_100220_ev")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelPath = os.path.join(MODELDIR, "best_model")
model = load_model(modelPath)

cutoff = 5.5
calc_schnet = SpkCalculator(model,
                            device=device,
                            energy="energy",
                            forces="forces",
                            energy_units="eV",
                            forces_units="eV/Angstrom",
                            #  stress_units="eV/Angstrom/Angstrom/Angstrom"
                            #  collect_triples=True,
                            environment_provider=AseEnvironmentProvider(cutoff),
                           )

#  def warn(*argv):
#      # write to stderr
#      print(*argv, file=sys.stderr, flush=True)


def createSystem(topology):
    # initil version taken from: simtk/openmm/app/forcefield.py
    sys = mm.System()
    for atom in topology.atoms():
        # Add the particle to the OpenMM system.
        #mass = self._atomTypes[typename].mass
        mass = atom.element.mass
        sys.addParticle(mass)

    # Set periodic boundary conditions.
    boxVectors = topology.getPeriodicBoxVectors()
    if boxVectors is not None:
        sys.setDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2])

    return sys


def Minimize(simulation, outFile, iters=150):
    simulation.minimizeEnergy(tolerance=0.001, maxIterations=iters)
    position = simulation.context.getState(getPositions=True).getPositions()
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    with open(outFile, 'w') as outF:
        app.PDBFile.writeFile(simulation.topology, position, outF)
    #  warn('Energy at Minima is {:3.3f}'.format(energy._value))
    return simulation


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")
    parser.add_argument('-in', help='Molecule',
                        dest='inFile', metavar='pdb' ,  type=str, required=True)
    parser.add_argument('-out', help='optimized molecule output',
                        dest='outFile', metavar='pdb' ,  type=str, required=True)
    args = parser.parse_args()

    # make parsed parameter local variables
    locals().update(args.__dict__)


    temperature = 100 * u.kelvin
    pdb = app.PDBFile(inFile)
    aseMol = read(inFile)
    chemicalSymbols = aseMol.get_chemical_symbols()

    modeller = app.Modeller(pdb.topology, pdb.positions)

    topo = modeller.topology
    system = createSystem(modeller.topology)
    atomNum = []
    for atom in topo.atoms():
        atomNum.append(atom.element.atomic_number)
    #  warn(atomNum)

    #################################################
    # add PY force to system
    ecomputer = EnergyComputer(chemicalSymbols=chemicalSymbols,
                               aseCalculator=calc_schnet,
                               device=device,
                              )

    #################################################
    # test ecomputer.computeEnergyAndForces
    #ecomputer.computeEnergyAndForces(aseMol.get_positions(), True, True)

    f = openmm_py.PYForce(ecomputer)
    f.setUsesPeriodicBoundaryConditions(True)
    system.addForce(f)
    ###################################################

    barostat = mm.MonteCarloBarostat(1.0*unit.bar, 200.0*unit.kelvin, 1)
    system.addForce(barostat)


    integrator = mm.LangevinIntegrator(
        temperature, 1 / u.picosecond,  0.0005 * u.picoseconds)
    print(mm.Platform.getPluginLoadFailures())
    #  platform = Platform.getPlatformByName('CUDA')
    #  prop = dict(CudaPrecision='mixed', DeviceIndex='0,1,2,3') # Use mixed single/double precision and gpu ids

    simulation = app.Simulation(modeller.topology, system, integrator)#, platform, prop)
    simulation.context.setPositions(modeller.positions)
    simulation = Minimize(simulation,outFile,1000)

    simulation.reporters.append(app.PDBReporter('output.pdb', 10))
    simulation.reporters.append(app.CheckpointReporter('output.chk', 1000))
    simulation.reporters.append(app.StateDataReporter("output.csv", 1, step=True,
        potentialEnergy=True, temperature=True, volume=True))
    simulation.reporters.append(app.StateDataReporter(sys.stdout, 1, progress=True,
                                                      remainingTime=True, totalSteps=20000))
simulation.step(20000)
