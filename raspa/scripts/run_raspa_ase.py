from ase.io import read
from raspa_ase import Raspa
import os



#  os.environ["RASPA_DIR"] = "/arf/home/otayfuroglu/deepMOF_dev/raspa/gRASPA/patch_Allegro"
os.environ["RASPA_DIR"] = "/arf/home/otayfuroglu/deepMOF_dev/raspa/gRASPA/src_clean"

atoms = read("MgMOF74-small_unitcell.cif")
atoms.info = {
    "UnitCells": [0, 5, 3, 3],
    #  "HeliumVoidFraction": 0.29,
    "ExternalTemperature": 300.0,
    "ExternalPressure": 1e5,
}
components = [
    {
        "MoleculeName": "CO2",
        "CreateNumberOfMolecules": 0,
        "IdealGasRosenbluthWeight": 1.0,
        "FugacityCoefficient":     1.0,
        "TranslationProbability":   1.0,
        "RotationProbability":      1.0,
        "ReinsertionProbability":   1.0,
        "SwapProbability": 1.0,
        "DNNPseudoAtoms": "C_co2 O_co2",
    }
]
parameters = {
    "UseGPUReduction": "no",
    "Useflag": "yes",
    "NumberOfInitializationCycles": 5000,
    "NumberOfEquilibrationCycles":  0,
    "NumberOfProductionCycles":     10000,
    "UseMaxStep": "yes",
    "MaxStepPerCycle": 1,
    "UseChargesFromCIFFile": "no",
    "RestartFile": "no",
    "RandomSeed":  0,
    "BMCBiasingMethod": "no",
    "NumberOfTrialPositions": 10,
    "NumberOfTrialOrientations": 10,
    "NumberOfBlocks": 1,
    "AdsorbateAllocateSpace": 10240,
    "NumberOfSimulations": 1,
    "SingleSimulation": " yes",
    "DifferentFrameworks": " yes",
    #  "InputFileType": "cif",
    #  "FrameworkName": "MgMOF74-small_unitcell",
    "ChargeMethod": "Ewald",

    "OverlapCriteria": 1e5,
    "CutOffVDW": 14.0,
    "CutOffCoulomb": 14.0,
    "EwaldPrecision": 1e-6,

    "UseDNNforHostGuest": "yes",
    "DNNModelName": "co2-mof74-deployed.pth",
    "DNNMethod": "Allegro" ,
    "DNNEnergyUnit": "ev",
    "MaxDNNDrift": 100000.0,

}
calc = Raspa(components=components, parameters=parameters)

atoms.calc = calc
atoms.get_potential_energy()
print(calc.results)

