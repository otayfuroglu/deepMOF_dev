import numpy as np
from ase.thermochemistry import HarmonicThermo
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.io import read
#  from asebazant.bazant_calc import BazantCalculator
from n2p2AseInterFace import n2p2Calculator
from ase.calculators.lammpsrun import LAMMPS

from ase import units
import minimahopping.opt.optim as opt
import argparse, os

import tqdm


def getLammpsCalc(supercoder:list) -> LAMMPS:

    prepareModel()

    parameters = {
        'units': 'metal',
        "atom_style": "atomic",
        'pair_style': ' nnp dir ./  maxew 30000 cflength 1.8897261328 cfenergy 0.0367493254  emap "1:Al,2:Li,3:H"',
        'pair_coeff': ['* * 6.5']}

    return LAMMPS(parameters=parameters, specorder=supercoder, tmp_dir="tmp_lammps")


def prepareModel():
    os.system(f"cp {args.MODEL_DIR}/input.nn ./")
    os.system(f"cp {args.MODEL_DIR}/scaling.data ./")
    weights_files = [item for item in os.listdir(args.MODEL_DIR) if "weights" in item]
    best_weights_files = [item for item in weights_files if int(item.split(".")[-2]) == args.best_epoch]
    assert len(best_weights_files) != 0, "Erro: NOT FOUND best epoch number"
    for best_weights_file in best_weights_files:
        os.system(f"cp {args.MODEL_DIR}/{best_weights_file} ./{best_weights_file[:11]}.data")



def test():


    prepareModel()

    parameters = {
        'units': 'metal',
        "atom_style": "atomic",
        'pair_style': ' nnp dir ./  maxew 30000 cflength 1.8897261328 cfenergy 0.0367493254  emap "1:Al,2:Li,3:H"',
        'pair_coeff': ['* * 6.5']}
    #  calc = LAMMPSlib(lmpcmds=cmds, log_file='lammps.log')
    #  calc = LAMMPS(parameters=parameters, specorder=["Al", "Li", "H"], tmp_dir="tmp")
    supercoder = ["Al", "Li", "H"]
    calc = getLammpsCalc(supercoder)
    atoms = read(struc_path)
    #  atoms.pbc = True
    atoms.calc = calc


    print(atoms.get_potential_energy())


def main():
    # read the structures (OMER HERE YOU HAVE TO READ YOUR STRUCTURES)
    structs = read(struc_path, index=":")
    # set up a calculator (OMER HERE YOU HAVE TO USE N2P2 OR NEQUIP)
    #  calculator = BazantCalculator()

    calculator = getLammpsCalc(supercoder=["Al", "Li", "H"])

    f =open(f"free_energies_{struc_path.split('.')[0]}.csv", "w")
    f.write("Label,PotE,HelmholtzFreeE,Temp\n")

    free_energies=[]
    for atoms in tqdm.tqdm(structs):
        # attach calculator to atoms object
        atoms.calc = calculator
        fmax = 5e-3  # convergence criterion for geometry optimization
        outpath = './' # output path of debug files for gemetry optimization
        initial_step_size = 1e-2 # initial step size
        nhist_max = 10 # maximal history of sqnm
        lattice_weight = 2 # lattice weight for geometry optimization
        alpha_min = 1e-3 # lowest possible step size
        eps_subsp = 1e-3 # subspace stepsize
        verbose_output = True # verbosity of the output

        # perform the geometry optimization
        positions, lattice, noise, opt_trajectory, number_of_opt_steps, epot_max_geopt = opt.optimization(atoms=atoms,
                                                                    calculator=calculator,
                                                                    max_force_threshold=fmax,
                                                                    outpath=outpath,
                                                                    initial_step_size=initial_step_size,
                                                                    nhist_max=nhist_max,
                                                                    lattice_weight=lattice_weight,
                                                                    alpha_min=alpha_min,
                                                                    eps_subsp=eps_subsp,
                                                                    verbose=verbose_output)

        # set optimized postions and lattice
        atoms.set_positions(positions)
        atoms.set_cell(lattice)

        T = 0.00001 # temperature at which free energy is evaluated
        h = 1e-4 # shift to get the hessian matrix

        # evaluate hessian matrix
        vib = Vibrations(atoms, delta = h)
        vib.run()
        vib_energies = vib.get_energies()
        vib.clean()
        vib_energies = [mode for mode in np.real(vib_energies) if mode > 1e-3]
        vib_energies = [complex(1.0e-8, 0) if energy < 1.0e-4 else energy for energy in vib_energies]

        potentialenergy = atoms.get_potential_energy()

        # get free energy from ase
        #  free_energy_class = HarmonicThermo(vib_energies, potentialenergy=0.)#potentialenergy)
        free_energy_class = HarmonicThermo(vib_energies, potentialenergy=0.0)
        free_energy = free_energy_class.get_helmholtz_energy(T, verbose=True)
        free_energies.append(free_energy)

        #  atoms.free_energy = free_energy
        #  write(f"opt_freq_{struc_path}", atoms)

        f.write(f"{atoms.info['label']},{potentialenergy},{free_energy},{T}\n")
        f.flush()

    f.close()

    #  diff = []
    #  for i in range(len(free_energies)):
    #      for j in range(i+1, len(free_energies)):
    #          difference = free_energies[i] - free_energies[j]
    #          diff.append(difference)
    #          f.write(str(difference) + '\n')
    #  f.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-struc_path", type=str, required=True, help="..")
    parser.add_argument("-MODEL_DIR", type=str, required=True, help="..")
    parser.add_argument("-energy_u", type=str, required=True, help="..")
    parser.add_argument("-length_u",type=str, required=True, help="..")
    parser.add_argument("-best_epoch", type=int, required=True, help="..")
    args = parser.parse_args()
        #  nn = MinimumDistanceNN()
    struc_path = args.struc_path

    #  test()
    main()


