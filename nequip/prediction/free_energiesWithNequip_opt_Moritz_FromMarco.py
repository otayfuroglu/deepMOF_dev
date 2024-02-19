import numpy as np
from ase.thermochemistry import HarmonicThermo
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from ase.io import read, write
#  from asebazant.bazant_calc import BazantCalculator
from nequip.ase import NequIPCalculator
from ase import units
import minimahopping.opt.optim as opt


def main():
    # read the structures (OMER HERE YOU HAVE TO READ YOUR STRUCTURES)
    #  structs = read("opt_nequip_polymeric_all.extxyz", index=":")
    fl_name = args.extxyz_path
    #  fl_name = "selected_10_isolated.extxyz"
    structs = read(fl_name, index=slice(0, -1))
    # set up a calculator (OMER HERE YOU HAVE TO USE N2P2 OR NEQUIP)
    model_path = "model_31k.pth"
    calculator = NequIPCalculator.from_deployed_model(
        model_path=model_path, device="cuda")

    T = 300 # temperature at which free energy is evaluated
    # to  write the free energies
    f =open(f"free_energies_{fl_name.split('.')[0]}_{int(T)}.csv", "w")
    f.write("Label,PotE,HelmholtzFreeE,Temp\n")

    free_energies=[]
    for atoms in structs:
        # attach calculator to atoms object
        atoms.calc = calculator
        fmax = 1e-5  # convergence criterion for geometry optimization
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
        free_energy_class = HarmonicThermo(vib_energies, potentialenergy=0)
        free_energy = free_energy_class.get_helmholtz_energy(T, verbose=True)
        free_energies.append(free_energy)

        #  atoms.free_energy = free_energy
        #  write(f"opt_freq_{fl_name}", atoms)

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
    import argparse
    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-extxyz_path", type=str, required=True)
    args = parser.parse_args()
    main()





