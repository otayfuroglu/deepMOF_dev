import os
import re
from io import StringIO
from pathlib import Path

import numpy as np

from ase.io import read
from ase.units import Bohr, Hartree
from ase.utils import reader, writer

# Made from NWChem interface


@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif (line.startswith('end') and stopline == -1):
            stopline = index
        elif (line.startswith('*') and stopline == -1):
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0., 0., 0.))  # no unit cell defined

    return atoms


@writer
def write_orca(fd, atoms, params):
    # conventional filename: '<name>.inp'
    fd.write(f"! {params['orcasimpleinput']} \n")
    fd.write(f"{params['orcablocks']} \n")

    fd.write('*xyz')
    fd.write(" %d" % params['charge'])
    fd.write(" %d \n" % params['mult'])
    for atom in atoms:
        if atom.tag == 71:  # 71 is ascii G (Ghost)
            symbol = atom.symbol + ' : '
        else:
            symbol = atom.symbol + '   '
        fd.write(symbol +
                 str(atom.position[0]) + ' ' +
                 str(atom.position[1]) + ' ' +
                 str(atom.position[2]) + '\n')
    fd.write('*\n')


@reader
def read_orca_energy(fd):
    """Read Energy from ORCA output file."""
    text = fd.read()
    re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
    re_not_converged = re.compile(r"Wavefunction not fully converged")

    found_line = re_energy.finditer(text)
    energy = float('nan')
    for match in found_line:
        if not re_not_converged.search(match.group()):
            energy = float(match.group().split()[-1]) * Hartree
    if np.isnan(energy):
        raise RuntimeError('No energy')
    else:
        return energy


@reader
def read_orca_forces(fd):
    """Read Forces from ORCA output file."""
    getgrad = False
    gradients = []
    tempgrad = []
    for i, line in enumerate(fd):
        if line.find('# The current gradient') >= 0:
            getgrad = True
            gradients = []
            tempgrad = []
            continue
        if getgrad and "#" not in line:
            grad = line.split()[-1]
            tempgrad.append(float(grad))
            if len(tempgrad) == 3:
                gradients.append(tempgrad)
                tempgrad = []
        if '# The at' in line:
            getgrad = False

    forces = -np.array(gradients) * Hartree / Bohr
    return forces


@reader
def read_orca_h_charges(fd):
    """Read Hirshfeld charges from ORCA output file."""

    lines = fd.readlines()

    h_charges = []
    start = False
    stop = False
    for i, line in enumerate(lines):
        if "HIRSHFELD ANALYSIS" in line:
            start = True
            i_start = i
        if not stop:
            if start:
                if "TOTAL" in line:
                    if len(h_charges) == 0:
                        raise RuntimeError('No charges')
                    return np.array(h_charges)
                if i >= (i_start + 7):
                    if not len(line) == 1:
                        charge = float(line.split()[2])
                        h_charges += [charge]


@reader
def read_orca_chelpg_charges(fd):
    """Read CHELPG charges from ORCA pc_chelpg file."""

    lines = fd.readlines()

    chelpg_charges = []
    #  stop = False
    for i, line in enumerate(lines):
        if i > 1:
            charge = float(line.split()[1])
            chelpg_charges += [charge]
    if len(chelpg_charges) == 0:
        raise RuntimeError('No charges')
    return np.array(chelpg_charges)


@reader
def read_orca_ddec_charges(fd):
    """Read DDEC charges from chargemol output which obtained orca wfx file."""

    lines = fd.readlines()

    ddec_charges = []
    for i, line in enumerate(lines):
        if i > 1:
            if len(line) <= 2:
                if len(ddec_charges) == 0:
                    raise RuntimeError('No charges')
                return np.array(ddec_charges)
            charge = float(line.split()[4])
            ddec_charges += [charge]

#  stdout_path = "/arf/home/otayfuroglu/deepMOF_dev/data_gen/works/mof74/test/sp_sp_non_equ_geoms_MgF1_v5/MgF1_00001/DDEC6_even_tempered_net_atomic_charges.xyz"
#  fd = Path(stdout_path)
#  print(read_orca_ddec_charges(fd))
#  quit()

def read_orca_outputs(directory, stdout_path):
    results = {}
    energy = read_orca_energy(Path(stdout_path))
    results['energy'] = energy
    results['free_energy'] = energy

    h_charges = read_orca_h_charges(Path(stdout_path))
    results["h_charges"] = h_charges

    pc_chelpg_path = f"{stdout_path.replace('.out', '.pc_chelpg')}"
    chelpg_charges = read_orca_h_charges(Path(pc_chelpg_path))
    results["chelpg_charges"] = chelpg_charges

    # Does engrad always exist? - No!
    # Will there be other files -No -> We should just take engrad
    # as a direct argument.  Or maybe this function does not even need to
    # exist.
    engrad_path = Path(stdout_path).with_suffix('.engrad')
    if os.path.isfile(engrad_path):
        results['forces'] = read_orca_forces(engrad_path)
    return results

