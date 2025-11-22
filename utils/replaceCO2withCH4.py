from ase import Atoms
from ase.io import read, write
import numpy as np
import argparse



def find_co2_molecules(atoms, distance_cutoff=1.5, angle_tolerance=10):
    """Finds CO2 molecules: a C atom with two O neighbors at correct distance and ~180° angle."""
    co2_indices = []
    for i, atom in enumerate(atoms):
        if atom.symbol == 'C':
            neighbors = []
            for j, other in enumerate(atoms):
                if other.symbol == 'O':
                    dist = atoms.get_distance(i, j)
                    if dist < distance_cutoff:
                        neighbors.append(j)
            if len(neighbors) == 2:
                # Check if O-C-O angle is close to 180°
                vec1 = atoms[neighbors[0]].position - atoms[i].position
                vec2 = atoms[neighbors[1]].position - atoms[i].position
                cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.degrees(np.arccos(cosine_angle))
                if abs(angle - 180) <= angle_tolerance:
                    co2_indices.append((i, neighbors[0], neighbors[1]))
    return co2_indices

def create_methane(position):
    """Creates a methane (CH4) molecule with tetrahedral geometry centered at position."""
    bond_length = 1.09  # Approximate C-H bond length in angstroms

    # Tetrahedral unit vectors
    vectors = np.array([
        [ 1,  1,  1],
        [-1, -1,  1],
        [-1,  1, -1],
        [ 1, -1, -1],
    ])
    vectors = bond_length * vectors / np.linalg.norm(vectors[0])

    methane = Atoms('CHHHH',
                    positions=np.vstack(([0, 0, 0], vectors)))
    methane.translate(position)
    return methane


parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-extxyz_path", type=str, required=True,)
args = parser.parse_args()

# --- Load the structure ---
extxyz_path = args.extxyz_path
atoms_list = read(extxyz_path, index=":")  # Replace with your real filename


for atoms in atoms_list:
    # --- Find all CO2 molecules ---
    co2_groups = find_co2_molecules(atoms)
    print(f"Found {len(co2_groups)} CO2 molecules.")

    # --- Prepare new structure ---
    new_atoms = atoms.copy()

    # --- Remove C and O atoms of CO2 ---
    # Flatten the list of indices to remove
    indices_to_remove = sorted([idx for group in co2_groups for idx in group], reverse=True)
    for idx in indices_to_remove:
        del new_atoms[idx]

    # --- Add CH4 molecules at C positions ---
    for group in co2_groups:
        c_index = group[0]
        c_position = atoms[c_index].position
        methane = create_methane(c_position)
        new_atoms += methane

    # --- Save the new structure ---
    write(extxyz_path.replace("CO2", "CH4"), new_atoms, append=True)
