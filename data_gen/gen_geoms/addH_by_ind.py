#
from ase.io import write, read
from ase import Atoms
import numpy as np



def add_H_by_index(atoms, idx, bond_length=1.1):
    pos = atoms[idx].position

    # Random direction vector
    direction = np.random.rand(3) - 0.5
    direction /= np.linalg.norm(direction)
    atoms += Atoms('H', positions=[pos + bond_length * direction])
    #  atoms += Atoms('H', positions=[pos + bond_length])
    return atoms



def add_hydrogen_with_angle(atoms, idx, bond_length=1.1):
    # Position of target atom
    pos_target = atoms[idx].position

    # Get neighbors (distance-based)
    cutoff = 2.0  # Å, approximate bonding cutoff
    neighbor_vectors = []
    for j in range(len(atoms)):
        if j == idx:
            continue
        vec = atoms[j].position - pos_target
        dist = np.linalg.norm(vec)
        if dist < cutoff:
            neighbor_vectors.append(vec / dist)  # normalized

    # If no neighbors, just place random H
    if not neighbor_vectors:
        direction = np.random.rand(3) - 0.5
        direction /= np.linalg.norm(direction)
    else:
        # Sum neighbor directions (approximate bonded direction)
        avg_dir = np.sum(neighbor_vectors, axis=0)
        # Opposite direction → ensures angle > 120°
        direction = -avg_dir / np.linalg.norm(avg_dir)

    # Place H
    h_pos = pos_target + bond_length * direction
    atoms += Atoms('H', positions=[h_pos])
    return atoms


import numpy as np
from ase import Atoms

def add_hydrogen_with_angle_range(
    atoms,
    idx,
    bond_length=1.0,
    base_angle_deg=110.0,
    tolerance_deg=10.0,
    neighbor_cutoff=2.0
):
    """
    Add a hydrogen to atom `idx` such that the angle between the
    H–atom vector and the average neighbor direction is ~base_angle_deg ± tolerance_deg.
    """
    pos_target = atoms[idx].position

    # --- 1. Find neighbor directions (normalized) ---
    neighbor_dirs = []
    for j in range(len(atoms)):
        if j == idx:
            continue
        vec = atoms[j].position - pos_target
        dist = np.linalg.norm(vec)
        if 0 < dist < neighbor_cutoff:
            neighbor_dirs.append(vec / dist)

    # If no neighbors, just place random H
    if not neighbor_dirs:
        direction = np.random.rand(3) - 0.5
        direction /= np.linalg.norm(direction)
    else:
        # Average direction of neighbors (pointing FROM central atom TO neighbors)
        avg_dir = np.sum(neighbor_dirs, axis=0)
        avg_dir /= np.linalg.norm(avg_dir)

        # --- 2. Build an orthonormal basis (e1 = avg_dir) ---
        e1 = avg_dir
        # Pick a vector not parallel to e1
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, e1)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])

        # Make tmp orthogonal to e1 to form e2
        e2 = tmp - np.dot(tmp, e1) * e1
        e2 /= np.linalg.norm(e2)

        # e3 is orthogonal to both (not strictly needed, but nice to have)
        e3 = np.cross(e1, e2)
        e3 /= np.linalg.norm(e3)

        # --- 3. Choose angle in [base_angle - tol, base_angle + tol] ---
        angle_deg = base_angle_deg + np.random.uniform(-tolerance_deg, tolerance_deg)
        angle_rad = np.deg2rad(angle_deg)

        # Place H in the plane spanned by e1 and e2
        # v_H = cos(theta)*e1 + sin(theta)*e2
        direction = np.cos(angle_rad) * e1 + np.sin(angle_rad) * e2
        direction /= np.linalg.norm(direction)

    # --- 4. Place hydrogen at bond_length along this direction ---
    h_pos = pos_target + bond_length * direction
    atoms += Atoms('H', positions=[h_pos])
    return atoms


extxyz_path = "non_equ_geoms_Co_co2_ch4_all_v6_2.extxyz"
atoms_list = read(extxyz_path, index=":")
for atoms in atoms_list:
    atoms = add_hydrogen_with_angle_range(atoms, 19)   # add H to atom index 1
    write(f"{extxyz_path.split('.')[0]}_addH.extxyz", atoms, append=True)
