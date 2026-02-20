#!/usr/bin/env python3
import numpy as np
from collections import defaultdict, deque

from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell, molecule
from ase.data import atomic_numbers, atomic_masses
import argparse


# =========================
# Helpers
# =========================
def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (via random quaternion)."""
    u1, u2, u3 = rng.random(3)
    q1 = np.sqrt(1 - u1) * np.sin(2*np.pi*u2)
    q2 = np.sqrt(1 - u1) * np.cos(2*np.pi*u2)
    q3 = np.sqrt(u1) * np.sin(2*np.pi*u3)
    q4 = np.sqrt(u1) * np.cos(2*np.pi*u3)
    # quaternion to rotation
    R = np.array([
        [1 - 2*(q3*q3 + q4*q4),     2*(q2*q3 - q1*q4),     2*(q2*q4 + q1*q3)],
        [    2*(q2*q3 + q1*q4), 1 - 2*(q2*q2 + q4*q4),     2*(q3*q4 - q1*q2)],
        [    2*(q2*q4 - q1*q3),     2*(q3*q4 + q1*q2), 1 - 2*(q2*q2 + q3*q3)]
    ])
    return R

def min_image_diff_frac(df: np.ndarray) -> np.ndarray:
    """Apply minimum image in fractional coords in all 3 dims."""
    return df - np.rint(df)

def frac_to_cart(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return frac @ cell

def cart_to_frac(pos: np.ndarray, inv_cell: np.ndarray) -> np.ndarray:
    return pos @ inv_cell

def check_min_dist(new_pos_cart: np.ndarray, existing_pos_cart: np.ndarray,
                   cell: np.ndarray, inv_cell: np.ndarray, min_dist: float) -> bool:
    """Return True if all distances >= min_dist under full 3D PBC."""
    if existing_pos_cart.size == 0:
        return True

    new_frac = cart_to_frac(new_pos_cart, inv_cell)
    exist_frac = cart_to_frac(existing_pos_cart, inv_cell)

    # For each new atom, compute distances to all existing atoms with PBC
    min2 = min_dist * min_dist
    for f in new_frac:
        df = exist_frac - f
        df = min_image_diff_frac(df)
        dcart = frac_to_cart(df, cell)
        d2 = np.einsum("ij,ij->i", dcart, dcart)
        if np.any(d2 < min2):
            return False
    return True

def place_molecule_randomly(mol: Atoms, cell: np.ndarray, inv_cell: np.ndarray,
                            rng: np.random.Generator,
                            existing_pos_cart: np.ndarray,
                            min_dist: float, max_tries: int) -> Atoms:
    """Randomly rotate + translate molecule into the periodic cell without overlaps."""
    cell = np.asarray(cell)
    a, b, c = cell

    mol0 = mol.copy()
    # center at COM
    com = mol0.get_center_of_mass()
    mol0.translate(-com)

    for _ in range(max_tries):
        # random rotation
        R = random_rotation_matrix(rng)
        pos = mol0.get_positions() @ R.T

        # random fractional position for COM
        f = rng.random(3)
        r = f[0]*a + f[1]*b + f[2]*c
        pos = pos + r

        if check_min_dist(pos, existing_pos_cart, cell, inv_cell, min_dist):
            placed = mol.copy()
            placed.set_cell(cell)
            placed.set_pbc([True, True, True])
            placed.set_positions(pos)
            return placed

    raise RuntimeError("Failed to place a molecule without overlaps. "
                       "Try fewer molecules, larger supercell, or smaller MIN_DIST.")

# =========================
# Optional: your custom LAMMPS writer (same ordering idea)
# =========================
def write_lammps_data_custom(filename, atoms, charge_default=0.0):
    """
    Atom types are keyed by (symbol, group_id) and ordered:
      1) MOF (group_id=0) by mass desc
      2) CO2 (group_id=2): C then O
      3) CH4 (group_id=3): C then H
      4) others
    """
    cell = np.array(atoms.get_cell())
    symbols = atoms.get_chemical_symbols()
    group_ids = atoms.arrays["group_id"]
    mol_ids = atoms.arrays.get("mol_id", np.zeros(len(atoms), int))

    # Build unique keys in appearance order
    keys = []
    for s, g in zip(symbols, group_ids):
        k = (s, int(g))
        if k not in keys:
            keys.append(k)

    def elem_mass(sym):
        return atomic_masses[atomic_numbers[sym]]

    mof = [k for k in keys if k[1] == 0]
    co2 = [k for k in keys if k[1] == 2]
    ch4 = [k for k in keys if k[1] == 3]
    other = [k for k in keys if k not in mof + co2 + ch4]

    mof = sorted(mof, key=lambda k: -elem_mass(k[0]))
    co2 = sorted(co2, key=lambda k: {"C": 0, "O": 1}.get(k[0], 99))
    ch4 = sorted(ch4, key=lambda k: {"C": 0, "H": 1}.get(k[0], 99))
    other = sorted(other, key=lambda k: (k[1], k[0]))

    ordered = mof + co2 + ch4 + other
    type_map = {k: i+1 for i, k in enumerate(ordered)}

    # Use ASE cell; write as triclinic-ish (simple; OK for most)
    # For a robust triclinic writer you already have one; reuse it if you prefer.
    # Here we keep it minimal: orthogonal assumption not enforced; LAMMPS can read triclinic if given tilt.
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = cell
    xlo, xhi = 0.0, ax
    ylo, yhi = 0.0, by
    zlo, zhi = 0.0, cz
    xy, xz, yz = bx, cx, cy

    if "charges" in atoms.arrays:
        charges = atoms.arrays["charges"]
    else:
        charges = np.full(len(atoms), float(charge_default))

    with open(filename, "w") as f:
        f.write("LAMMPS data file (bulk MOF + CO2 + CH4)\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{len(ordered)} atom types\n\n")
        f.write(f"{xlo:16.8f} {xhi:16.8f} xlo xhi\n")
        f.write(f"{ylo:16.8f} {yhi:16.8f} ylo yhi\n")
        f.write(f"{zlo:16.8f} {zhi:16.8f} zlo zhi\n\n")
        f.write(f"{xy:16.8f} {xz:16.8f} {yz:16.8f} xy xz yz\n\n")

        f.write("Masses\n\n")
        for k, tid in sorted(type_map.items(), key=lambda kv: kv[1]):
            sym, gid = k
            mass = atomic_masses[atomic_numbers[sym]]
            label = {0: "MOF", 2: "CO2", 3: "CH4"}.get(gid, f"group{gid}")
            f.write(f"{tid:4d} {mass:16.8f}  # {sym} ({label})\n")
        f.write("\n")

        f.write("Atoms  # atom_style full\n\n")
        pos = atoms.get_positions()
        for i, (sym, gid, mid, q, r) in enumerate(zip(symbols, group_ids, mol_ids, charges, pos), start=1):
            tid = type_map[(sym, int(gid))]
            x, y, z = r
            f.write(f"{i:6d} {int(mid):6d} {tid:4d} {q:12.6f} "
                    f"{x:16.8f} {y:16.8f} {z:16.8f}\n")

# =========================
# Main
# =========================
def main():
    rng = np.random.default_rng(SEED)

    # 1) Read MOF and build supercell
    mof = read(MOF_FILE)
    mof = make_supercell(mof, SC_MATRIX)
    mof.set_pbc([True, True, True])

    cell = np.array(mof.get_cell())
    inv_cell = np.linalg.inv(cell)

    # label MOF
    mof.new_array("group_id", np.full(len(mof), 0, dtype=int))
    mof.new_array("mol_id", np.zeros(len(mof), dtype=int))

    system = mof.copy()

    # Existing positions (for overlap checks)
    existing_pos = system.get_positions()

    # templates
    co2_t = molecule("CO2")
    ch4_t = molecule("CH4")

    # 2) Insert CO2 molecules
    mol_id = 1
    for _ in range(N_CO2):
        placed = place_molecule_randomly(co2_t, cell, inv_cell, rng, existing_pos, MIN_DIST, MAX_TRIES)
        placed.set_array("group_id", np.full(len(placed), 2, dtype=int))
        placed.set_array("mol_id", np.full(len(placed), mol_id, dtype=int))
        system += placed
        existing_pos = system.get_positions()
        mol_id += 1

    # 3) Insert CH4 molecules
    for _ in range(N_CH4):
        placed = place_molecule_randomly(ch4_t, cell, inv_cell, rng, existing_pos, MIN_DIST, MAX_TRIES)
        placed.set_array("group_id", np.full(len(placed), 3, dtype=int))
        placed.set_array("mol_id", np.full(len(placed), mol_id, dtype=int))
        system += placed
        existing_pos = system.get_positions()
        mol_id += 1

    system.set_cell(cell)
    system.set_pbc([True, True, True])

    print(f"Final system atoms: {len(system)}")
    print(f"MOF atoms: {len(mof)}, CO2 molecules: {N_CO2}, CH4 molecules: {N_CH4}")

    # 4) Write outputs
    write(OUT_XYZ, system)
    print(f"Wrote: {OUT_XYZ}")

    # Optional LAMMPS data
    write_lammps_data_custom(OUTPUT_DATA, system, charge_default=0.0)
    print(f"Wrote: {OUTPUT_DATA}")


# =========================
# User settings
# =========================

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-file_path", type=str, required=True)
args = parser.parse_args()

file_path = args.file_path
file_base = file_path.split("/")[-1].split(".")[0]
#  MOF_FILE = f"{file_base}.extxyz"                 # your MOF unit cell
MOF_FILE = f"{file_base}.cif"                 # your MOF unit cell

SC_MATRIX = np.diag([1, 1, 2])

N_CO2 = 36
N_CH4 = 2

MIN_DIST = 2.3          # Å, min distance between any two atoms (MOF-gas and gas-gas)
MAX_TRIES = 30000
SEED = 42

OUT_XYZ = f"{file_base}_{N_CO2}CO2_{N_CH4}CH4.extxyz"
OUTPUT_DATA = f"data.{file_base}_{N_CO2}CO2_{N_CH4}CH4"

if __name__ == "__main__":
    main()

