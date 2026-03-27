#!/usr/bin/env python3
"""
Build MOF permeability system with a generic binary gas mixture + graphene cap +
custom LAMMPS data writer.

Geometry along MOF c-vector (transport direction):

    [ graphene ] [ gas region ] [ MOF slab ] [ vacuum ]

- PBC: True, True, False
- Works for non-orthogonal MOF cells.
- Gas species are fully generalized through GAS_SPECS.

group_id convention:
    0 -> MOF
    1 -> graphene
    >=2 -> gas species in the order given in GAS_SPECS
"""

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell, molecule
from ase.data import atomic_numbers, atomic_masses


# ---------------------------
# Geometry helpers
# ---------------------------

def get_c_info(cell):
    cell = np.array(cell)
    c_vec = cell[2]
    L_c = np.linalg.norm(c_vec)
    c_hat = c_vec / L_c
    return L_c, c_hat


def cart_to_frac(positions, cell, inv_cell=None):
    cell = np.array(cell)
    if inv_cell is None:
        inv_cell = np.linalg.inv(cell)
    return positions @ inv_cell


def frac_to_cart(frac, cell):
    cell = np.array(cell)
    return frac @ cell


# ---------------------------
# Generic binary gas mixture generator
# ---------------------------

def build_random_binary_mixture_in_tube(
    cell,
    w_min,
    w_max,
    gas_specs,
    min_dist=2.5,
    max_tries=20000,
    seed=42,
    existing_atoms=None,
    start_mol_id=1,
):
    """
    Build a generic binary gas mixture in [w_min, w_max] along c.

    gas_specs: list of dicts, e.g.
        [
            {"name": "CO2", "ase_name": "CO2", "count": 20, "group_id": 2},
            {"name": "CH4", "ase_name": "CH4", "count": 180, "group_id": 3},
        ]

    Returns:
        gas (Atoms), next_mol_id
    """
    if len(gas_specs) != 2:
        raise ValueError("This function expects exactly 2 gas species.")

    rng = np.random.default_rng(seed)
    cell = np.array(cell)
    inv_cell = np.linalg.inv(cell)
    _, c_hat = get_c_info(cell)

    gas = Atoms(cell=cell, pbc=[True, True, False])

    mol_templates = {}
    for spec in gas_specs:
        mol_templates[spec["name"]] = molecule(spec["ase_name"])

    def random_orientation(mol_templ):
        mol = mol_templ.copy()
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(0.0, 360.0)
        mol.rotate(angle, v=axis, center="COP", rotate_cell=False)
        return mol

    def try_place(mol_templ):
        if existing_atoms is not None:
            if len(gas) > 0:
                existing_pos = np.vstack([gas.get_positions(), existing_atoms.get_positions()])
            else:
                existing_pos = existing_atoms.get_positions()
        else:
            existing_pos = gas.get_positions()

        existing_frac = cart_to_frac(existing_pos, cell, inv_cell) if len(existing_pos) > 0 else None

        for _ in range(max_tries):
            mol = random_orientation(mol_templ)

            u = rng.uniform(0.0, 1.0)
            v = rng.uniform(0.0, 1.0)
            w_c = rng.uniform(w_min, w_max)

            a_vec, b_vec, _ = cell
            r_xy = u * a_vec + v * b_vec
            r_c = c_hat * w_c
            r_com_target = r_xy + r_c

            r_com_current = mol.get_center_of_mass()
            shift = r_com_target - r_com_current
            mol.translate(shift)

            if existing_frac is None and len(gas) == 0:
                return mol

            new_pos = mol.get_positions()
            new_frac = cart_to_frac(new_pos, cell, inv_cell)

            ok = True
            for k in range(len(new_pos)):
                f_new = new_frac[k]
                if existing_frac is not None:
                    diff = existing_frac - f_new
                    diff[:, 0] -= np.rint(diff[:, 0])
                    diff[:, 1] -= np.rint(diff[:, 1])
                    diff_cart = frac_to_cart(diff, cell)
                    dist2 = np.sum(diff_cart**2, axis=1)
                    if np.any(dist2 < min_dist**2):
                        ok = False
                        break
            if ok:
                return mol

        raise RuntimeError("Could not place molecule without overlap; adjust density / gaps / min_dist.")

    next_mol_id = start_mol_id

    for spec in gas_specs:
        mol_templ = mol_templates[spec["name"]]
        for _ in range(spec["count"]):
            mol = try_place(mol_templ)
            mol_group = np.full(len(mol), spec["group_id"], dtype=int)
            mol_id_arr = np.full(len(mol), next_mol_id, dtype=int)
            mol.set_array("group_id", mol_group)
            mol.set_array("mol_id", mol_id_arr)
            gas += mol
            next_mol_id += 1

    return gas, next_mol_id


# ---------------------------
# Graphene sheet builder
# ---------------------------

def build_graphene_sheet_in_cross_section(cell, c_hat, w_layer, na=8, nb=8):
    cell = np.array(cell)
    frac = []
    for i in range(na):
        for j in range(nb):
            u = (i + 0.5) / na
            v = (j + 0.5) / nb
            frac.append([u, v, 0.0])
    frac = np.array(frac)

    pos = frac_to_cart(frac, cell)
    for k in range(len(pos)):
        w = np.dot(pos[k], c_hat)
        pos[k] += c_hat * (w_layer - w)

    graph = Atoms(
        symbols=["C"] * len(pos),
        positions=pos,
        cell=cell,
        pbc=[True, True, False],
    )

    n = len(graph)
    graph.new_array("group_id", np.full(n, 1, dtype=int))
    graph.new_array("mol_id", np.zeros(n, dtype=int))
    return graph


# ---------------------------
# Generic LAMMPS data writer
# ---------------------------

def write_lammps_data_custom(filename, atoms, gas_specs, charge_default=0.0):
    """
    Ordering:
        1) MOF (group 0), by mass desc
        2) graphene (group 1), by mass desc
        3) gas species in GAS_SPECS order, with element order from each spec
        4) anything else
    """
    gas_order_map = {spec["group_id"]: spec for spec in gas_specs}

    cell = np.array(atoms.get_cell())
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = cell

    xlo = 0.0
    xhi = ax
    xy = bx
    xz = cx
    ylo = 0.0
    yhi = by
    zlo = 0.0
    zhi = cz
    yz = cy

    symbols = atoms.get_chemical_symbols()
    group_ids = atoms.arrays["group_id"]
    mol_ids = atoms.arrays.get("mol_id", np.zeros(len(atoms), int))

    type_keys = []
    for sym, gid in zip(symbols, group_ids):
        key = (sym, int(gid))
        if key not in type_keys:
            type_keys.append(key)

    def elem_mass(sym):
        return atomic_masses[atomic_numbers[sym]]

    mof_keys = [k for k in type_keys if k[1] == 0]
    graphene_keys = [k for k in type_keys if k[1] == 1]

    mof_keys = sorted(mof_keys, key=lambda k: -elem_mass(k[0]))
    graphene_keys = sorted(graphene_keys, key=lambda k: -elem_mass(k[0]))

    ordered_type_keys = mof_keys + graphene_keys

    used = set(ordered_type_keys)

    for spec in gas_specs:
        gid = spec["group_id"]
        atom_order = spec["atom_order"]
        order_map = {sym: i for i, sym in enumerate(atom_order)}
        gas_keys = [k for k in type_keys if k[1] == gid]
        gas_keys = sorted(gas_keys, key=lambda k: order_map.get(k[0], 999))
        ordered_type_keys.extend(gas_keys)
        used.update(gas_keys)

    other_keys = [k for k in type_keys if k not in used]
    other_keys = sorted(other_keys, key=lambda k: (k[1], k[0]))
    ordered_type_keys.extend(other_keys)

    type_map = {key: i + 1 for i, key in enumerate(ordered_type_keys)}

    masses = {}
    for key, tid in type_map.items():
        sym, gid = key
        masses[tid] = atomic_masses[atomic_numbers[sym]]

    if "charges" in atoms.arrays:
        charges = atoms.arrays["charges"]
    else:
        charges = np.full(len(atoms), float(charge_default))

    with open(filename, "w") as f:
        f.write("LAMMPS data file (generic binary gas + MOF + graphene)\n\n")

        natoms = len(atoms)
        ntypes = len(ordered_type_keys)

        f.write(f"{natoms} atoms\n")
        f.write(f"{ntypes} atom types\n\n")

        f.write(f"{xlo:16.8f} {xhi:16.8f} xlo xhi\n")
        f.write(f"{ylo:16.8f} {yhi:16.8f} ylo yhi\n")
        f.write(f"{zlo:16.8f} {zhi:16.8f} zlo zhi\n\n")
        f.write(f"{xy:16.8f} {xz:16.8f} {yz:16.8f} xy xz yz\n\n")

        f.write("Masses\n\n")
        for key, tid in sorted(type_map.items(), key=lambda kv: kv[1]):
            sym, gid = key
            mass = masses[tid]
            if gid == 0:
                label = "MOF"
            elif gid == 1:
                label = "graphene"
            elif gid in gas_order_map:
                label = gas_order_map[gid]["name"]
            else:
                label = f"group{gid}"
            f.write(f"{tid:4d} {mass:16.8f}  # {sym} ({label})\n")
        f.write("\n")

        f.write("Atoms  # atom_style full\n\n")
        pos = atoms.get_positions()
        for i, (sym, gid, mid, q, r) in enumerate(zip(symbols, group_ids, mol_ids, charges, pos), start=1):
            atype = type_map[(sym, int(gid))]
            x, y, z = r
            f.write(
                f"{i:6d} {int(mid):6d} {atype:4d} {q:12.6f} "
                f"{x:16.8f} {y:16.8f} {z:16.8f}\n"
            )


# ---------------------------
# Main construction
# ---------------------------

def main():
    mof_unit = read(MOF_FILE)
    mof_super = make_supercell(mof_unit, SC_MATRIX)

    cell_mof = np.array(mof_super.get_cell())
    L_MOF, c_hat = get_c_info(cell_mof)

    graphene_w = 0.0
    gas_w_min = graphene_w + GAP_GRAPHENE_GAS
    gas_w_max = gas_w_min + L_GAS
    mof_w_min = gas_w_max + GAP_GAS_MOF
    mof_w_max = mof_w_min + L_MOF
    Lz_total = mof_w_max + L_VAC

    scale_c = Lz_total / L_MOF
    new_cell = cell_mof.copy()
    new_cell[2] *= scale_c

    mof_super.set_cell(new_cell, scale_atoms=False)
    mof_super.set_pbc([True, True, False])

    n_mof = len(mof_super)
    mof_super.new_array("group_id", np.full(n_mof, 0, dtype=int))
    mof_super.new_array("mol_id", np.zeros(n_mof, dtype=int))
    mof_super.translate(c_hat * mof_w_min)

    system = Atoms(cell=new_cell, pbc=[True, True, False])
    system += mof_super

    graph = build_graphene_sheet_in_cross_section(new_cell, c_hat, graphene_w)
    system += graph

    print("Building gas mixture:")
    for spec in GAS_SPECS:
        print(f"  {spec['name']}: {spec['count']} molecules")

    gas, next_mol_id = build_random_binary_mixture_in_tube(
        cell=new_cell,
        w_min=gas_w_min,
        w_max=gas_w_max,
        gas_specs=GAS_SPECS,
        min_dist=MIN_DIST,
        max_tries=MAX_TRIES,
        seed=RNG_SEED,
        existing_atoms=system,
        start_mol_id=1,
    )
    system += gas

    suffix = "_".join(f"{spec['count']}{spec['name']}" for spec in GAS_SPECS)

    write(f"{file_base}_{suffix}.extxyz", system)
    write_lammps_data_custom(f"data.{file_base}_{suffix}", system, GAS_SPECS, charge_default=0.0)

    print(f"Final system: {len(system)} atoms")
    print(f"Wrote: {file_base}_{suffix}.extxyz")
    print(f"Wrote: data.{file_base}_{suffix}")


# ---------------------------
# User parameters
# ---------------------------

file_base = "MgMOF74_clean_fromCORE_1x1x8_corrected"
MOF_FILE = f"{file_base}.extxyz"

L_GAS = 30.0
L_VAC = 100.0
GAP_GAS_MOF = 2.0
GAP_GRAPHENE_GAS = 2.0

SC_MATRIX = np.diag([1, 1, 1])

# Generic binary gas definition
# ase_name must be recognized by ase.build.molecule()
# atom_order controls type ordering in the LAMMPS writer
GAS_SPECS = [
    {
        "name": "CO2",
        "ase_name": "CO2",
        "count": 100,
        "group_id": 2,
        "atom_order": ["C", "O"],
    },
    #  {
    #      "name": "CH4",
    #      "ase_name": "CH4",
    #      "count": 180,
    #      "group_id": 3,
    #      "atom_order": ["C", "H"],
    #  },
    {
        "name": "H2",
        "ase_name": "H2",
        "count": 100,
        "group_id": 3,
        "atom_order": ["H"],
    },
]

MIN_DIST = 2.5
MAX_TRIES = 20000
RNG_SEED = 33

if __name__ == "__main__":
    main()
