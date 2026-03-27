#
"""
Count CO2 and CH4 molecules per frame from a LAMMPS custom dump (lammpstrj).

Assumptions:
- Dump was written with at least: id, type, mol (order can be arbitrary),
  e.g.:  dump ... custom ... id mol type x y z
- CO2 and CH4 are distinguished by atom *types* that you know from the
  LAMMPS data/Masses section.
- We count molecules by unique mol-ID whose atoms have those types.

Output:
- A text file with columns:
    timestep   n_CO2   n_CH4

Edit the USER PARAMETERS section below for your system.
"""

import sys, argparse


# -------------------------
# PARSER
# -------------------------

def parse_dump(filename):
    """
    Generator over frames in a LAMMPS custom dump.

    Yields:
        timestep (int),
        header (list of column names),
        data (list of lists of strings for each atom line)
    """
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            # Expect ITEM: TIMESTEP
            if not line.startswith("ITEM: TIMESTEP"):
                # skip until we find the next timestep
                continue

            ts_line = f.readline()
            if not ts_line:
                break
            timestep = int(ts_line.strip())

            # ITEM: NUMBER OF ATOMS
            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Unexpected format: expected 'ITEM: NUMBER OF ATOMS'")
            n_line = f.readline()
            n_atoms = int(n_line.strip())

            # ITEM: BOX BOUNDS ...
            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Unexpected format: expected 'ITEM: BOX BOUNDS'")
            # skip 3 lines of box bounds
            for _ in range(3):
                if not f.readline():
                    raise RuntimeError("Unexpected EOF in box bounds")

            # ITEM: ATOMS ...
            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError("Unexpected format: expected 'ITEM: ATOMS'")
            header = line.strip().split()[2:]  # after "ITEM: ATOMS"

            data = []
            for _ in range(n_atoms):
                atom_line = f.readline()
                if not atom_line:
                    raise RuntimeError("Unexpected EOF while reading atom data")
                data.append(atom_line.strip().split())

            yield timestep, header, data


def count_species_per_frame(header, data,
                            co2_type_ids, ch4_type_ids,
                            use_mol=True,
                            n_atoms_co2=3, n_atoms_ch4=5):
    """
    Given one frame (header + data), count CO2 and CH4 molecules.

    Returns:
        n_co2, n_ch4
    """
    # Find column indices
    try:
        type_idx = header.index("type")
    except ValueError:
        raise RuntimeError("No 'type' column in ATOMS header")

    if use_mol:
        try:
            mol_idx = header.index("mol")
        except ValueError:
            raise RuntimeError("USE_MOL_ID=True but no 'mol' column in ATOMS header")

        co2_mols = set()
        ch4_mols = set()

        for row in data:
            atype = int(row[type_idx])
            mol_id = int(row[mol_idx])

            if atype in co2_type_ids:
                co2_mols.add(mol_id)
            elif atype in ch4_type_ids:
                ch4_mols.add(mol_id)

        return len(co2_mols), len(ch4_mols)

    else:
        # Fallback: count atoms per species and divide by n_atoms_per_molecule
        n_co2_atoms = 0
        n_ch4_atoms = 0

        for row in data:
            atype = int(row[type_idx])
            if atype in co2_type_ids:
                n_co2_atoms += 1
            elif atype in ch4_type_ids:
                n_ch4_atoms += 1

        n_co2 = n_co2_atoms / float(n_atoms_co2)
        n_ch4 = n_ch4_atoms / float(n_atoms_ch4)
        return n_co2, n_ch4


def main():
    print(f"Reading dump file: {DUMP_FILE}")
    out = open(OUTPUT_FILE, "w")
    out.write("# timestep  n_CO2  n_CH4\n")

    n_frames = 0
    for ts, header, data in parse_dump(DUMP_FILE):
        n_frames += 1
        n_co2, n_ch4 = count_species_per_frame(
            header, data,
            CO2_TYPE_IDS, CH4_TYPE_IDS,
            use_mol=USE_MOL_ID,
            n_atoms_co2=N_ATOMS_CO2,
            n_atoms_ch4=N_ATOMS_CH4,
        )
        out.write(f"{ts:10d}  {n_co2:10.3f}  {n_ch4:10.3f}\n")

    out.close()
    print(f"Wrote counts for {n_frames} frames to {OUTPUT_FILE}")

parser = argparse.ArgumentParser(description="Give something ...")
parser.add_argument("-trj_file", type=str, required=True)
args = parser.parse_args()

DUMP_FILE = args.trj_file
OUTPUT_FILE = "counts_co2_ch4.dat"

# Set these according to your Masses section (type IDs for CO2 and CH4 atoms)
# Example (adjust to your actual types):
#   MOF:      1..4
#   graphene: 5
#   CO2:      6 (C), 7 (O)
#   CH4:      8 (C), 9 (H)
CO2_TYPE_IDS = {6, 7}   # <<< EDIT
CH4_TYPE_IDS = {8, 9}   # <<< EDIT

# If your dump does *not* have 'mol', we can approximate by atom count
# and dividing by number of atoms per molecule:
USE_MOL_ID = True   # set to False if you have no mol column
N_ATOMS_CO2 = 3     # C + 2O
N_ATOMS_CH4 = 5     # C + 4H


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)

