#!/usr/bin/env python3
"""
Compute CO2 and CH4 permeation flux and permeability (in Barrer)
from a LAMMPS dump (.lammpstrj) that does NOT contain 'mol' in ATOMS.

Steps
-----
1) Parse the dump.
2) From the first frame:
   - determine MOF max z -> define vacuum threshold z_vac
   - reconstruct CO2 / CH4 molecules from distance connectivity
   - estimate membrane thickness from MOF atoms: t_mof = z_max - z_min
3) For all frames:
   - track each molecule's COM z
   - mark molecules that ever go beyond z_vac
4) Compute flux:
   J = N_cross / (A * t)
5) Compute permeability (SI + Barrer):
   P_SI = J * l / Δp   (mol m / (m^2 s Pa))
   P_Barrer = P_SI / BARRER_SI

Usage:
    python compute_flux_and_permeability_nomol.py dump.nemd.lammpstrj
"""

import sys
import re
import math
import argparse
from collections import defaultdict, deque

# -----------------------------
# User settings: ADAPT THESE
# -----------------------------

# Atom types (from your Masses section)
# Example (adapt to your system!):
#   1–4 : MOF
#   5   : graphene
#   6–7 : CO2
#   8–9 : CH4

MOF_TYPES       = {1, 2, 3, 4}
GRAPHENE_TYPES  = {5}        # not strictly needed here
CO2_TYPES       = {6, 7}
CH4_TYPES       = {8, 9}
GAS_TYPES       = CO2_TYPES | CH4_TYPES

# Margin beyond MOF end where vacuum is considered to start (Å)
VAC_MARGIN = 2.0

# Skip early timesteps (optional: to ignore equilibration)
SKIP_STEPS_BEFORE_COUNT = 0   # e.g. 1_000_000

# Distance cutoff to build bonds for gas molecules (Å)
BOND_CUTOFF = 1.7

# --- Time control ---

# Option A: use time from timesteps and MD timestep (fs)
USE_TIMESTEPS = True
TIMESTEP_FS = 1.0   # your LAMMPS "timestep" in fs

# Option B: force a fixed total duration in ns
FORCED_DURATION_NS = 2.0   # used only if USE_TIMESTEPS = False

# --- Pressure difference for permeability ---

# Set your pressure drop across the membrane (feed - permeate).
# You can set either DELTA_P_BAR or DELTA_P_PA directly.
DELTA_P_BAR = 1.0         # example: 1 bar driving force
DELTA_P_PA  = DELTA_P_BAR * 1.0e5   # 1 bar = 1e5 Pa

# --- Barrer conversion ---

# 1 Barrer = 10^-10 cm^3(STP)*cm/(cm^2*s*cmHg)
#          = 3.35e-16 mol m/(m^2 s Pa) approximately
BARRER_SI = 3.35e-16      # mol m / (m^2 s Pa)


# -----------------------------
# Dump parser
# -----------------------------

def parse_dump(filename):
    """
    Generator over frames in a LAMMPS custom dump.

    Yields (timestep, box, atoms), where:
      - box = (xlo, xhi, ylo, yhi, zlo, zhi)
      - atoms = list of dicts with: 'id', 'type', 'x', 'y', 'z'

    Handles:
      - 'ITEM: TIMESTEP 0' or 'ITEM: TIMESTEP' + '0'
      - 'ITEM: NUMBER OF ATOMS2186' or next line with number
      - 'ITEM: BOX BOUNDS xy xz yz ...' with 3 floats per line
    """
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break  # EOF

            line = line.strip()
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            # TIMESTEP
            parts = line.split()
            if len(parts) >= 3:
                timestep = int(parts[-1])
            else:
                timestep = int(f.readline().strip())

            # NUMBER OF ATOMS
            line = f.readline().strip()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError("Expected 'ITEM: NUMBER OF ATOMS', got:\n" + line)
            m = re.search(r'(\d+)\s*$', line)
            if m:
                natoms = int(m.group(1))
            else:
                natoms = int(f.readline().strip())

            # BOX BOUNDS
            line = f.readline().strip()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError("Expected 'ITEM: BOX BOUNDS', got:\n" + line)

            line1 = f.readline().split()
            line2 = f.readline().split()
            line3 = f.readline().split()

            if len(line1) < 2 or len(line2) < 2 or len(line3) < 2:
                raise RuntimeError("Box bounds lines malformed.")

            xlo = float(line1[0]); xhi = float(line1[1])
            ylo = float(line2[0]); yhi = float(line2[1])
            zlo = float(line3[0]); zhi = float(line3[1])
            box = (xlo, xhi, ylo, yhi, zlo, zhi)

            # ATOMS header
            line = f.readline().strip()
            if not line.startswith("ITEM: ATOMS"):
                raise RuntimeError("Expected 'ITEM: ATOMS', got:\n" + line)

            header = line.split()[2:]
            try:
                idx_id = header.index("id")
                idx_typ = header.index("type")
                idx_x  = header.index("x")
                idx_y  = header.index("y")
                idx_z  = header.index("z")
            except ValueError as e:
                raise RuntimeError(
                    "Dump must contain at least id, type, x, y, z.\n"
                    f"Found header: {header}"
                ) from e

            atoms = []
            for _ in range(natoms):
                cols = f.readline().split()
                atoms.append({
                    "id":   int(cols[idx_id]),
                    "type": int(cols[idx_typ]),
                    "x":    float(cols[idx_x]),
                    "y":    float(cols[idx_y]),
                    "z":    float(cols[idx_z]),
                })

            yield timestep, box, atoms


# -----------------------------
# Molecule reconstruction
# -----------------------------

def classify_mol_from_types(types):
    s = set(types)
    if s & CO2_TYPES:
        return "CO2"
    if s & CH4_TYPES:
        return "CH4"
    return "OTHER"


def build_molecules_from_first_frame(atoms):
    """Reconstruct CO2 / CH4 molecules from first frame using distance connectivity."""
    gas_atoms = [a for a in atoms if a["type"] in GAS_TYPES]
    if not gas_atoms:
        raise RuntimeError("No gas atoms in first frame. Check CO2_TYPES/CH4_TYPES.")

    n = len(gas_atoms)
    xs = [a["x"] for a in gas_atoms]
    ys = [a["y"] for a in gas_atoms]
    zs = [a["z"] for a in gas_atoms]

    adj = [[] for _ in range(n)]
    cutoff2 = BOND_CUTOFF ** 2

    # Naive O(N^2) distance graph
    for i in range(n):
        xi, yi, zi = xs[i], ys[i], zs[i]
        for j in range(i + 1, n):
            dx = xs[j] - xi
            dy = ys[j] - yi
            dz = zs[j] - zi
            r2 = dx*dx + dy*dy + dz*dz
            if r2 < cutoff2:
                adj[i].append(j)
                adj[j].append(i)

    # Connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if visited[i]:
            continue
        q = deque([i])
        visited[i] = True
        comp = [i]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    q.append(v)
                    comp.append(v)
        components.append(comp)

    molecules = []
    for comp in components:
        atom_ids = [gas_atoms[i]["id"] for i in comp]
        types = [gas_atoms[i]["type"] for i in comp]
        kind = classify_mol_from_types(types)
        if kind in ("CO2", "CH4"):
            molecules.append({"kind": kind, "atom_ids": atom_ids})

    if not molecules:
        raise RuntimeError("No CO2/CH4 molecules reconstructed. "
                           "Check BOND_CUTOFF and type sets.")

    n_co2 = sum(1 for m in molecules if m["kind"] == "CO2")
    n_ch4 = sum(1 for m in molecules if m["kind"] == "CH4")
    print(f"Reconstructed {len(molecules)} gas molecules in first frame "
          f"({n_co2} CO2, {n_ch4} CH4).")
    return molecules


# -----------------------------
# Main analysis
# -----------------------------

def count_flux_and_permeability(filename):
    frames = parse_dump(filename)

    # First frame
    try:
        first_timestep, first_box, first_atoms = next(frames)
    except StopIteration:
        print("Empty trajectory.")
        return

    xlo, xhi, ylo, yhi, zlo, zhi = first_box
    Lx = xhi - xlo
    Ly = yhi - ylo
    area_A2 = Lx * Ly   # Å^2

    # MOF thickness from z spread of MOF atoms
    mof_z = [a["z"] for a in first_atoms if a["type"] in MOF_TYPES]
    if not mof_z:
        raise RuntimeError("No MOF atoms in first frame. Check MOF_TYPES.")
    z_mof_min = min(mof_z)
    z_mof_max = max(mof_z)
    t_mof_A = z_mof_max - z_mof_min   # Å
    t_mof_m = t_mof_A * 1e-10         # convert Å -> m

    z_vac = z_mof_max + VAC_MARGIN

    print(f"First timestep: {first_timestep}")
    print(f"Box: Lx = {Lx:.3f} Å, Ly = {Ly:.3f} Å -> A = {area_A2:.3f} Å^2")
    print(f"MOF z-range: [{z_mof_min:.3f}, {z_mof_max:.3f}] Å -> thickness ~ {t_mof_A:.3f} Å")
    print(f"Vacuum threshold z_vac = {z_vac:.3f} Å (MOF max + {VAC_MARGIN} Å)")
    print(f"Using Δp = {DELTA_P_PA:.3e} Pa for permeability.\n")

    # Reconstruct molecules
    molecules = build_molecules_from_first_frame(first_atoms)

    # Track permeation
    permeated_CO2 = set()
    permeated_CH4 = set()
    last_timestep = first_timestep

    def atom_map_by_id(atoms):
        return {a["id"]: a for a in atoms}

    def process_frame(timestep, atoms):
        nonlocal last_timestep
        last_timestep = timestep

        idmap = atom_map_by_id(atoms)
        for midx, mol in enumerate(molecules):
            kind = mol["kind"]
            if kind not in ("CO2", "CH4"):
                continue

            zs = []
            for aid in mol["atom_ids"]:
                a = idmap.get(aid)
                if a is None:
                    continue
                zs.append(a["z"])
            if not zs:
                continue

            z_com = sum(zs) / len(zs)
            if z_com > z_vac:
                if kind == "CO2":
                    permeated_CO2.add(midx)
                else:
                    permeated_CH4.add(midx)

    # process first frame
    if first_timestep >= SKIP_STEPS_BEFORE_COUNT:
        process_frame(first_timestep, first_atoms)

    # process remaining frames
    for timestep, box, atoms in frames:
        if timestep < SKIP_STEPS_BEFORE_COUNT:
            continue
        process_frame(timestep, atoms)

    # --- counts ---
    n_co2 = len(permeated_CO2)
    n_ch4 = len(permeated_CH4)

    print("=== Permeation counts ===")
    print(f"CO2 molecules reaching vacuum: {n_co2}")
    print(f"CH4 molecules reaching vacuum: {n_ch4}")

    # --- time ---
    if USE_TIMESTEPS:
        dt_fs = TIMESTEP_FS
        nsteps = last_timestep - first_timestep
        total_time_fs = nsteps * dt_fs
        total_time_s  = total_time_fs * 1e-15
        total_time_ns = total_time_s * 1e9
        print(f"\nTime from timesteps: {nsteps} steps × {dt_fs} fs "
              f"= {total_time_fs:.2f} fs = {total_time_ns:.3f} ns")
    else:
        total_time_ns = FORCED_DURATION_NS
        total_time_s  = total_time_ns * 1e-9
        print(f"\nUsing forced duration: {total_time_ns:.3f} ns")

    # --- area conversions ---
    area_nm2 = area_A2 * 0.01       # 1 Å^2 = 0.01 nm^2
    area_m2  = area_A2 * 1.0e-20    # 1 Å^2 = 1e-20 m^2

    # --- fluxes ---
    NA = 6.02214076e23

    # flux in molecules/(nm^2·ns)
    J_CO2_mol_nm2_ns = n_co2 / (area_nm2 * total_time_ns) if area_nm2 > 0 and total_time_ns > 0 else float("nan")
    J_CH4_mol_nm2_ns = n_ch4 / (area_nm2 * total_time_ns) if area_nm2 > 0 and total_time_ns > 0 else float("nan")

    # flux in mol/(m^2·s)
    J_CO2_mol_m2_s = (n_co2 / NA) / (area_m2 * total_time_s) if area_m2 > 0 and total_time_s > 0 else float("nan")
    J_CH4_mol_m2_s = (n_ch4 / NA) / (area_m2 * total_time_s) if area_m2 > 0 and total_time_s > 0 else float("nan")

    print("\n=== Flux ===")
    print("Units: molecules / (nm^2·ns)")
    print(f"J_CO2 = {J_CO2_mol_nm2_ns:.4e}  molecules/(nm^2·ns)")
    print(f"J_CH4 = {J_CH4_mol_nm2_ns:.4e}  molecules/(nm^2·ns)")
    print("\nUnits: mol / (m^2·s)")
    print(f"J_CO2 = {J_CO2_mol_m2_s:.4e}  mol/(m^2·s)")
    print(f"J_CH4 = {J_CH4_mol_m2_s:.4e}  mol/(m^2·s)")

    # --- permeability in SI and Barrer ---
    # P_SI = J * l / Δp  [mol m / (m^2 s Pa)]
    if DELTA_P_PA > 0.0:
        P_CO2_SI = J_CO2_mol_m2_s * t_mof_m / DELTA_P_PA
        P_CH4_SI = J_CH4_mol_m2_s * t_mof_m / DELTA_P_PA

        P_CO2_Barrer = P_CO2_SI / BARRER_SI
        P_CH4_Barrer = P_CH4_SI / BARRER_SI
    else:
        P_CO2_SI = P_CH4_SI = P_CO2_Barrer = P_CH4_Barrer = float("nan")

    print("\n=== Permeability ===")
    print("SI units: mol·m / (m^2·s·Pa)")
    print(f"P_CO2 = {P_CO2_SI:.4e}  mol·m/(m^2·s·Pa)")
    print(f"P_CH4 = {P_CH4_SI:.4e}  mol·m/(m^2·s·Pa)")

    print("\nBarrer (1 Barrer ≈ 3.35×10^-16 mol·m/(m^2·s·Pa))")
    print(f"P_CO2 = {P_CO2_Barrer:.4e}  Barrer")
    print(f"P_CH4 = {P_CH4_Barrer:.4e}  Barrer")


if __name__ == "__main__":

    # -----------------------------
    # User settings: ADAPT THESE
    # -----------------------------

    # Atom type sets (from your data file's Masses section)
    # Example:
    #   1–4 : MOF
    #   5   : graphene
    #   6–7 : CO2
    #   8–9 : CH4

    MOF_TYPES       = {1, 2, 3, 4}   # or {1,2,3,4,5} if you want to include graphene as MOF
    GRAPHENE_TYPES  = {5}            # not used directly, but OK to keep
    CO2_TYPES       = {6, 7}
    CH4_TYPES       = {8, 9}
    GAS_TYPES       = CO2_TYPES | CH4_TYPES

    # Margin beyond MOF end where vacuum is considered to start (Å)
    VAC_MARGIN = 2.0

    # If you want to skip early timesteps (equilibration), set this:
    SKIP_STEPS_BEFORE_COUNT = 1000   # e.g. 100000

    # Distance cutoff to decide bonds inside a gas molecule (Å)
    # CO2 C–O ~1.16 Å, CH4 C–H ~1.09 Å, so 1.7 Å is safe.
    BOND_CUTOFF = 1.7

    # --- Time control ---

    # Option A: use time from timesteps and your MD timestep (fs)
    USE_TIMESTEPS = True      # set False to force a fixed duration

    # MD timestep in fs (metal units: 1 fs => timestep 0.001 ps)
    TIMESTEP_FS = 0.5         # adjust if your dt is different

    # Option B: force a fixed total duration in ns (e.g. exactly 2 ns)
    FORCED_DURATION_NS = 0.05  # used only if USE_TIMESTEPS = False

    parser = argparse.ArgumentParser(description="Give something ...")
    parser.add_argument("-trj_file", type=str, required=True)
    args = parser.parse_args()

    dump_file = args.trj_file
    #  count_permeated(dump_file)

    count_flux_and_permeability(dump_file)
