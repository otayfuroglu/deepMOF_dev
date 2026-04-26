#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_lammps_msd_block(fname, dt_fs=1.0, drop_zero_rows=False, eps=1e-15):
    """
    Parse LAMMPS fix ave/time mode vector output from compute msd/chunk.

    Expected format:
      step nrows
      row_id msdx msdy msdz msd_xyz

    Returns:
      time_ps, msd_xyz, msd_x, msd_y, msd_z, nsamples
    """
    steps = []
    msd_x = []
    msd_y = []
    msd_z = []
    msd_xyz = []
    nsamples = []

    with open(fname, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) != 2:
            continue

        step = int(float(parts[0]))
        nrows = int(float(parts[1]))

        vals = []
        for _ in range(nrows):
            if i >= len(lines):
                break

            row = lines[i].split()
            i += 1

            if len(row) < 5:
                continue

            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            xyz = float(row[4])

            if drop_zero_rows and abs(xyz) <= eps and abs(x) <= eps and abs(y) <= eps and abs(z) <= eps:
                continue

            vals.append((x, y, z, xyz))

        if len(vals) == 0:
            msd_x.append(0.0)
            msd_y.append(0.0)
            msd_z.append(0.0)
            msd_xyz.append(0.0)
            nsamples.append(0)
        else:
            arr = np.array(vals, dtype=float)
            msd_x.append(arr[:, 0].mean())
            msd_y.append(arr[:, 1].mean())
            msd_z.append(arr[:, 2].mean())
            msd_xyz.append(arr[:, 3].mean())
            nsamples.append(arr.shape[0])

        steps.append(step)

    steps = np.array(steps, dtype=float)
    time_ps = steps * dt_fs * 1e-3

    return (
        time_ps,
        np.array(msd_xyz),
        np.array(msd_x),
        np.array(msd_y),
        np.array(msd_z),
        np.array(nsamples),
    )


def write_raspa_like(outname, time_ps, msd_xyz, msd_x, msd_y, msd_z, nsamples):
    with open(outname, "w") as f:
        f.write("# column 1: time [ps]\n")
        f.write("# column 2: msd xyz [A^2]\n")
        f.write("# column 3: msd x [A^2]\n")
        f.write("# column 4: msd y [A^2]\n")
        f.write("# column 5: msd z [A^2]\n")
        f.write("# column 6: number of samples [-]\n")

        for t, xyz, x, y, z, n in zip(time_ps, msd_xyz, msd_x, msd_y, msd_z, nsamples):
            f.write(f"{t:.6f} {xyz:.8e} {x:.8e} {y:.8e} {z:.8e} {int(n)}\n")


def plot_msd(outpng, time_ps, msd_xyz, msd_x, msd_y, msd_z):
    plt.figure(figsize=(7, 5))

    plt.plot(time_ps, msd_xyz, label="MSD xyz", lw=2)
    plt.plot(time_ps, msd_x, label="MSD x", lw=1.5, alpha=0.8)
    plt.plot(time_ps, msd_y, label="MSD y", lw=1.5, alpha=0.8)
    plt.plot(time_ps, msd_z, label="MSD z", lw=1.5, alpha=0.8)

    plt.xlabel("Time (ps)")
    plt.ylabel(r"MSD ($\mathrm{\AA^2}$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=300)
    print(f"Wrote plot: {outpng}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("msd_file", help="LAMMPS MSD file, e.g. msd_gas_ch4.data")
    parser.add_argument("--dt-fs", type=float, default=1.0,
                        help="LAMMPS timestep in fs. Default: 1.0")
    parser.add_argument("--drop-zero-rows", action="store_true",
                        help="Drop all-zero rows before averaging. Useful for minority gas with global chunk IDs.")
    parser.add_argument("--out-prefix", default="msd_plot",
                        help="Output prefix")
    args = parser.parse_args()

    time_ps, msd_xyz, msd_x, msd_y, msd_z, nsamples = parse_lammps_msd_block(
        args.msd_file,
        dt_fs=args.dt_fs,
        drop_zero_rows=args.drop_zero_rows,
    )

    outdat = f"{args.out_prefix}.dat"
    outpng = f"{args.out_prefix}.png"

    write_raspa_like(outdat, time_ps, msd_xyz, msd_x, msd_y, msd_z, nsamples)
    plot_msd(outpng, time_ps, msd_xyz, msd_x, msd_y, msd_z)

    print(f"Wrote data: {outdat}")
    print(f"Number of time points: {len(time_ps)}")
    print(f"Time range: {time_ps.min():.6f} to {time_ps.max():.6f} ps")


if __name__ == "__main__":
    main()
