#
with open("sym_func.txt", "w") as fl:

    r_cutoff = 12.0

    # radial part
    fl.write("# radial symetry functions\n")
    sym_func_type = 2
    r_shift = 0.0
    element_pairs_ethas = {
        "H-H":[0.000, 0.007, 0.020, 0.053, 0.178],
        "H-C":[0.000, 0.007, 0.019, 0.052, 0.173],
        "H-O":[0.000, 0.007, 0.020, 0.055, 0.193],
        "H-Zn":[0.000, 0.006, 0.015, 0.034, 0.082],
        "C-C":[0.000, 0.006, 0.015, 0.034, 0.086],
        "C-O":[0.000, 0.006, 0.016, 0.038, 0.099],
        "C-Zn":[0.000, 0.003, 0.008, 0.014, 0.024],
        "O-O":[0.000, 0.004, 0.008, 0.015, 0.025],
        "O-Zn":[0.000, 0.004, 0.010, 0.020, 0.040],
        "Zn-Zn":[0.000, 0.002, 0.003, 0.005, 0.006],
    }

    for element_pair, ethas in element_pairs_ethas.items():
        fl.write("# " + element_pair + "\n")
        elements = element_pair.split("-")
        for etha in ethas:
            fl.write('symfunction_short {:<4}{:<3}{:<7}{:<15.6f}{:<15.6f}{:<15.6f}\n'.format(
                elements[0], sym_func_type, elements[1], etha, r_shift, r_cutoff))
        fl.write("\n")

    # angular part
    fl.write("# Angular symetry functions\n")
    sym_func_type = 3
    etha = 0.0
    element_pairs_lamdas_zetas = {
        "H-H-H":[[-1, 1], [1, 2, 4, 16]],
        "H-H-C":[[-1, 1], [1, 2, 4, 16]],
        "H-H-O":[[-1, 1], [1, 2, 4, 16]],
        "H-H-Zn":[[-1, 1], [1, 2, 4, 16]],
        "C-H-C":[[-1, 1], [1, 2, 4, 16]],
        "C-H-O":[[-1, 1 ], [1, 2, 4, 16]],
        "C-H-Zn":[[-1, 1], [1, 2, 4, 16]],
        "O-H-O":[[-1, 1], [1, 2, 4, 16]],
        "O-H-Zn":[[-1, 1], [1, 2, 4, 16]],
        "Zn-H-Zn-1":[[1], [1, 2, 4, 16]],
        "Zn-H-Zn-2":[[-1], [1, 2, 4]],
        "H-C-H":[[-1, 1], [1, 2, 4, 16]],
        "H-C-C":[[-1, 1], [1, 2, 4, 16]],
        "H-C-O":[[-1, 1], [1, 2, 4, 16]],
        "H-C-Zn":[[-1, 1], [1, 2, 4, 16]],
        "C-C-C":[[-1, 1], [1, 2, 4, 16]],
        "C-C-O":[[-1, 1], [1, 2, 4, 16]],
        "C-C-Zn":[[-1, 1], [1, 2, 4, 16]],
        "O-C-O":[[-1, 1], [1, 2, 4, 16]],
        "O-C-Zn":[[-1, 1], [1, 2, 4, 16]],
        "Zn-C-Zn-1":[[1], [1, 2, 4, 16]],
        "Zn-C-Zn-2":[[ -1], [1, 2, 4]],
        "H-O-H":[[-1, 1], [1, 2, 4, 16]],
        "H-O-C":[[-1, 1], [1, 2, 4, 16]],
        "H-O-O":[[-1, 1], [1, 2, 4, 16]],
        "H-O-Zn":[[-1, 1], [1, 2, 4, 16]],
        "C-O-C":[[-1, 1], [1, 2, 4, 16]],
        "C-O-O":[[-1, 1], [1, 2, 4, 16]],
        "C-O-Zn":[[-1, 1], [1, 2, 4, 16]],
        "O-O-O":[[-1, 1], [1, 2, 4, 16]],
        "O-O-Zn":[[-1, 1], [1, 2, 4, 16]],
        "Zn-O-Zn":[[-1, 1], [1, 2, 4, 16]],
        "H-Zn-H":[[-1, 1], [1, 2, 4, 16]],
        "H-Zn-C":[[-1, 1], [1, 2, 4, 16]],
        "H-Zn-O":[[-1, 1], [1, 2, 4, 16]],
        "H-Zn-Zn":[[-1, 1], [1, 2, 4, 16]],
        "C-Zn-C":[[-1, 1], [1, 2, 4, 16]],
        "C-Zn-O":[[-1, 1], [1, 2, 4, 16]],
        "C-Zn-Zn":[[-1, 1], [1, 2, 4, 16]],
        "O-Zn-O":[[-1, 1], [1, 2, 4, 16]],
        "O-Zn-Zn":[[-1, 1], [1, 2, 4, 16]],
        "Zn-Zn-Zn-1":[[1], [1, 2, 4, 16]],
        "Zn-Zn-Zn-2":[[-1], [1, 2, 4]],
    }

    passed_sets = []
    for element_pair, lamdas_zetas in element_pairs_lamdas_zetas.items():
        elements = element_pair.split("-")
        lamdas = lamdas_zetas[0]
        zetas = lamdas_zetas[1]

        if set(elements[1:]) in passed_sets:
            continue
        passed_sets.append(set(elements[1:]))
        fl.write("# " + element_pair + "\n")

        for lamda in lamdas:
            for zeta in zetas:
                fl.write('symfunction_short {:<4}{:<3}{:<3}{:<7}{:<10.4f}{:<10.1f}{:<10.1f}{:<15.6f}\n'.format(
                    elements[0], sym_func_type, elements[1], elements[2], etha, lamda, zeta, r_cutoff))
            fl.write("\n")

