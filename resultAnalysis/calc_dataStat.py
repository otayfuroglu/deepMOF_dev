#
import pandas as pd
import numpy as np


def calc_error(xarray, yarray):
    """
    args:
        xarray: numpy array for x directin
        yarray: numpy array for y directin

    retruns: errors fo numpy array for xarray and yarray

    """
    return(xarray - yarray)

def calc_rmse(xarray, yarray):
    """
    args:
        xarray: numpy array for x directin
        yarray: numpy array for y directin

    retruns: root mean square error for xarray and yarray

    """
    N = len(xarray)
    return(np.sqrt(sum(calc_error(xarray, yarray)**2) / N))


if __name__ == "__main__":


    for mof_num in [1]:
        print("IR-MOF-%s" %mof_num)
        RESULT_DIR = "/home/omert/Desktop/deepMOF_local/deepMOF/HDNNP/schnetpack/results/IRMOF%s/batch24_alldata" % mof_num
        tresholds = {"E": 0.05, "F": 0.25, "FC": 0.5}
        for calc_type in ["test"]:
            print("%s:" %calc_type)
            labels = {"E": "energiesPerAtom", "F":"fmax", "FC": "fmax_component"}
            for key, value in labels.items():
                df_data = pd.read_csv("%s/qm_sch_SP_%s_%s.csv" %(RESULT_DIR, key, calc_type))
                treshold = tresholds[key]
                x = df_data["qm_SP_%s" %value].to_numpy()
                y = df_data["n2p2_SP_%s" %value].to_numpy()
                errors = calc_error(x, y)
                n_greater = errors[np.where(errors > treshold)].shape[0]
                percent_n_greater = 100 * n_greater / len(x)
                print("Prob %s: Nuber of values which is greather than %.3f --> %d (%.2f%%)"\
                      %(key, treshold, n_greater, percent_n_greater))

                print("Prob %s: RMSE --> %.3f" %(key, calc_rmse(x, y)))
                print("Prob %s: maxError --> %.3f" %(key, max(errors)))
                print()
        print("*" * 50)
        print()


            #  print(calc_rmse(x, y))
