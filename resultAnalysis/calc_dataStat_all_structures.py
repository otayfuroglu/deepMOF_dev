#! /home/omert/miniconda3/bin/python
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
    return np.abs(xarray - yarray)

def calc_rmse(xarray, yarray):
    """
    args:
        xarray: numpy array for x directin
        yarray: numpy array for y directin

    retruns: root mean square error for xarray and yarray

    """
    return(np.sqrt((calc_error(xarray, yarray)**2).mean()))


if __name__ == "__main__":

    BASE_DIR = "/home/omert/Desktop/deepMOF_dev"
    model_type = "results_best_epoch_46"
    nn_type = "n2p2"
    #  IDX_MOFs = [1, 4, 6, 7, 10]

    #  print("IR-MOF-%s" %mof_num)
    #  RESULT_DIR = "/home/omert/Desktop/deepMOF/deepMOF/HDNNP/schnetpack/results/IRMOF%s" % mof_num
    #  tresholds = {"E": 0.05, "F": 0.25, "FC": 0.5}
    tresholds = {"E": 0.005, "FC": 0.5}
    for val_type in ["train", "test"]:
        print("%s:" %val_type)
        #  labels = {"E": "energiesPerAtom", "F":"fmax", "FC": "fmax_component"}
        labels = {"E": "energiesPerAtom", "FC": "fmax_component"}
        for key, value in labels.items():
            #  dfs = []
            #  for single_mof_idx in IDX_MOFs:
                #  RESULT_DIR = "%s/schnetpack/results/IRMOF%s/%s" % (BASE_DIR, single_mof_idx, model_type)
            RESULT_DIR = "%s/%s/works/runTest/%s" % (BASE_DIR, nn_type, model_type)
            prop = key
            csv_file_name = "%s/qm_sch_SP_%s_%s.csv" %(RESULT_DIR, prop, val_type)
            #  column_names = ["FileNames", "qm_SP_energies", "schnet_SP_energies", "Error", "ErrorPerAtom"]
            try:
                df_data = pd.read_csv(csv_file_name)
            except:
                print("There is not %s data" % val_type)
                continue
                #  dfs.append(df_data)

            #  df_data = pd.concat(dfs)
            treshold = tresholds[key]
            x = df_data["qm_SP_%s" %value].to_numpy()
            y = df_data["%s_SP_%s" %(nn_type, value)].to_numpy()
            errors = calc_error(x, y)
            n_greater = errors[np.where(errors > treshold)].shape[0]
            percent_n_greater = 100 * n_greater / len(x)
            print("Total numer of data points: %d" %len(x))
            print("Prob %s: Nuber of values which is greather than %.4f --> %d (%.2f%%)"\
                  %(key, treshold, n_greater, percent_n_greater))

            print("Prob %s: RMSE --> %.4f" %(key, calc_rmse(x, y)))
            print("Prob %s: maxError --> %.4f" %(key, max(errors)))
            print()
    print("*" * 50)
    print()


            #  print(calc_rmse(x, y))
