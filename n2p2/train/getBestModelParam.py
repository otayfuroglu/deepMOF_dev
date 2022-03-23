#
import argparse
import os

def prepareModel():
    os.system(f"cp {model_dir}/input.nn {result_dir}")
    os.system(f"cp {model_dir}/scaling.data {result_dir}")
    weights_files = [item for item in os.listdir(model_dir) if "weights" in item]
    best_weights_files = [item for item in weights_files if int(item.split(".")[-2]) == best_epoch]
    assert len(best_weights_files) != 0, "Erro: NOT FOUND best epoch number"
    for best_weights_file in best_weights_files:
        print(f"Chosen weights file as best parameters --> ",best_weights_file)
        os.system(f"cp {model_dir}/{best_weights_file} {result_dir}/{best_weights_file[:11]}.data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give something ...")
    #  parser.add_argument("-mof_num", "--mof_num",
    #                      type=int, required=True,
                        #  help="..")
    parser.add_argument("-best_epoch", "--best_epoch",
                        type=int, required=True,
                        help="..")
    parser.add_argument("-model_dir", "--model_dir",
                        type=str, required=True,
                        help="..")
    args = parser.parse_args()
    best_epoch = args.best_epoch
    model_dir = args.model_dir
    result_dir = f"best_epoch_{best_epoch}"
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    prepareModel()

















