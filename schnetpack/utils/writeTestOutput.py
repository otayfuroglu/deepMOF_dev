#
import os
import pandas as pd


def writeTestOutput(file_path, headers, test_outputs):
    assert len(headers) == len(test_outputs),\
            "Number of headers and outputs don't match !!"
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=headers)
    else:
        df = pd.read_csv(file_path, index_col=0)

    df.loc[len(df.index)] = test_outputs
    df.to_csv(file_path)

# tests
#  file_path = "test.csv"
#  headers = ["A", "B", "C"]
#  test_output = ["a", 1, 2]
#  writeTestOutput(file_path, headers, test_output)
