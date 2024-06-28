import os
import re
from io import StringIO
from pathlib import Path

import numpy as np

from ase.io import read
from ase.units import Bohr, Hartree
from ase.utils import reader, writer



@reader
def read_esp_charges(fd):
    """Read M-K esp charges from gaussian output file."""

    lines = fd.readlines()

    esp_charges = []
    start = False
    for i, line in enumerate(lines):
        if line.startswith(" ESP charges:"):
            start = True
            i_start = i
        if start:
            if line.startswith(" Sum of ESP charges"):
                if len(esp_charges) == 0:
                    raise RuntimeError('No charges')
                return np.array(esp_charges)
            if i >= (i_start + 2):
                if not len(line) == 1:
                    charge = float(line.split()[2])
                    esp_charges += [charge]



#  stdout_path = "/truba_scratch/otayfuroglu/deepMOF_dev/data_gen/works/mof74/testGaussian/run_engrad_co2/co2_00000/co2_00000.log"
#  fd = Path(stdout_path)
#  print(read_esp_charges(fd))
#  quit()

