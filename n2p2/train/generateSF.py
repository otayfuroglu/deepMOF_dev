
import sys
import getpass
USER = getpass.getuser()
BASE_DIR = f"/kernph/{USER}/deepMOF_dev"
sys.path.append(f"{BASE_DIR}/n2p2/python/symfunc_paramgen/src")
from sfparamgen import SymFuncParamGenerator



myGenerator = SymFuncParamGenerator( elements=['H', 'Li', 'Al'],
                                    r_cutoff = 6.0)

myGenerator.symfunc_type = 'radial'
myGenerator.generate_radial_params(rule='imbalzano2018',
                                   mode='center', nb_param_pairs=6)

f = open('sym_func_params.txt', 'w')
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)

zetas = [1.0, 2.0, 4.0, 16.0]
myGenerator.zetas = zetas
myGenerator.symfunc_type = 'angular_narrow'
myGenerator.generate_radial_params(rule='imbalzano2018',
                                   mode='center', nb_param_pairs=3)
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)

#  myGenerator.symfunc_type = 'angular_wide'
#  myGenerator.generate_radial_params(rule='imbalzano2018',
#                                     mode='center', nb_param_pairs=3)
#  myGenerator.write_settings_overview(fileobj=f)
#  myGenerator.write_parameter_strings(fileobj=f)
#


