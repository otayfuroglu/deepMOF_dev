
import sys
import getpass
USER = getpass.getuser()
BASE_DIR = f"/truba_scratch/{USER}/deepMOF_dev"
sys.path.append(f"{BASE_DIR}/n2p2/python/symfunc_paramgen/src")
from sfparamgen import SymFuncParamGenerator




myGenerator = SymFuncParamGenerator( elements=['H', 'C', 'O', 'Zn'],
                                    r_cutoff = 6.0)


myGenerator.symfunc_type = 'weighted_radial'
myGenerator.generate_radial_params(rule='gastegger2018',
                                   mode='shift', nb_param_pairs=15, r_lower=0.7)

f = open('sym_func_params.txt', 'w')
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)

zetas = [1, 6]
myGenerator.symfunc_type = 'weighted_angular'
myGenerator.zetas = zetas
myGenerator.generate_radial_params(rule='gastegger2018',
                                   mode='center', nb_param_pairs=3, r_lower=0.7)
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)

#  myGenerator.symfunc_type = 'angular_wide'
#  myGenerator.zetas = zetas
#  myGenerator.generate_radial_params(rule='imbalzano2018',
#                                     mode='center', nb_param_pairs=3, r_lower=1.5)
#  myGenerator.write_settings_overview(fileobj=f)
#  myGenerator.write_parameter_strings(fileobj=f)



