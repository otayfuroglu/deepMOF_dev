
import sys
import getpass
USER = getpass.getuser()
BASE_DIR = f"/truba_scratch/{USER}/deepMOF_dev"
sys.path.append(f"{BASE_DIR}/n2p2/python/symfunc_paramgen/src")
from sfparamgen import SymFuncParamGenerator




myGenerator = SymFuncParamGenerator( elements=['H', 'C', 'O', 'Zn'],
                                    r_cutoff = 6.0)


myGenerator.symfunc_type = 'radial'
myGenerator.generate_radial_params(rule='imbalzano2018',
                                   mode='shift', nb_param_pairs=5)

f = open('sym_func_params.txt', 'w')
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)


myGenerator.symfunc_type = 'angular_narrow'
myGenerator.zetas = [1.0, 4.0]
myGenerator.generate_radial_params(rule='gastegger2018',
                                   mode='center', nb_param_pairs=3, r_lower=1.5)
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)

myGenerator.symfunc_type = 'angular_wide'
myGenerator.zetas = [1.0, 4.0]
myGenerator.generate_radial_params(rule='gastegger2018',
                                   mode='center', nb_param_pairs=3, r_lower=1.5)
myGenerator.write_settings_overview(fileobj=f)
myGenerator.write_parameter_strings(fileobj=f)


