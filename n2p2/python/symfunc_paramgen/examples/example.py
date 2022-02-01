
import sys
sys.path.append('../src')

from sfparamgen import SymFuncParamGenerator

myGenerator = SymFuncParamGenerator(elements=['H', 'C', 'O', 'Zn'], r_cutoff = 6.)


#  myGenerator.symfunc_type = 'radial'
myGenerator.generate_radial_params(rule='imbalzano2018', mode='shift', nb_param_pairs=5)

myGenerator.symfunc_type = 'angular_narrow'
myGenerator.zetas = [1.0, 6.0]
myGenerator.generate_radial_params(rule='gastegger2018', mode='center', nb_param_pairs=3, r_lower=1.5)


with open('example-outfile.txt', 'w') as f:
    myGenerator.write_settings_overview(fileobj=f)
    myGenerator.write_parameter_strings(fileobj=f)



