# import isotherms

# import the iast module
import pygaps
import pygaps.prediction.iast as pgi
from pygaps import PointIsotherm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_isoterm(df_data):
    # Sample data (replace with your actual data)
    df_data = df_data.sort_values(by="Pressure(Pascal)")
    pressure = df_data["Pressure(Pascal)"] / 1e5 # in bar
    loading = df_data["ExcesUptake(mmol/g)"] # in mmol/g
    # Create a PointIsotherm object
    return PointIsotherm(
        pressure=pressure,
        loading=loading,
        material="MOF-74(Mg)",
        adsorbate="CO2",
        temperature=298
    )


#  df_data_co2 = pd.read_csv("./uptakes_co2_ch4_mixture_nnp_CO2_v10.csv")
#  df_data_ch4 = pd.read_csv("./uptakes_co2_ch4_mixture_nnp_CH4_v10.csv")
sim_type = "flexible"

if sim_type == "rigid":
    df_pure_data_ch4 = pd.read_csv("../../CH4/rigid_ch4_up5bar.csv")
    df_pure_data_co2 = pd.read_csv("../../CO2/rigid_co2_up5bar.csv")
if sim_type == "flexible":
    df_pure_data_ch4 = pd.read_csv("../../CH4/flexible_ch4_up5bar.csv")
    #  df_pure_data_ch4 = pd.read_csv("../../CH4/experiment_ch4_up5bar.csv")
    df_pure_data_co2 = pd.read_csv("../../CO2/flexible_co2_up5bar.csv")

#  isotherm = get_isoterm(df_data)
# Fit Langmuir model
isotherms_iast_models = []
for df_data in [df_pure_data_co2, df_pure_data_ch4]:
    isotherm = get_isoterm(df_data)
    model = pygaps.ModelIsotherm.from_pointisotherm(isotherm,
                                                    model="quadratic",
                                                    #  optimization_method="L-BFGS-B",
                                                    verbose=True)
    isotherms_iast_models += [model]


#  gas_fraction = [0.5, 0.5]
#  total_pressure = 1
#  pgi.iast_binary_vle(isotherm, total_pressure, verbose=True)

#  gas_fraction = [0.5, 0.5]
#  total_pressure = 10
#  res = pgi.iast_point_fraction(isotherms_iast_models, gas_fraction, total_pressure, verbose=False)
#  print(res)
#
#  plt.show()


# Define gas-phase composition and total pressure
upper_pressure_limit = 5  # in bar
gas_fraction = np.array([0.5, 0.5])  # gas mole fractions
#  y = np.array([0.9, 0.1])  # gas mole fractions

# Generate mixture isotherm (loading vs. pressure)
total_pressures = np.linspace(start=0.001, stop=upper_pressure_limit, num=50)
loadings_co2 = []
loadings_ch4 = []

for P in total_pressures:
    #  loading = pyiast.iast(P * y, [iso_ch4, iso_co2], verboseflag=True)
    loading = pgi.iast_point_fraction(isotherms_iast_models, gas_fraction, P, verbose=False)
    loadings_co2.append(loading[0])
    loadings_ch4.append(loading[1])

df_iast = pd.DataFrame()
df_iast["Pressure(Pascal)"] = total_pressures * 1e5 # bar to pascal
df_iast["CH4ExcesUptake(mmol/g)"] = loadings_ch4
df_iast["CO2ExcesUptake(mmol/g)"] = loadings_co2
df_iast.to_csv(f"{sim_type}_iast_{int(gas_fraction[0]*100)}_{int(gas_fraction[1]*100)}.csv")



