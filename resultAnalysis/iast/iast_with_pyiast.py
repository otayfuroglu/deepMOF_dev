#
import numpy as np
import pandas as pd
import pyiast

import matplotlib.pyplot as plt


# fit model types
#['Langmuir', 'Quadratic', 'BET', 'Henry', 'TemkinApprox', 'DSLangmuir']


def getModelIsoterm(df_data, fit_type="analaytical"):
    df_data = df_data.sort_values(by="Pressure(Pascal)")
    df_data["Pressure(bar)"] = df_data["Pressure(Pascal)"] / 1e5 # in bar
    if fit_type == "analaytical":
        model_isoterm = pyiast.ModelIsotherm(df_data[df_data["Pressure(bar)"] <= upper_pressure_limit],
                                        loading_key="ExcesUptake(mmol/g)",
                                        pressure_key="Pressure(bar)",
                                        #  model="BET",
                                        #  model="TemkinApprox",
                                        #  model="Quadratic",
                                        #  model="Langmuir",
                                        #  model="DSLangmuir",
                                        model="DualSiteMF",
                                        #  optimization_method="Nelder-Mead",
                                        optimization_method="L-BFGS-B", # for DualSiteMF beter results others
                                        #  optimization_method="Powell",
                                )
    elif fit_type == "interpolation":
        model_isoterm = pyiast.InterpolatorIsotherm(df_data[df_data["Pressure(bar)"] <= upper_pressure_limit],
                                        loading_key="ExcesUptake(mmol/g)",
                                        pressure_key="Pressure(bar)",
                                        fill_value=df_data["ExcesUptake(mmol/g)"].max() * 1.2,
                                )
    else:
        raise ValueError("Invalid fit type. Choose 'analaytical' or 'interpolation'.")

    pyiast.plot_isotherm(model_isoterm, withfit=True, xlogscale=False,)
    return model_isoterm


#  df_pure_data_co2 = pd.read_csv("./flexible_co2_up5bar.csv")
#  df_pure_data_co2 = pd.read_csv("./experiment_co2_up5bar.csv")
#  df_pure_data_ch4 = pd.read_csv("./experiment_ch4_up5bar.csv")

upper_pressure_limit = 25  # in bar
#  sim_type = "rigid"
#  sim_type = "flexible"
gas_type_list = ["ch4", "co2"]
gas_csv = {}
#  if sim_type == "rigid":
#  for sim_type in ["rigid", "flexible", "experiment"]:

# Define gas-phase composition and total pressure
#  y = np.array([0.5, 0.5])  # gas mole fractions
y = np.array([0.9, 0.1])  # gas mole fractions

# Generate mixture isotherm (loading vs. pressure)
total_pressures = np.linspace(start=0.001, stop=8, num=60)

for sim_type in ["experiment"]:
    for gas_type in gas_type_list:
        if sim_type == "experiment":
            gas_csv[gas_type] = pd.read_csv(f"../../{gas_type.upper()}/{sim_type}_{gas_type}.csv")
        else:
            gas_csv[gas_type] = pd.read_csv(f"../../{gas_type.upper()}/{sim_type}_{gas_type}_up5bar.csv")

    gas_iso = {}
    for gas_type in gas_type_list:
        gas_iso[gas_type] = getModelIsoterm(gas_csv[gas_type], fit_type="analaytical")

    #  for gas_type in gas_type_list:
    loadings = np.zeros((len(total_pressures), 2))  # initialize loadings array for each gas
    for i, P in enumerate(total_pressures):
        loading = pyiast.iast(P * y, [gas_iso[gas_type] for gas_type in gas_type_list], verboseflag=True)
        for j, gas_type in enumerate(gas_type_list):
            loadings[i][j] = loading[j]

    for i, gas_type in enumerate(gas_type_list):
        df_iast = pd.DataFrame()
        df_iast["Pressure(Pascal)"] = total_pressures * 1e5 # bar to pascal
        df_iast["ExcesUptake(mmol/g)"] = loadings[:, i]
        df_iast.to_csv(f"{sim_type}_{gas_type}_iast_{int(y[0]*100)}_{int(y[1]*100)}.csv")


