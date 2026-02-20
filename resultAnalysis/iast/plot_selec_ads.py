#
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



def get_press_loading(df_data):
    df_data = df_data.sort_values(by="Pressure(Pascal)")
    df_data["Pressure(bar)"] = df_data["Pressure(Pascal)"] / 1e5 # in bar
    #  df_data = df_data[df_data["Pressure(bar)"] <6.0]
    return df_data["Pressure(Pascal)"] / 1e5, df_data[f"ExcesUptake(mmol/g)"]


def get_press_loading_binary(df_data):
    df_data = df_data.sort_values(by="Pressure(Pascal)")
    df_data["Pressure(bar)"] = df_data["Pressure(Pascal)"] / 1e5 # in bar
    #  df_data = df_data[df_data["Pressure(bar)"] <6.0]
    return df_data["Pressure(Pascal)"] / 1e5, df_data[f"{gas_type.upper()}ExcesUptake(mmol/g)"]


def get_press_selec(loading_gas1, loading_gas2):
    return (loading_gas1 / loading_gas2)



y = np.array([0.5, 0.5])  # gas mole fractions
#  y = np.array([0.9, 0.1])  # gas mole fractions

sim_type = "rigid"
#  sim_type = "flexible"
#  gas_type = "ch4"
gas_type = "co2"

if gas_type == "co2":
    gas_id = 2
elif gas_type == "ch4":
    gas_id = 1

uptake_type = f"{sim_type}_binary"

#  df_data_single_gas = pd.read_csv(f"../../{gas_type.upper()}/{sim_type}_{gas_type}_up5bar.csv")
#
#  df_data_single_gas_exp = pd.read_csv(f"../../{gas_type.upper()}/experiment_{gas_type}_up5bar.csv")
#
#  df_data_mix_gas = pd.read_csv(f"./avg_5replica_{uptake_type}_uptakes_{int(y[0]*100)}_{int(y[1]*100)}_{gas_id}.csv")

gas_data ={}
for gas_type in ["co2", "ch4"]:
    df_data_gas = pd.read_csv(f"../{sim_type}_iast_{int(y[0]*100)}_{int(y[1]*100)}.csv")
    pressure_mix_iast_gas, loading_mix_iast_gas = get_press_loading_binary(df_data_gas)
    gas_data[gas_type] = (pressure_mix_iast_gas, loading_mix_iast_gas)

    #  plt.plot(pressure_mix_iast_gas, loading_mix_iast_gas, "s", color="k", label=f'{gas_type} IAST', markerfacecolor='none')


selec = get_press_selec(gas_data["co2"][1], gas_data["ch4"][1])
plt.plot(gas_data["co2"][0], selec, "s", color="k", label=f'{gas_type} IAST', markerfacecolor='none')
plt.show()
quit()

# Plot mixture isotherms
#  pressure_pure_gas, loading_pure_gas = get_press_loading(df_data_single_gas)
#  pressure_pure_gas_exp, loading_pure_gas_exp = get_press_loading(df_data_single_gas_exp)
#
#  pressure_mix_gas, loading_mix_gas = get_press_loading_binary(df_data_mix_gas)
#
#  pressure_mix_iast_gas, loading_mix_iast_gas = get_press_loading_binary(df_iast_data_mix_gas)

#  plt.plot(pressure_pure_gas, loading_pure_gas, "o", label=f'single {gas_type} {sim_type} sim.',)
#  plt.plot(pressure_pure_gas_exp, loading_pure_gas_exp, "k--", label=f'single {gas_type} exp.', markerfacecolor='none')
#  plt.plot(pressure_mix_gas, loading_mix_gas, "o", label=f'{gas_type} in mixture', markerfacecolor='none')
#  plt.plot(pressure_mix_iast_gas, loading_mix_iast_gas, "s", color="k", label=f'{gas_type} IAST', markerfacecolor='none')


plt.xlabel('Total Pressure (bar)')
plt.ylabel('Loading (mmol/g)')
plt.title(f'Mixture Isotherms from IAST ({int(y[0]*100)}_{int(y[1]*100)} CO2/CH4)')
#  plt.title('Mixture Isotherms from IAST (10/90 CO2/CH4)')
plt.legend()
plt.xlim(-0.1, 5.5)
#  plt.xlim(0, 0.2)
#  plt.ylim(0, 14.1)
#  plt.ylim(0, 0.5)
#  plt.grid(True)
#  plt.show()

plt.savefig(f"{uptake_type}_gas{gas_type}_iast_{int(y[0]*100)}_{int(y[1]*100)}.png", dpi=1000)

