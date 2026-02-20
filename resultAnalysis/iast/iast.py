#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import least_squares, brentq
from scipy.integrate import quad



# -------------------------
# Dual-Site Langmuir model
# -------------------------
@dataclass
class DSLangmuir:
    M1: float
    K1: float
    M2: float
    K2: float

    def loading(self, p_bar):
        p = np.asarray(p_bar, dtype=float)
        k1p = self.K1 * p
        k2p = self.K2 * p
        return self.M1 * k1p / (1.0 + k1p) + self.M2 * k2p / (1.0 + k2p)

    def spreading_pressure(self, p_bar):
        p = float(p_bar)
        if p <= 0:
            return 0.0
        return self.M1*np.log(1+self.K1*p) + self.M2*np.log(1+self.K2*p)


def fit_ds_langmuir(p_bar, n):
    p = np.asarray(p_bar, float)
    q = np.asarray(n, float)

    mask = np.isfinite(p) & np.isfinite(q) & (p >= 0) & (q >= 0)
    p, q = p[mask], q[mask]

    qmax = np.max(q)
    slope = q[1]/p[1] if p[1] > 0 else 1.0

    x0 = [0.6*qmax, 0.5*slope/max(qmax,1e-12),
          0.4*qmax, 0.5*slope/max(qmax,1e-12)]

    lb = [1e-12]*4
    ub = [np.inf]*4

    def resid(x):
        M1,K1,M2,K2 = x
        return DSLangmuir(M1,K1,M2,K2).loading(p) - q

    res = least_squares(resid, x0, bounds=(lb,ub), max_nfev=20000)
    if not res.success:
        raise RuntimeError(res.message)

    return DSLangmuir(*res.x)


@dataclass
class DualSiteSips:
    M1: float
    K1: float
    a1: float
    M2: float
    K2: float
    a2: float

    def loading(self, p_bar):
        p = np.asarray(p_bar, dtype=float)
        t1 = (self.K1 * p) ** self.a1
        t2 = (self.K2 * p) ** self.a2
        return self.M1 * t1 / (1.0 + t1) + self.M2 * t2 / (1.0 + t2)

    def spreading_pressure(self, p_bar):
        p = float(p_bar)
        if p <= 0:
            return 0.0
        return (self.M1 / self.a1) * np.log(1.0 + (self.K1 * p) ** self.a1) + \
               (self.M2 / self.a2) * np.log(1.0 + (self.K2 * p) ** self.a2)


def fit_dual_site_sips(p_bar, n):
    """
    Fit Dual-site Sips (dual-site Langmuir–Freundlich) with bounded least squares.
    """
    p = np.asarray(p_bar, float)
    q = np.asarray(n, float)

    mask = np.isfinite(p) & np.isfinite(q) & (p >= 0) & (q >= 0)
    p, q = p[mask], q[mask]
    if p.size < 10:
        raise ValueError("Need >=10 points for stable DualSiteSips fit.")

    qmax = float(np.max(q))

    # estimate slope using lowest nonzero point
    nonzero = p > 0
    if np.any(nonzero):
        idx = np.argmin(p[nonzero])
        pmin = float(p[nonzero][idx])
        qmin = float(q[nonzero][idx])
        slope = qmin / pmin if pmin > 0 else 1.0
    else:
        slope = 1.0

    # initial guesses
    K_guess = 0.5 * slope / max(qmax, 1e-12)
    x0 = np.array([
        0.6*qmax, K_guess, 1.0,
        0.4*qmax, 0.2*K_guess, 1.0
    ], dtype=float)

    # bounds: M>0, K>0, a in [0.1, 5]
    lb = np.array([1e-12, 1e-12, 0.1,  1e-12, 1e-12, 0.1], dtype=float)
    ub = np.array([np.inf, np.inf, 5.0,  np.inf, np.inf, 5.0], dtype=float)

    def resid(x):
        M1, K1, a1, M2, K2, a2 = x
        return DualSiteSips(M1, K1, a1, M2, K2, a2).loading(p) - q

    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=50000)
    if not res.success:
        raise RuntimeError(f"DualSiteSips fit failed: {res.message}")

    return DualSiteSips(*map(float, res.x))


@dataclass
class DualSiteMF:
    """
    Dual-site Mean-Field Langmuir (MF lateral interactions), per your figure:

    theta_i(P) = K_i* P / (1 + K_i* P)
    K_eff,i(P) = K_i* * exp(J_i * theta_i(P))
    q_i(P)     = qsat_i * (K_eff,i(P) * P) / (1 + K_eff,i(P) * P)

    Total q(P) = q1(P) + q2(P)

    Notes:
    - If J_i = 0, reduces to dual-site Langmuir with K_i = K_i*
    - Spreading pressure is computed numerically (no simple closed-form).
    """
    q1: float
    K1s: float
    J1: float
    q2: float
    K2s: float
    J2: float

    def _site_loading(self, P, qsat, Ks, J):
        P = np.asarray(P, dtype=float)
        theta = (Ks * P) / (1.0 + Ks * P)  # in [0,1)
        Keff = Ks * np.exp(J * theta)
        return qsat * (Keff * P) / (1.0 + Keff * P)

    def loading(self, p_bar):
        P = np.asarray(p_bar, dtype=float)
        return self._site_loading(P, self.q1, self.K1s, self.J1) + \
               self._site_loading(P, self.q2, self.K2s, self.J2)

    def spreading_pressure(self, p_bar):
        """
        Pi(P) = ∫_0^P q(p)/p dp
        Numerically integrated with a safe p->0 treatment.
        """
        P = float(p_bar)
        if P <= 0.0:
            return 0.0

        # Avoid singularity at p=0 by using the limiting slope q/p -> KH.
        # Estimate KH with a tiny pressure.
        p0 = min(1e-8, 1e-8 * max(1.0, P))
        q0 = float(self.loading(p0))
        KH = q0 / p0 if p0 > 0 else 0.0

        eps = 1e-15 * max(1.0, P)

        def integrand(p):
            if p <= eps:
                return KH
            return float(self.loading(p) / p)

        val, _ = quad(integrand, 0.0, P, limit=200)
        return float(val)


def fit_dual_site_mf(p_bar, n):
    """
    Fit DualSiteMF parameters to data with bounded least squares.
    Parameters:
      q1,q2 > 0
      K1s,K2s > 0
      J1,J2 in a reasonable range [-50, 50] (tune if needed)
    """
    p = np.asarray(p_bar, float)
    q = np.asarray(n, float)

    mask = np.isfinite(p) & np.isfinite(q) & (p >= 0) & (q >= 0)
    p, q = p[mask], q[mask]
    if p.size < 10:
        raise ValueError("Need >=10 points for stable DualSiteMF fit.")

    qmax = float(np.max(q))

    # slope estimate from smallest nonzero point
    nonzero = p > 0
    if np.any(nonzero):
        idx = np.argmin(p[nonzero])
        pmin = float(p[nonzero][idx])
        qmin = float(q[nonzero][idx])
        slope = qmin / pmin if pmin > 0 else 1.0
    else:
        slope = 1.0

    # initial guesses: split capacity, J=0 -> dual-site Langmuir-like start
    K_guess = 0.5 * slope / max(qmax, 1e-12)
    x0 = np.array([
        0.6*qmax, K_guess, 0.0,
        0.4*qmax, 0.2*K_guess, 0.0
    ], dtype=float)

    lb = np.array([1e-12, 1e-12, -50.0,  1e-12, 1e-12, -50.0], dtype=float)
    ub = np.array([np.inf, np.inf,  50.0,  np.inf, np.inf,  50.0], dtype=float)

    def resid(x):
        q1, K1s, J1, q2, K2s, J2 = x
        model = DualSiteMF(q1, K1s, J1, q2, K2s, J2)
        return model.loading(p) - q

    res = least_squares(resid, x0, bounds=(lb, ub), max_nfev=80000)
    if not res.success:
        raise RuntimeError(f"DualSiteMF fit failed: {res.message}")

    return DualSiteMF(*map(float, res.x))

# -------------------------
# Plot helper
# -------------------------
def plot_fit(df, model, pcol, qcol, name, MODEL,save_png=True, show=True):
    p = df[pcol].values
    q = df[qcol].values

    pgrid = np.linspace(max(p.min(),1e-6), p.max(), 300)
    qfit = model.loading(pgrid)

    rmse = np.sqrt(np.mean((model.loading(p) - q)**2))

    plt.figure()
    plt.scatter(p, q, s=25, label="data")
    plt.plot(pgrid, qfit, label=f"{MODEL} fit\nRMSE={rmse:.4g}")
    plt.xlabel("Pressure (bar)")
    plt.ylabel("Loading")
    plt.title(f"{name} {MODEL} fit")
    plt.legend()
    plt.tight_layout()

    if save_png:
        plt.savefig(f"{name}_ds_langmuir_fit.png", dpi=200)

    if show:
        plt.show()
    else:
        plt.close()


# -------------------------
# IAST core
# -------------------------

def inv_spreading_pressure(model, pi, p_hi=1.0, p_floor=1e-15):
    """
    Solve for p such that Pi(p)=pi with p constrained to be > p_floor for pi>0.
    """
    pi = float(pi)
    if pi <= 0.0:
        return 0.0

    def f(p):
        return model.spreading_pressure(p) - pi

    lo = float(p_floor)
    # Ensure hi > lo
    hi = max(float(p_hi), lo * 10.0)

    # If even at lo we're already above pi, return lo (pi extremely small)
    if f(lo) >= 0.0:
        return lo

    # grow hi until f(hi) > 0
    for _ in range(200):
        if f(hi) > 0.0:
            break
        hi *= 2.0
    else:
        raise RuntimeError("Could not bracket p for inverse spreading pressure. "
                           "Check units/fit or increase p_hi.")

    return brentq(f, lo, hi, maxiter=500)


def iast_binary(P, y, isoterms, p_floor=1e-15):
    """
    Robust binary IAST with automatic pi bracketing that avoids p0=0 divisions.
    P in bar, y=[y1,y2]. Returns [n1,n2].
    """
    P = float(P)
    m1, m2 = isoterms[0], isoterms[1]
    y1, y2 = map(float, y)

    if P <= 0.0:
        return np.array([0.0, 0.0])

    if y1 < 0 or y2 < 0 or abs((y1 + y2) - 1.0) > 1e-10:
        raise ValueError("y must be nonnegative and sum to 1.")

    def h(pi):
        p01 = inv_spreading_pressure(m1, pi, p_hi=P, p_floor=p_floor)
        p02 = inv_spreading_pressure(m2, pi, p_hi=P, p_floor=p_floor)
        # p01, p02 guaranteed >= p_floor for pi>0
        return y1 * P / p01 + y2 * P / p02 - 1.0

    # --- choose pi_lo so that inverse pressures are not at the floor ---
    pi_lo = 1e-12  # start modest; will increase if needed
    for _ in range(100):
        p01 = inv_spreading_pressure(m1, pi_lo, p_hi=P, p_floor=p_floor)
        p02 = inv_spreading_pressure(m2, pi_lo, p_hi=P, p_floor=p_floor)
        if p01 > 10*p_floor and p02 > 10*p_floor:
            break
        pi_lo *= 10.0
    else:
        raise RuntimeError("Could not find a safe pi_lo (inverse pressures too small). "
                           "Check model parameters/units.")

    hlo = h(pi_lo)

    # --- bracket pi_hi where sign flips ---
    pi_hi = pi_lo
    for _ in range(200):
        pi_hi *= 2.0
        hhi = h(pi_hi)
        if hlo * hhi < 0.0:
            break
    else:
        raise RuntimeError(f"Could not bracket pi root. h(pi_lo)={hlo:.3e}, h(pi_hi)={hhi:.3e}. "
                           "Likely a units issue (bar vs Pa) or bad fit.")

    pi_star = brentq(h, pi_lo, pi_hi, maxiter=500)

    # fictitious pressures
    p01 = inv_spreading_pressure(m1, pi_star, p_hi=P, p_floor=p_floor)
    p02 = inv_spreading_pressure(m2, pi_star, p_hi=P, p_floor=p_floor)

    # adsorbed phase composition
    x1 = y1 * P / p01
    x2 = y2 * P / p02
    xsum = x1 + x2
    x1 /= xsum
    x2 /= xsum

    # pure loadings at fictitious pressures
    n01 = float(m1.loading(p01))
    n02 = float(m2.loading(p02))

    # total uptake
    ntot = 1.0 / (x1 / n01 + x2 / n02)
    return np.array([ntot * x1, ntot * x2])


# -------------------------
# MAIN
# -------------------------
def main():

    PCOL = "Pressure(bar)"
    QCOL = "ExcesUptake(mmol/g)"

    upper_pressure_limit = 20.0 # bar

    y = np.array([0.5,0.5])
    #  Pgrid = np.linspace(0.1,20,60)
    Pgrid = np.linspace(start=0.001, stop=8, num=60)

    # Choose model: "DSLangmuir" or "DualSiteSips"
    MODEL = "DualSiteSips"
    #  MODEL = "DualSiteMF"

    #  sim_type = "flexible"
    gas_type_list = ["ch4", "co2"]
    gas_csv = {}
    for sim_type in ["experiment"]:
        for gas_type in gas_type_list:
            if sim_type == "experiment":
                df_gas = pd.read_csv(f"../../{gas_type.upper()}/{sim_type}_{gas_type}.csv")
                df_gas["Pressure(bar)"] = df_gas["Pressure(Pascal)"] / 1e5 # in bar
                df_gas =  df_gas[df_gas["Pressure(bar)"] <= upper_pressure_limit]
                gas_csv[gas_type] = df_gas
            else:
                df_gas = pd.read_csv(f"../../{gas_type.upper()}/{sim_type}_{gas_type}.csv")
                df_gas["Pressure(bar)"] = df_gas["Pressure(Pascal)"] / 1e5 # in bar
                df_gas =  df_gas[df_gas["Pressure(bar)"] <= upper_pressure_limit]
                gas_csv[gas_type] = df_gas


        gas_iso = {}
        if MODEL == "DSLangmuir":
            for gas_type in gas_type_list:
                gas_iso[gas_type] = fit_ds_langmuir(gas_csv[gas_type][PCOL], gas_csv[gas_type][QCOL])
                print(f"{gas_type} params:", gas_iso[gas_type])
                # -------- PLOTS --------
                plot_fit(gas_csv[gas_type], gas_iso[gas_type], PCOL, QCOL, gas_type, MODEL, show=False)
        elif MODEL == "DualSiteSips":
            for gas_type in gas_type_list:
                gas_iso[gas_type] = fit_dual_site_sips(gas_csv[gas_type][PCOL], gas_csv[gas_type][QCOL])
                print(f"{gas_type} params:", gas_iso[gas_type])
                # -------- PLOTS --------
                plot_fit(gas_csv[gas_type], gas_iso[gas_type], PCOL, QCOL, gas_type, MODEL, show=False)
        elif MODEL == "DualSiteMF":
            for gas_type in gas_type_list:
                gas_iso[gas_type] = fit_dual_site_mf(gas_csv[gas_type][PCOL], gas_csv[gas_type][QCOL])
                print(f"{gas_type} params:", gas_iso[gas_type])
                # -------- PLOTS --------
                plot_fit(gas_csv[gas_type], gas_iso[gas_type], PCOL, QCOL, gas_type, MODEL, show=False)
        else:
            raise ValueError("Unknown MODEL")

        #  plot_fit(df_ch4, iso_ch4, PCOL, QCOL, "CH4", MODEL)

        #  print("CH4 params:", iso_ch4)
    #  co2_csv = "./experiment_co2.csv"
    #  ch4_csv = "./experiment_ch4.csv"
    #
    #
    #  df_co2 = pd.read_csv(co2_csv).sort_values(by="Pressure(Pascal)")
    #  df_co2["Pressure(bar)"] = df_co2["Pressure(Pascal)"] / 1e5 # in bar
    #  df_co2 = df_co2[df_co2["Pressure(bar)"] <= upper_pressure_limit]
    #
    #  df_ch4 = pd.read_csv(ch4_csv).sort_values(by="Pressure(Pascal)")
    #  df_ch4["Pressure(bar)"] = df_ch4["Pressure(Pascal)"] / 1e5 # in bar
    #  df_ch4 = df_ch4[df_ch4["Pressure(bar)"] <= upper_pressure_limit]


    #
    # -------- IAST --------
    rows=[]
    loadings = np.zeros((len(Pgrid), 2))  # initialize loadings array for each gas
    for i, P in enumerate(Pgrid):
        loading = iast_binary(P, y, [gas_iso[gas_type] for gas_type in gas_type_list])
        for j, gas_type in enumerate(gas_type_list):
            loadings[i][j] = loading[j]
        #  rows.append({
        #      "P_bar":P,
        #      "CO2":n[0],
        #      "CH4":n[1],
        #      "n_total":np.sum(n)

        #  })

    for i, gas_type in enumerate(gas_type_list):
        df_iast = pd.DataFrame()
        df_iast["Pressure(Pascal)"] = Pgrid * 1e5 # bar to pascal
        df_iast["ExcesUptake(mmol/g)"] = loadings[:, i]
        df_iast.to_csv(f"{sim_type}_{gas_type}_iast_{int(y[0]*100)}_{int(y[1]*100)}.csv", index=False)



if __name__ == "__main__":
    main()

