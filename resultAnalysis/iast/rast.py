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
# RAST core
# -------------------------

class RegularSolutionGamma:
    """
    Simple 2D regular-solution style activity coefficients:
      ln(gamma1) = A * x2^2
      ln(gamma2) = A * x1^2
    A can be constant or a function A(pi).
    A=0 -> IAST.
    """
    def __init__(self, A):
        self.A = A  # float or callable(pi)->float

    def gammas(self, x1, pi):
        x1 = float(x1)
        x2 = 1.0 - x1
        A = self.A(pi) if callable(self.A) else float(self.A)
        g1 = np.exp(A * (x2**2))
        g2 = np.exp(A * (x1**2))
        return g1, g2


class Margules2:
    def __init__(self, A12, A21):
        self.A12 = A12  # float or callable(pi)->float
        self.A21 = A21

    def gammas(self, x1, pi):
        x1 = float(x1); x2 = 1.0 - x1
        A12 = self.A12(pi) if callable(self.A12) else float(self.A12)
        A21 = self.A21(pi) if callable(self.A21) else float(self.A21)

        ln_g1 = (A12 + 2.0*(A21 - A12)*x1) * (x2**2)
        ln_g2 = (A21 + 2.0*(A12 - A21)*x2) * (x1**2)
        return np.exp(ln_g1), np.exp(ln_g2)


class WilsonBinary_Phi:
    """
    Wilson activity coefficients for RAST exactly in the style of the uploaded SI:
      ln(gamma_i) = ln(gamma_i)_Wilson * (1 - exp(-C * Phi))

    where:
      Phi = (pi*A)/(R*T)  [mol/kg]  (surface potential)
      C has units [kg/mol]
      Lambda11 = Lambda22 = 1

    Pass Phi (NOT pi) into gammas().
    """

    def __init__(self, L12, L21, C):
        # L12, L21 > 0 ; C >= 0
        self.L12 = L12  # float or callable(Phi)->float
        self.L21 = L21  # float or callable(Phi)->float
        self.C = C      # float or callable(Phi)->float

    def gammas(self, x1, Phi):
        x1 = float(x1)
        x2 = 1.0 - x1
        Phi = float(Phi)

        L12 = self.L12(Phi) if callable(self.L12) else float(self.L12)
        L21 = self.L21(Phi) if callable(self.L21) else float(self.L21)
        C   = self.C(Phi)   if callable(self.C)   else float(self.C)

        if L12 <= 0 or L21 <= 0:
            raise ValueError("Wilson Lambdas (L12,L21) must be > 0.")
        if C < 0:
            raise ValueError("C must be >= 0.")
        if not (0.0 < x1 < 1.0):
            # still works at endpoints, but avoid log/0 issues in solvers
            x1 = min(max(x1, 1e-15), 1.0 - 1e-15)
            x2 = 1.0 - x1

        eps = 1e-15
        t1 = max(x1 + L12 * x2, eps)
        t2 = max(x2 + L21 * x1, eps)

        # Classical Wilson binary (Lambda11=Lambda22=1)
        ln_g1_w = -np.log(t1) + x2 * (L12 / t1 - L21 / (L21 * x1 + x2 + eps))
        ln_g2_w = -np.log(t2) + x1 * (L21 / t2 - L12 / (x1 + L12 * x2 + eps))

        # RAST correction factor (Henry limit -> 0, saturation -> 1)
        expo = np.clip(-C * Phi, -700.0, 700.0)
        F = 1.0 - np.exp(expo)

        ln_g1 = ln_g1_w * F
        ln_g2 = ln_g2_w * F

        return float(np.exp(ln_g1)), float(np.exp(ln_g2))


def rast_binary(P, y, isoterms, gamma_model, *, p_floor=1e-15, dphi_rel=1e-6):
    """
    RAST for binary mixture, SI-consistent:
      - Composition constraint uses modified Raoult-like relation with activity coefficients:
          y_i * P = x_i * gamma_i(x, Phi) * P_i^0(Phi)
      - Total loading uses excess correction:
          q_t = sum_i x_i * q_i^0(P_i^0) + q_excess
          1/q_excess = d/dPhi (G_excess/RT)
          G_excess/RT = x1 ln(gamma1) + x2 ln(gamma2)

    Assumptions / conventions:
      - We treat fugacity ~ pressure (i.e. f_i = y_i P).
      - m.spreading_pressure(p) returns the “surface potential” Phi used in the SI
        (pyIAST-style reduced spreading pressure works in this role).
      - m.loading(p) returns pure loading q^0 at pressure p.
      - gamma_model.gammas(x1, Phi) -> (gamma1, gamma2).
    """
    P = float(P)
    y1, y2 = map(float, y)

    if P <= 0.0:
        return np.array([0.0, 0.0]), np.array([np.nan, np.nan]), 0.0

    if y1 < 0 or y2 < 0 or abs((y1 + y2) - 1.0) > 1e-12:
        raise ValueError("y must be nonnegative and sum to 1.")

    # ---------- inverse Phi: find P0 such that Phi(P0) = Phi_target ----------
    def inv_phi(model, Phi_target):
        Phi_target = float(Phi_target)
        if Phi_target <= 0.0:
            return 0.0

        def f(p):
            return model.spreading_pressure(p) - Phi_target

        lo = float(p_floor)
        hi = max(P, lo * 10.0)

        # if already above target at lo, clamp
        if f(lo) >= 0.0:
            return lo

        for _ in range(250):
            if f(hi) > 0.0:
                break
            hi *= 2.0
        else:
            raise RuntimeError("Could not bracket P0 in inv_phi; check units/fit.")
        return brentq(f, lo, hi, maxiter=700)

    # ---------- inner: for a given Phi, solve x self-consistently ----------
    def x_from_phi(Phi):
        p01 = inv_phi(m1, Phi)
        p02 = inv_phi(m2, Phi)

        # Solve x1 with self-consistency since gamma depends on x (and Phi)
        def r(x1):
            x1 = float(x1)
            g1, g2 = gamma_model.gammas(x1, Phi)
            # Modified Raoult-like relation -> unnormalized x estimates:
            x1_rhs = y1 * P / (g1 * p01)
            x2_rhs = y2 * P / (g2 * p02)
            s = x1_rhs + x2_rhs
            x1_new = x1_rhs / s
            return x1_new - x1

        a, b = 1e-12, 1.0 - 1e-12
        ra, rb = r(a), r(b)

        if ra * rb > 0:
            # fallback fixed-point iterations
            x1 = y1
            for _ in range(80):
                g1, g2 = gamma_model.gammas(x1, Phi)
                x1_rhs = y1 * P / (g1 * p01)
                x2_rhs = y2 * P / (g2 * p02)
                s = x1_rhs + x2_rhs
                x1_new = x1_rhs / s
                if abs(x1_new - x1) < 1e-12:
                    break
                x1 = x1_new
        else:
            x1 = brentq(r, a, b, maxiter=500)

        x2 = 1.0 - x1
        return float(x1), float(x2), float(p01), float(p02)

    # ---------- outer: solve Phi from normalization condition ----------
    # S(Phi) = sum_i y_i P / (gamma_i * P_i^0(Phi)) - 1 = 0
    def S(Phi):
        x1, x2, p01, p02 = x_from_phi(Phi)
        g1, g2 = gamma_model.gammas(x1, Phi)
        return y1 * P / (g1 * p01) + y2 * P / (g2 * p02) - 1.0

    Phi_lo = 1e-10
    s_lo = S(Phi_lo)

    Phi_hi = Phi_lo
    for _ in range(250):
        Phi_hi *= 2.0
        s_hi = S(Phi_hi)
        if s_lo * s_hi < 0.0:
            break
    else:
        raise RuntimeError("Could not bracket Phi root in RAST; check gamma model/units/fit.")

    Phi_star = brentq(S, Phi_lo, Phi_hi, maxiter=700)

    # ---------- final x and fictitious pressures ----------
    x1, x2, p01, p02 = x_from_phi(Phi_star)

    # pure loadings at fictitious pressures
    q01 = float(m1.loading(p01))
    q02 = float(m2.loading(p02))

    # ---------- excess correction for total loading (SI Eq. S33–S34) ----------
    # G_excess/RT = x1 ln(g1) + x2 ln(g2)
    def Gex_over_RT(Phi):
        g1, g2 = gamma_model.gammas(x1, Phi)  # IMPORTANT: derivative at fixed composition
        return x1 * np.log(max(g1, 1e-300)) + x2 * np.log(max(g2, 1e-300))

    dPhi = dphi_rel * max(1.0, abs(Phi_star))
    Phi_p = Phi_star + dPhi
    Phi_m = max(0.0, Phi_star - dPhi)

    if Phi_m == Phi_p:  # extremely small Phi
        dGdPhi = 0.0
    else:
        dGdPhi = (Gex_over_RT(Phi_p) - Gex_over_RT(Phi_m)) / (Phi_p - Phi_m)

    # 1/q_excess = d(Gex/RT)/dPhi  -> q_excess = 1/dGdPhi
    if abs(dGdPhi) < 1e-14:
        q_excess = 0.0
    else:
        q_excess = 1.0 / dGdPhi

    q_total = x1 * q01 + x2 * q02 + q_excess
    q_total = max(q_total, 0.0)

    # component mixture loadings
    n1 = x1 * q_total
    n2 = x2 * q_total

    return np.array([n1, n2]), np.array([x1, x2]), float(Phi_star)



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

    gamma = RegularSolutionGamma(A=1.8)   # A=0 -> IAST
    #  gamma = WilsonBinary_Phi(L12=3.622, L21=0.167, C=0.056)
    #  gamma = Margules2(A12=3.662, A21=0.867)

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
        #  loading = iast_binary(P, y, )
        loading, x, pi = rast_binary(P, y, [gas_iso[gas_type] for gas_type in gas_type_list], gamma_model=gamma)

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

