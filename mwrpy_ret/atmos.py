"""Module for atmsopheric functions."""
import numpy as np

import mwrpy_ret.constants as con

HPA_TO_P = 100


def spec_heat(T: np.ndarray) -> np.ndarray:
    """Specific heat for evaporation (J/kg)"""
    return con.LATENT_HEAT - 2420.0 * (T - con.T0)


def abs_hum(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    "Absolute humidity (kg/m^3)"
    es = calc_saturation_vapor_pressure(T)
    return (rh * es) / (con.RW * T)


def calc_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Goff-Gratch formula for saturation vapor pressure (Pa) over water adopted by WMO.
    Args:
        temperature: Temperature (K).
    Returns:
        Saturation vapor pressure (Pa).
    """
    ratio = con.T0 / temperature
    inv_ratio = ratio**-1
    return (
        10
        ** (
            10.79574 * (1 - ratio)
            - 5.028 * np.log10(inv_ratio)
            + 1.50475e-4 * (1 - (10 ** (-8.2969 * (inv_ratio - 1))))
            + 0.42873e-3 * (10 ** (4.76955 * (1 - ratio)) - 1)
            + 0.78614
        )
    ) * HPA_TO_P


def e2q(e, p):
    """
    Calculate the specific humidity from water vapour pressure and air pressure.

    Input:
    e is in Pa
    p is in Pa

    Output
    q in kg/kg
    """
    q = con.MW_RATIO * e / (p - (1 - con.MW_RATIO) * e)
    return q


def q2e(q, p):
    """
    Calculate water vapour pressure from the specific humidity and air pressure.

    Input:
    q in kg/kg
    p is in Pa

    Output
    e is in Pa
    """
    e = p / ((con.MW_RATIO / q) + 1 - con.MW_RATIO)
    return e


def rh2q(rh, T, p):
    """
    Calculate the specific humidity from relative humidity, air temperature,
    and pressure.

    Input:
    T is in K
    rh is in Pa/Pa
    p is in Pa

    Output
    q in kg/kg
    """

    eStar = calc_saturation_vapor_pressure(T)
    e = rh * eStar
    q = e2q(e, p)
    del e, eStar
    return q


def rh2a(rh, T):
    """
    Calculate the absolute humidity from relative humidity, air temperature,
    and pressure.

    Input T is in K
    rh is in Pa/Pa
    p is in Pa
    Output
    a in kg/m^3

    Source: Kraus: Chapter 8.1.2
    """

    if np.any(rh > 5):
        raise TypeError("rh must not be in %")

    e = rh * calc_saturation_vapor_pressure(T)
    a = e / (con.RW * T)
    return a


def moist_rho_rh(p, T, rh, *qm):
    """
    Input:
    p is in Pa
    T is in K
    rh is in Pa/Pa
    Optional, several possible:
    qm is in kg/kg other species which contribute to the air mass!
    (ice, snow, cloud etc.)

    Output:
    density of moist air [kg/m^3]

    Example:
    moist_rho_rh(p,T,rh,q_ice,q_snow,q_rain,q_cloud,q_graupel,q_hail)


    """

    q = rh2q(rh, T, p)

    return moist_rho_q(p, T, q, *qm)


def moist_rho_q(p, T, q, *qm):
    """
    Input p is in Pa
    T is in K
    q is in kg/kg
    Optional, several possible:
    qm is in kg/kg other species which contribute to the air mass!
    (ice, snow, cloud etc.)

    Output:
    density of moist air [kg/m^3]

    Example:
    moist_rho_q(p,T,q,q_ice,q_snow,q_rain,q_cloud,q_graupel,q_hail)
    """

    # if len(qm) > 0:
    #     # get rid of masked data!
    #     qm = np.ma.array(qm).filled(0)
    #     qm[qm < 0] = 0
    #     qm = np.sum(qm, axis=0)
    # else:
    #     qm = 0

    return p / (con.RS * T * (1 + (con.RW / con.RS - 1) * q - qm))


def rh_to_iwv(relhum_lev, temp_lev, press_lev, hgt_lev):
    """
    Calculate the integrated water vapour

    Input:
    T is in K
    rh is in Pa/Pa
    p is in Pa
    z is in m

    Output
    iwv in kg/m^2
    """
    dz = np.diff(hgt_lev, axis=-1)
    relhum = (relhum_lev[..., 0:-1] + relhum_lev[..., 1:]) / 2.0
    temp = (temp_lev[..., 0:-1] + temp_lev[..., 1:]) / 2.0

    xp = -1.0 * np.log(press_lev[..., 1:] / press_lev[..., 0:-1]) / dz
    press = -1.0 * press_lev[..., 0:-1] / xp * (np.exp(-xp * dz) - 1.0) / dz

    ql = rh2q(relhum, temp, press)
    rho_moist = moist_rho_q(press, temp, ql)

    return np.nansum(ql * rho_moist * dz)


def detect_liq_cloud(z, t, rh):
    # ***********
    # INPUT
    # z: height grid
    # T: temperature on z
    # rh: relative humidty on z
    # rh_thres: relative humidity threshold for the detection on liquid clouds on z
    # T_thres: don not detect liquid water clouds below this value (scalar)
    # ***********
    # OUTPUT
    # z_top: array of cloud tops
    # z_base: array of cloud bases
    # z_cloud: array of cloudy height levels
    # ***********

    rh_thres = 0.95  # 1
    t_thres = 253.15  # K
    n = len(z)
    # ***determine cloud boundaries
    # --> layers where mean rh GT rh_thres

    cloud_bound_ind = np.zeros(n, dtype=int)
    for i in np.arange(0, (n - 1)):
        if ((rh[i + 1] + rh[i]) / 2.0 > rh_thres) and (
            (t[i + 1] + t[i]) / 2.0 > t_thres
        ):
            cloud_bound_ind[i] = np.bitwise_or(1, cloud_bound_ind[i])
            cloud_bound_ind[i + 1] = np.bitwise_or(2, cloud_bound_ind[i + 1])
    i_cloud = np.where(cloud_bound_ind != 0)[0]

    # ***determine z_base & z_top arrays
    i_top = []
    i_base = []

    if len(i_cloud) != 0:
        i_base = np.where(cloud_bound_ind == 1)[0]
        i_top = np.where(cloud_bound_ind == 2)[0]
        n_base = len(i_base)
        n_top = len(i_top)
        if n_top != n_base:
            print("something wrong, number of bases NE number of cloud tops!")
            return [], [], []

    return z[i_top], z[i_base], z[i_cloud]


def adiab(i, T, P, z):
    """
    Adiabatic liquid water content assuming pseudo-adiabatic lapse rate
    throughout the whole cloud layer. Thus, the assumed temperature
    profile is different from the measured one
    Input:
    i no of levels
    T is in K
    p is in Pa
    z is in m
    Output:
    LWC
    """

    #   Set actual cloud base temperature to the measured one
    #   Initialize Liquid Water Content (LWC)
    #   Compute adiabatic LWC by integration from cloud base to level I

    TCL = T[0]
    LWC = 0.0

    for j in range(1, i + 1):
        deltaz = z[j] - z[j - 1]

        #   Compute actual cloud temperature

        #   1. Compute air density
        #   2. Compute water vapor density of saturated air
        #   3. Compute mixing ratio of saturated air
        #   4. Compute pseudoadiabatic lapse rate
        #   5. Compute actual cloud temperature

        R = moist_rho_rh(P[j], T[j], 1.0)
        RWV = rh2a(1.0, T[j])
        WS = RWV / (R - RWV)
        DTPS = pseudoAdiabLapseRate(T[j], WS)
        TCL = TCL + DTPS * (deltaz)

        #   Compute adiabatic LWC

        #   1. Compute air density
        #   2. Compute water vapor density of saturated air
        #   3. Compute mixing ratio of saturated air
        #   4. Compute specific heat of vaporisation
        #   5. Compute adiabatic LWC

        R = moist_rho_rh(P[j], TCL, 1.0)
        RWV = rh2a(1.0, TCL)
        WS = RWV / (R - RWV)
        L = spec_heat(TCL)

        LWC = LWC + (
            R
            * con.SPECIFIC_HEAT
            / L
            * ((con.g0 / con.SPECIFIC_HEAT) - pseudoAdiabLapseRate(TCL, WS))
            * deltaz
        )

    return LWC


def mod_ad(T_cloud, p_cloud, z_cloud):
    n_level = len(T_cloud)
    lwc = np.zeros(n_level - 1)

    thick = 0.0

    for jj in range(n_level - 1):
        deltaz = z_cloud[jj + 1] - z_cloud[jj]
        thick = deltaz + thick
        lwc[jj] = adiab(jj + 1, T_cloud, p_cloud, z_cloud)
        lwc[jj] = lwc[jj] * (-0.144779 * np.log(thick) + 1.239387)
    return lwc


def lwp_from_lwc(lwc, z):
    lwp = 0.0
    for i_z in range(len(lwc) - 1):
        lwp = lwp + lwc[i_z] * (z[i_z + 1] - z[i_z])

    return lwp


def pseudoAdiabLapseRate(T, Ws):
    """
    Pseudoadiabatic lapse rate
    Input: T   [K]  thermodynamic temperature
    Ws   [1]  mixing ratio of saturation
    Output: PSEUDO [K/m] pseudoadiabatic lapse rate
    Constants: Grav   [m/s2]     : constant of acceleration
        CP  [J/(kg K)]    : specific heat cap. at const. press
        Rair  [J/(kg K)]  : gas constant of dry air
        Rvapor [J/(kg K)] : gas constant of water vapor
    Source: Rogers and Yau, 1989: A Short Course in Cloud Physics
    (III.Ed.), Pergamon Press, 293p. Page 32
    translated to Python from pseudo1.pro by mx
    """

    # Compute specific humidity of vaporisation
    L = spec_heat(T)

    # Compute pseudo-adiabatic temperature gradient
    x = (
        (con.g0 / con.SPECIFIC_HEAT)
        * (1 + (L * Ws / con.RS / T))
        / (1 + (Ws * L**2 / con.SPECIFIC_HEAT / con.RW / T**2))
    )

    return x


def interp_log_p(p, z, z_int):
    p_interp = np.ones(len(z_int), np.float32) * -999.0
    for i_i, i_v in enumerate(z_int):
        for jjj in range(len(z) - 1):
            if (z[jjj] <= i_v) & (z[jjj + 1] >= i_v):
                if p[jjj + 1] < p[jjj]:
                    dz = z[jjj + 1] - z[jjj]
                    dz_new = i_v - z[jjj]
                    xp = -np.log(p[jjj + 1] / p[jjj]) / dz
                    p_interp[i_i] = p[jjj] * np.exp(-xp * dz_new)

    return p_interp * HPA_TO_P
