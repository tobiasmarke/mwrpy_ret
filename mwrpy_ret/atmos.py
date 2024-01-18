"""Module for atmsopheric functions."""
import os

import numpy as np
import pandas as pd

import mwrpy_ret.constants as con

HPA_TO_P = 100


def spec_heat(T: np.ndarray) -> np.ndarray:
    """Specific heat for evaporation (J/kg)"""
    return con.LATENT_HEAT - 2420.0 * (T - con.T0)


def abs_hum(T: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Absolute humidity (kg/m^3)"""
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


def abshum_to_vap(T, p, rho):
    e = rho * con.RW * T
    m = con.MW_RATIO * e / (p - e) / 1000.0
    return p * m / (con.MW_RATIO + m)


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

    e = rh * calc_saturation_vapor_pressure(T)
    return e / (con.RW * T)


def moist_rho_rh(p, T, rh):
    """
    Input:
    p is in Pa
    T is in K
    rh is in Pa/Pa

    Output:
    density of moist air [kg/m^3]

    Example:
    moist_rho_rh(p,T,rh,q_ice,q_snow,q_rain,q_cloud,q_graupel,q_hail)
    """

    eStar = calc_saturation_vapor_pressure(T)
    e = rh * eStar
    q = con.MW_RATIO * e / (p - (1 - con.MW_RATIO) * e)

    return p / (con.RS * T * (1 + (con.RW / con.RS - 1) * q))


def rh_to_iwv(T, rh, height):
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

    absh = abs_hum(T, rh)
    iwv = np.sum((absh[:-1] + absh[1:]) / 2.0 * np.diff(height))

    return iwv


def detect_liq_cloud(z, t, rh, p_rs):
    """
    # INPUT
    # z: height grid
    # T: temperature on z
    # rh: relative humidty on z
    # rh_thres: relative humidity threshold for the detection on liquid clouds on z
    # T_thres: do not detect liquid water clouds below this value (scalar)
    # ***********
    # OUTPUT
    # z_top: array of cloud tops
    # z_base: array of cloud bases
    # z_cloud: array of cloudy height levels
    """

    alpha = 0.59
    beta = 1.37
    sigma = p_rs / p_rs[0]
    rh_thres = 1.0 - alpha * sigma * (1.0 - sigma) * (1.0 + beta * (sigma - 0.5))
    # rh_thres = 0.95  # 1
    t_thres = 253.15  # K
    # ***determine cloud boundaries
    # --> layers where mean rh GT rh_thres

    i_cloud, i_top, i_base = (
        np.where((rh > rh_thres) & (t > t_thres))[0],
        np.empty(0, np.int32),
        np.empty(0, np.int32),
    )
    if len(i_cloud) > 1:
        i_base = np.unique(
            np.hstack((i_cloud[0], i_cloud[np.diff(np.hstack((0, i_cloud))) > 1]))
        )
        i_top = np.hstack(
            (i_cloud[np.diff(np.hstack((i_cloud, 0))) > 1] - 1, i_cloud[-1])
        )

        if len(i_top) != len(i_base):
            print("something wrong, number of bases NE number of cloud tops!")
            return [], []

    return z[i_top], z[i_base]


def detect_cloud_mod(z, lwc):
    """detect liquid cloud boundaries from model"""
    i_cloud, i_top, i_base = (
        np.where(lwc > 0.0)[0],
        np.empty(0, np.int32),
        np.empty(0, np.int32),
    )
    if len(i_cloud) > 1:
        i_base = np.unique(
            np.hstack((i_cloud[0], i_cloud[np.diff(np.hstack((0, i_cloud))) > 1]))
        )
        i_top = np.hstack(
            (i_cloud[np.diff(np.hstack((i_cloud, 0))) > 1] - 1, i_cloud[-1])
        )

        if len(i_top) != len(i_base):
            print("something wrong, number of bases NE number of cloud tops!")
            return [], []

    return z[i_top], z[i_base]


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
        TCL = TCL - DTPS * (deltaz)

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
    cloud_new = np.zeros(n_level - 1)

    thick = 0.0
    for jj in range(n_level - 1):
        deltaz = z_cloud[jj + 1] - z_cloud[jj]
        thick = deltaz + thick
        lwc[jj] = adiab(jj + 1, T_cloud, p_cloud, z_cloud)
        lwc[jj] = lwc[jj] * (-0.144779 * np.log(thick) + 1.239387)
        cloud_new[jj] = z_cloud[jj] + deltaz / 2.0
    return lwc, cloud_new


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


def get_cloud_prop(
    base: np.ndarray,
    top: np.ndarray,
    height_int: np.ndarray,
    input_dat: dict,
    method: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    lwc, lwp, height_new, cloud_new, lwc_new = (
        np.empty(0, np.float64),
        0.0,
        np.empty(0, np.float64),
        np.empty(0, np.float64),
        np.empty(0, np.float64),
    )

    for icl, _ in enumerate(top):
        xcl = np.where(
            (input_dat["height"][:] >= base[icl]) & (input_dat["height"][:] <= top[icl])
        )[0]
        if len(xcl) > 1:
            if method == "prognostic":
                lwcx, cloudx = input_dat["lwc"][xcl], input_dat["height"][xcl]
                lwp = lwp + np.sum(
                    lwcx * np.diff(input_dat["height"][xcl[0] : xcl[-1] + 2])
                )
            else:
                lwcx, cloudx = mod_ad(
                    input_dat["air_temperature"][xcl],
                    input_dat["air_pressure"][xcl],
                    input_dat["height"][xcl],
                )
                lwp = lwp + np.sum(lwcx * np.diff(input_dat["height"][xcl]))

            cloud_new = np.hstack((cloud_new, cloudx))
            lwc = np.hstack((lwc, lwcx))

            if len(height_new) == 0:
                height_new = height_int[height_int < base[0]]
            else:
                height_new = np.hstack(
                    (
                        height_new,
                        height_int[
                            (height_int > top[icl - 1]) & (height_int < base[icl])
                        ],
                    )
                )

            # New vertical grid
            height_new = np.hstack((height_new, height_int[height_int > top[-1]]))
            height_new = np.sort(np.hstack((height_new, cloud_new)))

            # Distribute liquid water
            lwc_new = np.zeros(len(height_new) - 1, np.float32)
            if len(lwc) > 0:
                _, xx, yy = np.intersect1d(
                    height_new, cloud_new, assume_unique=False, return_indices=True
                )
                lwc_new[xx] = lwc[yy]

    return height_new, lwc_new, lwp


def interp_log_p(p, z, z_int):
    return np.power(10.0, np.interp(np.log10(z_int), np.log10(z), np.log10(p)))


def era5_geopot(level, ps, gpot, temp, hum) -> tuple[np.ndarray, np.ndarray]:
    file_mh = (
        os.path.dirname(os.path.realpath(__file__))
        + "/rad_trans/coeff/era5_model_levels_137.csv"
    )
    mod_lvl = pd.read_csv(file_mh)
    a_cf = mod_lvl["a [Pa]"].values[:].astype("float")
    b_cf = mod_lvl["b"].values[:].astype("float")
    z_f = np.empty(len(level), np.float32)

    p_h = a_cf + b_cf * ps
    pres = (p_h + np.roll(p_h, 1, axis=0))[1:] / 2

    for lev in sorted(level, reverse=True):
        i_z = np.where(level == lev)[0]
        p_l = a_cf[lev - 1] + (b_cf[lev - 1] * ps)
        p_lp = a_cf[lev] + (b_cf[lev] * ps)

        if lev == 1:
            dlog_p = np.log(p_lp / 0.1)
            alpha = np.log(2)
        else:
            dlog_p = np.log(p_lp / p_l)
            alpha = 1.0 - ((p_l / (p_lp - p_l)) * dlog_p)

        temp[i_z] = (temp[i_z] * (1.0 + 0.609133 * hum[i_z])) * con.RS
        z_f[i_z] = gpot + (temp[i_z] * alpha)
        gpot = gpot + (temp[i_z] * dlog_p)

    return np.flip(z_f), np.flip(pres)
