import numpy as np

import mwrpy_ret.constants as con
from mwrpy_ret.rad_trans import calc_absorption
from mwrpy_ret.utils import GAUSS, exponential_integration, read_config


def calc_mw_rt(
    # [m] states grid of T_final [K], p_final [Pa], q_final [kgm^-3]
    z_final,
    T_final,
    p_final,
    q_final,
    LWC,
    theta,  # zenith angle of observation in deg.
    f,  # frequency vector in GHz
    coeff_bdw: dict,
    ape_ang: np.ndarray,
    tau_k: np.ndarray | None = None,
    tau_v: np.ndarray | None = None,
):
    """
    non-scattering microwave radiative transfer
    """
    config = read_config(None, "global_specs")
    if (
        config["cloud"].split()[-1] == "excluded"
        and np.sum((LWC[1:] + LWC[:-1]) / 2.0 * np.diff(z_final)) > 0.001
        or np.any(np.ma.array(T_final).mask)
    ):
        return (
            np.ones(len(f), np.float32) * -999.0,
            np.ones(len(f), np.float32) * -999.0,
            np.ones(len(f), np.float32) * -999.0,
        )
    tau = np.ones((len(f), len(T_final)), np.float32) * -999.0
    if config["corr"].split()[0] == "Without":
        # Calculate optical thickness and TB
        for ind, freq in enumerate(f):
            tau[ind, :] = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, freq, config["model"], theta
            )
        TB = TB_CALC(f, T_final, tau[:, :])

    else:
        # Calculate optical thickness
        if tau_k is None:
            tau_k = np.ones((len(f[:7]), len(T_final)), np.float32) * -999.0
            for ind, freq in enumerate(f[:7]):
                tau_k[ind, :] = TAU_CALC(
                    z_final,
                    T_final,
                    p_final,
                    q_final,
                    LWC,
                    freq,
                    config["model"],
                    theta,
                )
        if tau_v is None:
            tau_v = (
                np.ones((len(coeff_bdw["f_all"]), len(T_final)), np.float32) * -999.0
            )
            for ind, freq in enumerate(coeff_bdw["f_all"]):
                tau_v[ind, :] = TAU_CALC(
                    z_final,
                    T_final,
                    p_final,
                    q_final,
                    LWC,
                    freq,
                    config["model"],
                    theta,
                )

        # Calculate TB
        TB = np.empty(len(f), np.float32)
        # Antenna beamwidth
        ape_wgh = GAUSS(ape_ang + theta, theta)
        ape_wgh = ape_wgh / np.sum(ape_wgh)
        ape_0 = ape_wgh[0] if "beamwidth" in config["corr"] else 1.0
        TB_k = TB_CALC(f[:7], T_final, tau_k) * ape_0
        for ff in range(7):
            fr_wgh = coeff_bdw["bdw_wgh"][
                ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0
            ] / np.sum(coeff_bdw["bdw_wgh"][ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0])
            TB_v = (
                np.sum(
                    TB_CALC(
                        coeff_bdw["f_all"][
                            coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]
                        ],
                        T_final,
                        tau_v[coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1], :],
                    )
                    * fr_wgh
                )
                * ape_0
            )
            if "beamwidth" in config["corr"]:
                for ia, _ in enumerate(ape_ang[1:]):
                    if ff == 0:
                        # K-band calculations
                        TB_k = np.vstack(
                            (
                                TB_k,
                                TB_CALC(
                                    f[:7],
                                    T_final,
                                    tau_k,
                                )
                                * ape_wgh[ia + 1],
                            )
                        )
                    # V-band calculations
                    TB_v = np.hstack(
                        (
                            TB_v,
                            np.sum(
                                TB_CALC(
                                    coeff_bdw["f_all"][
                                        coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][
                                            ff + 1
                                        ]
                                    ],
                                    T_final,
                                    tau_v[
                                        coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][
                                            ff + 1
                                        ],
                                        :,
                                    ],
                                )
                                * fr_wgh
                            )
                            * ape_wgh[ia + 1],
                        )
                    )
            TB[ff + 7] = np.sum(TB_v)
        TB[:7] = np.sum(TB_k, axis=0) if "beamwidth" in config["corr"] else TB_k

    return (
        TB,  # [K] brightness temperature array of f grid
        tau[:7, :],  # total optical depth (K-band)
        tau[7:, :],  # total optical depth (V-band, incl. bandwidth)
    )


def TAU_CALC(z, T, p, rhow, LWC, f, model, theta):
    """
    Calculate optical thickness tau at height k (index counting from bottom of zgrid)
    """
    mu = MU_CALC(z, T, p, rhow, theta)
    deltaz = np.diff(np.hstack([0.0, z * 1.0 / mu]))

    abs_wv = np.array(
        [
            eval(f"calc_absorption.ABWV_{model}")(
                rhow[ii] * 1000.0, TT, p[ii] / 100.0, f
            )
            / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )
    abs_o2 = np.array(
        [
            eval(f"calc_absorption.ABO2_{model}")(
                TT, p[ii] / 100.0, rhow[ii] * 1000.0, f
            )
            / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )
    abs_n2 = np.array(
        [
            calc_absorption.ABN2_R(TT, p[ii] / 100.0 - rhow[ii] * 1000.0, f) / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )
    abs_liq = np.array(
        [
            calc_absorption.ABLIQ_R(LWC[ii] * 1000.0, f, TT) / 1000.0
            for ii, TT in enumerate(T)
        ],
        np.float32,
    )

    _, tau_wv = exponential_integration(True, abs_wv, deltaz, 1, len(T), 1)
    _, tau_dry = exponential_integration(True, abs_o2 + abs_n2, deltaz, 1, len(T), 1)
    _, tau_liq = exponential_integration(False, abs_liq, deltaz, 1, len(T), 1)

    return tau_wv + tau_dry + tau_liq


def MU_CALC(
    z,  # height [m]
    T,  # Temp. [K]
    p,  # press. [Pa]
    rhow,  # abs. hum. [kg m^-3]
    theta,  # zenith angle [deg]
):
    mu = np.zeros(len(z), np.float64)
    deltas = np.zeros(len(z) - 1, np.float64)
    coeff = [77.695, 71.97, 3.75406]
    re = 6370950.0 + z[0]
    theta_bot = np.deg2rad(theta)
    r_bot = re
    T_top = (T[1:] + T[:-1]) / 2.0
    p_top = (p[1:] + p[:-1]) / 2.0
    e_top = (rhow[1:] + rhow[:-1]) / 2.0
    n_top = (
        1.0
        + (
            coeff[0] * (((p_top / 100.0) - e_top) / T_top)
            + coeff[1] * (e_top / T_top)
            + coeff[2] * (e_top / (T_top**2.0))
        )
        * 1e-6
    )
    n_bot = n_top
    n_bot[1:] = (
        1.0
        + (
            coeff[0] * (((p_top[1:] / 100.0) - e_top[1:]) / T_top[1:])
            + coeff[1] * (e_top[1:] / T_top[1:])
            + coeff[2] * (e_top[1:] / (T_top[1:] ** 2.0))
        )
        * 1e-6
    )
    deltaz = np.diff(z)

    for iz in range(len(z) - 1):
        r_top = r_bot + deltaz[iz]
        theta_top = np.arcsin(
            ((n_bot[iz] * r_bot) / (n_top[iz] * r_top)) * np.sin(theta_bot)
        )
        alpha = np.pi - theta_bot
        deltas[iz] = r_bot * np.cos(alpha) + np.sqrt(
            r_top**2.0 + r_bot**2.0 * (np.cos(alpha) ** 2.0 - 1.0)
        )
        mu[iz + 1] = deltaz[iz] / deltas[iz]
        theta_bot = theta_top
        r_bot = r_top

    return mu


def TB_CALC(frq: np.ndarray, t: np.ndarray, taulay: np.ndarray) -> np.ndarray:
    """calculate brightness temperatures without scattering
    adapted from pyrtlib
    """

    hvk = np.dot(frq * 1e9, con.h) / con.kB
    nl = len(t)
    tauprof = np.zeros(taulay.shape, np.float64)
    boftatm = np.zeros(taulay.shape, np.float64)
    boft = np.zeros(taulay.shape, np.float64)

    boft[:, 0] = 1.0 / (np.exp(hvk / t[0]) - 1.0)
    for i in range(1, nl):
        boft[:, i] = 1.0 / (np.exp(hvk / t[i]) - 1.0)
        boftlay = (boft[:, i - 1] + boft[:, i] * np.exp(-taulay[:, i])) / (
            1.0 + np.exp(-taulay[:, i])
        )
        batmlay = boftlay * np.exp(-tauprof[:, i - 1]) * (1.0 - np.exp(-taulay[:, i]))
        boftatm[:, i] = boftatm[:, i - 1] + batmlay
        tauprof[:, i] = tauprof[:, i - 1] + taulay[:, i]

    i_g = np.where(tauprof[:, nl - 1] >= 125.0)[0]
    i_l = np.where(tauprof[:, nl - 1] < 125.0)[0]
    boftotl = np.zeros(len(frq))
    if len(i_l) > 0:
        boftbg = 1.0 / (np.exp(hvk[i_l] / con.Tc) - 1.0)
        boftotl[i_l] = boftbg * np.exp(-tauprof[i_l, nl - 1]) + boftatm[i_l, nl - 1]
    if len(i_g) > 0:
        boftotl[i_g] = boftatm[i_g, nl - 1]

    return hvk / np.log(1.0 + (1.0 / boftotl))
