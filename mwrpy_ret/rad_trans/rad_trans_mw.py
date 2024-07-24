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
        and np.sum((LWC[1:] + LWC[:-1] / 2.0 * np.diff(z_final))) > 0.001
        or len(z_final) != 137
        or np.any(np.ma.array(T_final).mask)
    ):
        return (
            np.ones(len(f), np.float32) * -999.0,
            np.ones(len(f), np.float32) * -999.0,
            np.ones(len(f), np.float32) * -999.0,
        )
    if config["corr"].split()[0] == "Without":
        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, f[:7], config["model"], theta
            )
        if tau_v is None:
            tau_v = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, f[7:], config["model"], theta
            )

        # Calculate TB
        # mu = np.ones(len(z_final), np.float32) * np.cos(np.deg2rad(theta))
        TB = TB_CALC(f, T_final, np.hstack((tau_k, tau_v)))

    else:
        # Antenna beamwidth
        ape_wgh = GAUSS(ape_ang + theta, theta)
        ape_wgh = ape_wgh / np.sum(ape_wgh)

        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, f[:7], config["model"], theta
            )
        if tau_v is None:
            tau_v = TAU_CALC(
                z_final,
                T_final,
                p_final,
                q_final,
                LWC,
                coeff_bdw["f_all"],
                config["model"],
                theta,
            )

        # Calculate TB
        TB = np.empty(len(f), np.float32)
        # mu = MU_CALC(z_final, T_final, p_final, q_final, theta + ape_ang[0])
        TB_k = TB_CALC(T_final, tau_k, f[:7]) * ape_wgh[0]
        for ff in range(7):
            fr_wgh = coeff_bdw["bdw_wgh"][
                ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0
            ] / np.sum(coeff_bdw["bdw_wgh"][ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0])
            TB_v = (
                np.sum(
                    TB_CALC(
                        T_final,
                        tau_v[:, coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]],
                        coeff_bdw["f_all"][
                            coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]
                        ],
                    )
                    * fr_wgh
                )
                * ape_wgh[0]
            )

            for ia, _ in enumerate(ape_ang[1:]):
                # Refractive index
                # mu = MU_CALC(z_final, T_final, p_final, q_final, theta + aa)
                if ff == 0:
                    # K-band calculations
                    TB_k = np.vstack(
                        (
                            TB_k,
                            TB_CALC(
                                T_final,
                                tau_k,
                                f[:7],
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
                                T_final,
                                tau_v[
                                    :, coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]
                                ],
                                coeff_bdw["f_all"][
                                    coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]
                                ],
                            )
                            * fr_wgh
                        )
                        * ape_wgh[ia + 1],
                    )
                )
            TB[ff + 7] = np.sum(TB_v)
        TB[:7] = np.sum(TB_k, axis=0)

    return (
        TB,  # [K] brightness temperature array of f grid
        tau_k,  # total optical depth (K-band)
        tau_v,  # total optical depth (V-band, incl. bandwidth)
    )


def TAU_CALC(
    z,  # height [m]
    T,  # Temp. [K]
    p,  # press. [Pa]
    rhow,  # abs. hum. [kg m^-3]
    LWC,  # LWC [kg m^-3]
    f,  # freq. [GHz]
    model,  # absorption model
    theta,  # ang [deg]
):
    """
    subroutine to determine optical thickness tau
    at height k (index counting from bottom of zgrid)
    """

    air_m = 1 / np.sin(np.deg2rad((90.0 - theta)))
    deltaz = np.diff(np.hstack([0, z * air_m]))
    n_f = len(f)
    abs_all = np.zeros((len(T), n_f), np.float32)
    tau = np.zeros((len(T), n_f), np.float32)

    for ii, tem in enumerate(T):
        # ****gas absorption
        # water vapor
        AWV = (
            eval("calc_absorption.ABWV_" + model)(
                rhow[ii] * 1000.0, tem, p[ii] / 100.0, f
            )
            / 1000.0
        )

        # oxygen
        AO2 = (
            eval("calc_absorption.ABO2_" + model)(
                tem, p[ii] / 100.0, rhow[ii] * 1000.0, f
            )
            / 1000.0
        )

        # nitrogen
        AN2 = calc_absorption.ABN2_R(tem, p[ii] / 100.0 - rhow[ii] * 1000.0, f) / 1000.0

        # liquid water
        ABLIQ = calc_absorption.ABLIQ_R(LWC[ii] * 1000.0, f, tem) / 1000.0

        absg = AWV + AO2 + AN2 + ABLIQ
        abs_all[ii, :] = absg
    for j in range(n_f):
        _, tau[:, j] = exponential_integration(
            True, abs_all[:, j], deltaz, 0, len(T), 1
        )

    return tau


# def MU_CALC(
#     z,  # height [m]
#     T,  # Temp. [K]
#     p,  # press. [Pa]
#     rhow,  # abs. hum. [kg m^-3]
#     theta,  # zenith angle [deg]
# ):
#     mu = np.zeros(len(z) - 1, np.float64)
#     deltas = np.zeros(len(z) - 1, np.float64)
#     coeff = [77.695, 71.97, 3.75406]
#     re = 6370950.0 + z[0]
#     e = abshum_to_vap(T, p, rhow)
#
#     theta_bot = np.deg2rad(theta)
#     r_bot = re
#     T_top = (T[1:] + T[:-1]) / 2.0
#     p_top = (p[1:] + p[:-1]) / 2.0
#     e_top = (e[1:] + e[:-1]) / 2.0
#     n_top = (
#         1.0
#         + (
#             coeff[0] * (((p_top / 100.0) - e_top) / T_top)
#             + coeff[1] * (e_top / T_top)
#             + coeff[2] * (e_top / (T_top**2.0))
#         )
#         * 1e-6
#     )
#     n_bot = n_top
#     n_bot[1:] = (
#         1.0
#         + (
#             coeff[0] * (((p_top[1:] / 100.0) - e_top[1:]) / T_top[1:])
#             + coeff[1] * (e_top[1:] / T_top[1:])
#             + coeff[2] * (e_top[1:] / (T_top[1:] ** 2.0))
#         )
#         * 1e-6
#     )
#     deltaz = np.diff(z)
#
#     for iz in range(len(z) - 1):
#         r_top = r_bot + deltaz[iz]
#         theta_top = np.arcsin(
#             ((n_bot[iz] * r_bot) / (n_top[iz] * r_top)) * np.sin(theta_bot)
#         )
#         alpha = np.pi - theta_bot
#         deltas[iz] = r_bot * np.cos(alpha) + np.sqrt(
#             r_top**2.0 + r_bot**2.0 * (np.cos(alpha) ** 2.0 - 1.0)
#         )
#         mu[iz] = deltaz[iz] / deltas[iz]
#         theta_bot = theta_top
#         r_bot = r_top
#
#     return mu


def TB_CALC(freq: np.ndarray, T: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """calculate brightness temperatures without scattering
    adapted from pyrtlib
    """

    f = freq * 1e9
    hvk = f * con.h / con.kB

    nl, nf = len(T), len(freq)
    TAUP = np.zeros((nl, nf), np.float32)
    BTATM = np.zeros((nl, nf), np.float32)
    BT = np.zeros((nl, nf), np.float32)

    BT[0, :] = 1.0 / (np.exp(hvk / T[0]) - 1.0)
    for i in range(1, nl):
        BT[i, :] = 1.0 / (np.exp(hvk / T[i]) - 1.0)
        boftlay = (BT[i - 1, :] + BT[i, :] * np.exp(-tau[i, :])) / (
            1.0 + np.exp(-tau[i, :])
        )
        batmlay = boftlay * np.exp(-TAUP[i - 1, :]) * (1.0 - np.exp(-tau[i, :]))
        BTATM[i, :] = BTATM[i - 1, :] + batmlay
        TAUP[i, :] = TAUP[i - 1, :] + tau[i, :]

    bth = np.where(TAUP[nl - 1, :] < 125.0)[0]
    ath = np.where(TAUP[nl - 1, :] >= 125.0)[0]
    boftotl = np.zeros(nf, np.float32)
    if any(bth):
        boftbg = 1.0 / (np.exp(hvk[bth] / con.Tc) - 1.0)
        bakgrnd = boftbg * np.exp(-TAUP[nl - 1, bth])
        boftotl[bth] = bakgrnd + BTATM[nl - 1, bth]
    if any(ath):
        boftotl[ath] = BTATM[nl - 1, ath]

    return hvk / np.log(1.0 + (1.0 / boftotl))
