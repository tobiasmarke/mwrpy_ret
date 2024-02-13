import numpy as np

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import abshum_to_vap
from mwrpy_ret.rad_trans import calc_absorption
from mwrpy_ret.utils import GAUSS, read_config


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
    if config["corr"].split()[0] == "Without":
        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, f[:7], config["model"]
            )
        if tau_v is None:
            tau_v = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, f[7:], config["model"]
            )

        # Calculate TB
        mu = np.ones(len(z_final), np.float32) * np.cos(np.deg2rad(theta))
        TB = TB_CALC(T_final, np.hstack((tau_k, tau_v)), mu, f)

    else:
        # Antenna beamwidth
        ape_wgh = GAUSS(ape_ang + theta, theta)
        ape_wgh = ape_wgh / np.sum(ape_wgh)

        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, f[:7], config["model"]
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
            )

        # Calculate TB
        TB = np.empty(len(f), np.float32)
        mu = MU_CALC(z_final, T_final, p_final, q_final, theta + ape_ang[0])
        TB_k = TB_CALC(T_final, tau_k, mu, f[:7]) * ape_wgh[0]
        for ff in range(7):
            fr_wgh = coeff_bdw["bdw_wgh"][
                ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0
            ] / np.sum(coeff_bdw["bdw_wgh"][ff, coeff_bdw["bdw_wgh"][ff, :] > 0.0])
            TB_v = (
                np.sum(
                    TB_CALC(
                        T_final,
                        tau_v[:, coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]],
                        mu,
                        coeff_bdw["f_all"][
                            coeff_bdw["ind1"][ff] : coeff_bdw["ind1"][ff + 1]
                        ],
                    )
                    * fr_wgh
                )
                * ape_wgh[0]
            )

            for ia, aa in enumerate(ape_ang[1:]):
                # Refractive index
                mu = MU_CALC(z_final, T_final, p_final, q_final, theta + aa)
                if ff == 0:
                    # K-band calculations
                    TB_k = np.vstack(
                        (
                            TB_k,
                            TB_CALC(
                                T_final,
                                tau_k,
                                mu,
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
                                mu,
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
    model,
):
    """
    subroutine to determine optical thickness tau
    at height k (index counting from bottom of zgrid)
    """

    deltaz = np.diff(z)
    z = (z[1:] + z[:-1]) / 2.0
    T_mean = (T[1:] + T[:-1]) / 2.0
    rhow_mean = (rhow[1:] + rhow[:-1]) / 2.0 * 1000.0
    deltap = p[1:] - p[:-1]
    if np.any(deltap) >= 0.0:
        p_ind = np.where(deltap >= 0.0)[0]
        p[p_ind] = p[p_ind] - 0.1
        if np.any(deltap) > 1.0:
            print(
                "Warning: p profile adjusted to assure monotonic" "decrease!",
                deltap,
            )

    xp = -np.log(p[1:] / p[:-1])
    p_mean = -p[:-1] / xp * (np.exp(-xp) - 1.0) / 100.0

    kmax = len(z) - 1
    n_f = len(f)
    abs_all = np.zeros((len(z), n_f), np.float32)
    tau = np.zeros((len(z), n_f), np.float32)

    for ii in range(len(z)):
        # ****gas absorption
        # water vapor
        AWV = (
            eval("calc_absorption.ABWV_" + model)(
                rhow_mean[kmax - ii], T_mean[kmax - ii], p_mean[kmax - ii], f
            )
            / 1000.0
        )

        # oxygen
        AO2 = (
            eval("calc_absorption.ABO2_" + model)(
                T_mean[kmax - ii], p_mean[kmax - ii], rhow_mean[kmax - ii], f
            )
            / 1000.0
        )

        # nitrogen
        AN2 = calc_absorption.ABN2_R(T_mean[kmax - ii], p_mean[kmax - ii], f) / 1000.0

        # liquid water
        ABLIQ = (
            calc_absorption.ABLIQ_R(LWC[kmax - ii] * 1000.0, f, T_mean[kmax - ii])
            / 1000.0
        )

        absg = AWV + AO2 + AN2 + ABLIQ
        abs_all[kmax - ii, :] = absg
        tau[kmax - ii] = np.sum(abs_all * np.tile(deltaz, (n_f, 1)).T, axis=0)

    return tau


def MU_CALC(
    z,  # height [m]
    T,  # Temp. [K]
    p,  # press. [Pa]
    rhow,  # abs. hum. [kg m^-3]
    theta,  # zenith angle [deg]
):
    mu = np.zeros(len(z) - 1, np.float64)
    deltas = np.zeros(len(z) - 1, np.float64)
    coeff = [77.695, 71.97, 3.75406]
    re = 6370950.0 + z[0]
    e = abshum_to_vap(T, p, rhow)

    theta_bot = np.deg2rad(theta)
    r_bot = re
    T_top = (T[1:] + T[:-1]) / 2.0
    p_top = (p[1:] + p[:-1]) / 2.0
    e_top = (e[1:] + e[:-1]) / 2.0
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
        mu[iz] = deltaz[iz] / deltas[iz]
        theta_bot = theta_top
        r_bot = r_top

    return mu


def TB_CALC(T, tau, mu, freq):
    """
    calculate brightness temperatures without scattering
    according to Simmer (94) pp. 87 - 91 (alpha = 1, no scattering)
    """
    kmax = len(T) - 1
    n_f = len(freq)
    freq_si = freq * 1e9
    lamda_si = con.c / freq_si

    IN = np.zeros(n_f, dtype=np.float64) + 2.73
    IN = (
        (2.0 * con.h * freq_si / (lamda_si**2.0))
        * 1.0
        / (np.exp(con.h * freq_si / (con.kB * IN)) - 1.0)
    )

    tau_top = np.zeros(n_f, dtype=np.float64)
    tau_bot = tau[kmax - 1, :]
    for i in range(kmax):
        if i > 0:
            tau_top = tau[kmax - i, :]
            tau_bot = tau[kmax - 1 - i, :]

        if np.all(tau_bot - tau_top) > 0.0:
            T_pl2 = (
                (2.0 * con.h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(con.h * freq_si / (con.kB * T[kmax - 1 - i])) - 1.0)
            )
            T_pl1 = (
                (2.0 * con.h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(con.h * freq_si / (con.kB * T[kmax - i])) - 1.0)
            )
            delta_tau = tau_bot - tau_top
            diff = (T_pl2 - T_pl1) / delta_tau
            A = np.ones(n_f, dtype=np.float64) - np.exp(-delta_tau / mu[kmax - i - 1])
            B = delta_tau - mu[kmax - i - 1] * (
                1.0 - np.exp(-delta_tau / mu[kmax - i - 1])
            )
            IN = IN * np.exp(-delta_tau / mu[kmax - i - 1]) + T_pl1 * A + diff * B

    TB = (
        (con.h * freq_si / con.kB)
        * 1.0
        / np.log((2.0 * con.h * freq_si / (IN * lamda_si**2.0)) + 1.0)
    )

    return TB
