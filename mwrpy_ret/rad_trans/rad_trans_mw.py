import numpy as np

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import abshum_to_vap
from mwrpy_ret.rad_trans.calc_absorption import ABLIQ_R, ABN2_R, ABO2_R22, ABWV_R22
from mwrpy_ret.utils import GAUSS


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
    inst_corr = 0
    if inst_corr == 0:
        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f[:7])
        if tau_v is None:
            tau_v = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f[7:])

        mu = MU_CALC(z_final, T_final, p_final, q_final, theta)

        # Calculate TB
        TB = np.empty(len(f), np.float64)
        TB[:7] = TB_CALC(T_final, tau_k, mu, f[:7])
        TB[7:] = TB_CALC(T_final, tau_v, mu, f[7:])

    else:
        # Antenna beamwidth
        ape_wgh = GAUSS(ape_ang + theta, theta)
        ape_wgh = ape_wgh / np.sum(ape_wgh)

        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f[:7])
        if tau_v is None:
            tau_v = TAU_CALC(
                z_final, T_final, p_final, q_final, LWC, coeff_bdw["f_all"]
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
):
    """
    subroutine to determine optical thickness tau
    at height k (index counting from bottom of zgrid)
    """

    kmax = len(z)
    n_f = len(f)
    abs_all = np.zeros((kmax - 1, n_f))
    tau = np.zeros((kmax - 1, n_f))

    for ii in range(kmax - 1):
        deltaz = z[kmax - 1 - ii] - z[kmax - 1 - ii - 1]
        T_mean = (T[kmax - 1 - ii] + T[kmax - 1 - ii - 1]) / 2.0
        deltap = p[kmax - 1 - ii] - p[kmax - 1 - ii - 1]

        if deltap >= 0.0:
            p[kmax - 1 - ii] = p[kmax - 1 - ii] - 0.1
            if deltap > 1.0:
                print(
                    "Warning: p profile adjusted by %f5.2 to assure monotonic"
                    "decrease!",
                    deltap,
                )

        xp = -np.log(p[kmax - 1 - ii] / p[kmax - 1 - ii - 1]) / deltaz
        p_mean = -p[kmax - 1 - ii - 1] / xp * (np.exp(-xp * deltaz) - 1.0) / deltaz
        # p_mean = np.sqrt(p[kmax - 1 - ii] * p[kmax - 1 - ii - 1])
        rhow_mean = (rhow[kmax - 1 - ii] + rhow[kmax - 1 - ii - 1]) / 2.0

        # ****gas absorption
        # water vapor
        AWV = ABWV_R22(rhow_mean * 1000.0, T_mean, p_mean / 100.0, f) / 1000.0

        # oxygen
        AO2 = ABO2_R22(T_mean, p_mean / 100.0, rhow_mean * 1000.0, f) / 1000.0

        # nitrogen
        AN2 = ABN2_R(T_mean, p_mean / 100.0, f) / 1000.0

        # liquid water
        ABLIQ = ABLIQ_R(LWC[kmax - 2 - ii], f, T_mean) / 1000.0

        absg = AWV + AO2 + AN2 + ABLIQ
        abs_all[kmax - 2 - ii, :] = absg
        tau_x = np.zeros(n_f)

        for jj in range(ii + 1):
            deltaz = z[kmax - 1 - jj] - z[kmax - 2 - jj]
            tau_x = (abs_all[kmax - 2 - jj, :]) * deltaz + tau_x

        tau[kmax - 2 - ii, :] = tau_x

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

    for iz in range(1, len(z)):
        T_top = 0.5 * (T[iz] + T[iz - 1])
        p_top = 0.5 * (p[iz] + p[iz - 1])
        e_top = 0.5 * (e[iz] + e[iz - 1])
        n_top = (
            1.0
            + (
                coeff[0] * (((p_top / 100.0) - e_top) / T_top)
                + coeff[1] * (e_top / T_top)
                + coeff[2] * (e_top / (T_top**2.0))
            )
            * 1e-6
        )

        if iz > 1:
            T_bot = 0.5 * (T[iz - 1] + T[iz - 2])
            p_bot = 0.5 * (p[iz - 1] + p[iz - 2])
            e_bot = 0.5 * (e[iz - 1] + e[iz - 2])
            n_bot = (
                1.0
                + (
                    coeff[0] * (((p_bot / 100.0) - e_bot) / T_bot)
                    + coeff[1] * (e_bot / T_bot)
                    + coeff[2] * (e_bot / (T_bot**2.0))
                )
                * 1e-6
            )
        else:
            n_bot = n_top

        deltaz = z[iz] - z[iz - 1]
        r_top = r_bot + deltaz
        theta_top = np.arcsin(((n_bot * r_bot) / (n_top * r_top)) * np.sin(theta_bot))
        alpha = np.pi - theta_bot
        deltas[iz - 1] = r_bot * np.cos(alpha) + np.sqrt(
            r_top**2.0 + r_bot**2.0 * (np.cos(alpha) ** 2.0 - 1.0)
        )
        mu[iz - 1] = deltaz / deltas[iz - 1]
        theta_bot = theta_top
        r_bot = r_top

    return mu


def TB_CALC(T, tau, mu, freq):
    """
    calculate brightness temperatures without scattering
    according to Simmer (94) pp. 87 - 91 (alpha = 1, no scattering)
    """
    kmax = len(T)
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
    tau_bot = tau[kmax - 2, :]
    for i in range(kmax - 1):
        if i > 0:
            tau_top = tau[kmax - 2 - i + 1, :]
            tau_bot = tau[kmax - 2 - i, :]

        if np.all(tau_bot - tau_top) > 0.0:
            T_pl2 = (
                (2.0 * con.h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(con.h * freq_si / (con.kB * T[kmax - 2 - i])) - 1.0)
            )
            T_pl1 = (
                (2.0 * con.h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(con.h * freq_si / (con.kB * T[kmax - 1 - i])) - 1.0)
            )
            delta_tau = tau_bot - tau_top
            diff = (T_pl2 - T_pl1) / delta_tau
            A = np.ones(n_f, dtype=np.float64) - np.exp(-1.0 * delta_tau / mu[i])
            B = delta_tau - mu[i] + mu[i] * np.exp(-1.0 * delta_tau / mu[i])
            IN = IN * np.exp(-1.0 * delta_tau / mu[i]) + T_pl1 * A + diff * B

    TB = (
        (con.h * freq_si / con.kB)
        * 1.0
        / np.log((2.0 * con.h * freq_si / (IN * lamda_si**2.0)) + 1.0)
    )

    return TB
