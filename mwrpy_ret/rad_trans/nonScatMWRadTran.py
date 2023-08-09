import os

import numpy as np

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import abshum_to_vap
from mwrpy_ret.utils import dcerror, loadCoeffsJSON


def STP_IM10(
    # [m] states grid of T_final [K], p_final [Pa], q_final [kgm^-3]
    z_final,
    T_final,
    p_final,
    q_final,
    LWC,
    theta,  # zenith angle of observation in deg.
    f,  # frequency vector in GHz
    f_all: np.ndarray,
    ind1: np.ndarray,
    tau_k: np.ndarray | None = None,
    tau_v: np.ndarray | None = None,
):
    """
    non-scattering microwave radiative transfer using Rosenkranz 1998 gas
    absorption
    """

    z_final = np.asarray(z_final)
    T_final = np.asarray(T_final)
    p_final = np.asarray(p_final)
    q_final = np.asarray(q_final)
    f = np.asarray(f)

    # Antenna beamwidth
    ape_ini = np.linspace(-9.9, 9.9, 199)
    ape_ang = ape_ini[GAUSS(ape_ini, 0.0) > 1e-3]
    ape_ang = ape_ang[ape_ang >= 0.0]
    ape_wgh = GAUSS(ape_ang + theta, theta)
    ape_wgh = ape_wgh / np.sum(ape_wgh)

    # Channel bandwidth
    path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/coeff/o2_bandpass_interp_freqs.json"
    )
    FFI = loadCoeffsJSON(path)
    bdw_fre = FFI["FFI"].T
    path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/coeff/o2_bandpass_interp_norm_resp.json"
    )
    FRIN = loadCoeffsJSON(path)
    bdw_wgh = FRIN["FRIN"].T

    # Calculate optical thickness
    if tau_k is None:
        tau_k = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f[0:7])
    if tau_v is None:
        f_all, ind1 = np.empty(0, np.float32), np.zeros(1, np.int32)
        for ff in range(7):
            ifr = np.where(bdw_fre[ff, :] >= 0.0)[0]
            f_all = np.hstack((f_all, bdw_fre[ff, ifr]))
            ind1 = np.hstack((ind1, ind1[len(ind1) - 1] + len(ifr)))
        tau_v = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f_all)

    # Calculate TB
    TB = np.empty(len(f), np.float32)
    mu = MU_CALC(z_final, T_final, p_final, q_final, theta + ape_ang[0])
    TB_k = TB_CALC(T_final, tau_k, mu, f[0:7]) * ape_wgh[0]
    mu_org = MU_CALC(z_final, T_final, p_final, q_final, theta)
    for ff in range(7):
        TB_v = np.empty(0, np.float32)
        fr_wgh = bdw_wgh[ff, bdw_fre[ff, :] >= 0.0] / np.sum(
            bdw_wgh[ff, bdw_fre[ff, :] >= 0.0]
        )
        TB_org = np.sum(
            TB_CALC(
                T_final,
                tau_v[:, ind1[ff] : ind1[ff + 1]],
                mu_org,
                f_all[ind1[ff] : ind1[ff + 1]],
            )
            * fr_wgh
        )
        for ia, aa in enumerate(ape_ang):
            # Refractive index
            mu = MU_CALC(z_final, T_final, p_final, q_final, theta + aa)

            if (ff == 0) & (ia > 0):
                # K-band calculations
                TB_k = np.vstack(
                    (
                        TB_k,
                        TB_CALC(
                            T_final,
                            tau_k,
                            mu,
                            f[0:7],
                        )
                        * ape_wgh[ia],
                    )
                )
            # V-band calculations
            TB_v = np.hstack(
                (
                    TB_v,
                    np.sum(
                        TB_CALC(
                            T_final,
                            tau_v[:, ind1[ff] : ind1[ff + 1]],
                            mu,
                            f_all[ind1[ff] : ind1[ff + 1]],
                        )
                        * fr_wgh
                    )
                    * ape_wgh[ia],
                )
            )
        TB[ff + 7] = (np.sum(TB_v) + TB_org) / 2.0
    TB[0:7] = (np.sum(TB_k, axis=0) + TB_CALC(T_final, tau_k, mu_org, f[0:7])) / 2.0

    return (
        TB,  # [K] brightness temperature array of f grid
        tau_k,  # total optical depth (K-band)
        tau_v,  # total optical depth (V-band, incl. bandwidth)
        f_all,
        ind1,
    )


def GAUSS(ape_ang, theta):
    ape_sigma = (2.35 * 0.5) / np.sqrt(-1.0 * np.log(0.5))
    arg = np.abs((ape_ang - theta) / ape_sigma)
    arg = arg[arg < 9.0]

    return np.exp(-arg * arg / 2) * arg


def TAU_CALC(
    z,  # height [m]
    T,  # Temp. [K]
    p,  # press. [Pa]
    rhow,  # abs. hum. [kg m^-3]
    LWC,  # LWC [kg m^-3]
    f,  # freq. [GHz]
):
    """
    Abstract:
    subroutine to determine optical thickness tau
    at height k (index counting from bottom of zgrid)
    on the basis of Rosenkranz 1998 water vapor and absorption model
    Rayleigh calculations
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
        rhow_mean = (rhow[kmax - 1 - ii] + rhow[kmax - 1 - ii - 1]) / 2.0

        # ****gas absorption
        # water vapor
        AWV = ABWV_R22(rhow_mean * 1000.0, T_mean, p_mean / 100.0, f)
        AWV = AWV / 1000.0

        # oxygen
        AO2 = ABO2_R22(T_mean, p_mean / 100.0, rhow_mean * 1000.0, f)
        AO2 = AO2 / 1000.0

        # nitrogen
        AN2 = ABN2_R22(T_mean, p_mean / 100.0, f)
        AN2 = AN2 / 1000.0

        # liquid water
        ABLIQ = ABLIQ_R22(LWC[kmax - 2 - ii], f, T_mean)
        ABLIQ = ABLIQ / 1000.0

        absg = AWV + AO2 + AN2 + ABLIQ
        abs_all[kmax - 2 - ii, :] = absg
        tau_x = np.zeros(n_f)

        for jj in range(ii + 1):
            deltaz = z[kmax - 1 - jj] - z[kmax - 2 - jj]
            tau_x = (abs_all[kmax - 2 - jj, :]) * deltaz + tau_x

        tau[kmax - 2 - ii, :] = tau_x

    return tau


def ABWV_R22(
    RHO,  # abs. humidity in gm-3
    T,  # temp. in K
    P,  # pressure in hPa
    F,  # freqeuncy in GHz
):
    """
    OUPUT:
    ALPHA                        absorption coefficient in nepers(??)/km
    KEYWORDS:
     Abstract:
    PURPOSE- COMPUTE ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR

    CALLING SEQUENCE PARAMETERS-SPECIFICATIONS

          NAME    UNITS    I/O  DESCRIPTON            VALID RANGE
          T       KELVIN    I   TEMPERATURE
          P       MILLIBAR  I   PRESSURE              .1 TO 1000
          RHO     G/M**3    I   WATER VAPOR DENSITY
          F       GHZ       I   FREQUENCY             0 TO 800
          ALPHA   NEPERS/KM O   ABSORPTION COEFFICIENT

       REFERENCES-
       P.W. ROSENKRANZ, RADIO SCIENCE V.33, PP.919-928 (1998) V.34, P.1025
     (1999).

       LINE INTENSITIES SELECTION THRESHOLD=
         HALF OF CONTINUUM ABSORPTION AT 1000 MB.
       WIDTHS MEASURED AT 22, 183, 380 GHZ, OTHERS CALCULATED.
         A.BAUER ET AL.ASA WORKSHOP (SEPT. 1989) (380GHz).
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/h2o_list.json"
    CF = loadCoeffsJSON(path)

    # ****number of frequencies
    n_f = len(F)

    # ****LOCAL VARIABLES:
    NLINES = 16
    DF = np.zeros((2, n_f))

    if RHO.all() <= 0:
        ALPHA = np.zeros(n_f)
    else:
        PVAP = RHO * T * 4.615228e-3
        PDA = P - PVAP
        TI = 300.0 / T

        CON = (
            (5.9197e-10 * PDA * TI**3 + 1.4162e-8 * PVAP * TI**7.5) * PVAP * F**2
        )
        REFTLINE = 296.0
        TI = REFTLINE / T
        TILN = np.log(TI)
        TI2 = np.exp(2.5 * TILN)

        # ****ADD RESONANCES
        SUM = np.zeros(n_f)
        for I in range(NLINES):
            if np.abs(F - CF["FL"][I]).any() < 750.1:
                WIDTH0 = (CF["W0"][I] / 1000.0) * PDA * TI ** CF["X"][I] + (
                    CF["W0S"][I] / 1000.0
                ) * PVAP * TI ** CF["XS"][I]
                if CF["W2"][I] > 0:
                    WIDTH2 = (
                        CF["W2"][I] * PDA * TI ** CF["XW2"][I]
                        + (CF["W2S"][I] / 1000.0) * PVAP * TI ** CF["XW2S"][I]
                    )
                else:
                    WIDTH2 = 0  # SUM = 0

                DELTA2 = (CF["D2"][I] / 1000.0) * PDA + (CF["D2S"][I] / 1000.0) * PVAP
                SHIFTF = (
                    (CF["SH"][I] / 1000.0)
                    * PDA
                    * (1.0 - CF["AAIR"][I] * TILN)
                    * TI ** CF["XH"][I]
                )
                SHIFTS = (
                    CF["SHS"][I]
                    * PVAP
                    * (1.0 - CF["ASELF"][I] * TILN)
                    * TI ** CF["XHS"][I]
                )
                SHIFT = SHIFTF + SHIFTS
                WSQ = WIDTH0**2

                S = CF["S1"][I] * TI2 * np.exp(CF["B2"][I] * (1.0 - TI))
                DF[0, :] = F - CF["FL"][I] - SHIFT
                DF[1, :] = F + CF["FL"][I] + SHIFT

                # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
                BASE = WIDTH0 / (562500.0 + WSQ)

                # DO FOR POSITIVE AND NEGATIVE RESONANCES
                RES = np.zeros(n_f)
                for J in range(2):
                    if (J == 1) & (WIDTH2 > 0.0):
                        INDEX1 = np.abs(DF[J, :]) < 10.0 * WIDTH0
                        INDEX2 = (np.abs(DF[J, :]) < 750) & ~INDEX1
                        XC = ((WIDTH0 - 1.5 * WIDTH2) + (DF[J] + 1.5 * DELTA2) * 1j) / (
                            WIDTH2 - DELTA2 * 1j
                        )
                        XRT = np.sqrt(XC)
                        PXW = (
                            1.77245385090551603
                            * XRT
                            * dcerror(-np.imag(XRT), np.real(XRT))
                        )
                        SD = 2.0 * (1.0 - PXW) / (WIDTH2 - DELTA2 * 1j)
                        RES[INDEX1] = (RES + np.real(SD) - BASE)[INDEX1]
                        RES[INDEX2] = (RES + WIDTH0 / (DF[J] ** 2 + WSQ) - BASE)[INDEX2]
                    else:
                        INDEX = np.abs(DF[J, :]) < 750
                        RES[INDEX] = (RES + WIDTH0 / (DF[J] ** 2 + WSQ) - BASE)[INDEX]

                SUM = SUM + S * RES * (F / CF["FL"][I]) ** 2

        ALPHA = 1.0e-10 * RHO * SUM / (np.pi * 2.9915075e-23) + CON

    return ALPHA


def ABO2_R22(TEMP, PRES, VAPDEN, FREQ):
    """
    #
    PURPOSE: RETURNS ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR,
             IN NEPERS/KM
    #
     5/1/95  P. Rosenkranz
     11/5/97  P. Rosenkranz - 1- line modification.
     12/16/98 pwr - updated submm freq's and intensities from HITRAN96
    #
    ARGUMENTS:
    TEMP, PRES, VAPDEN, FREQ
    NAME    UNITS    DESCRIPTION        VALID RANGE
    #
    TEMP    KELVIN   TEMPERATURE        UNCERTAIN, but believed to be
                                         valid for atmosphere
    PRES   MILLIBARS PRESSURE           3 TO 1000
    VAPDEN  G/M^3   WATER VAPOR DENSITY  (ENTERS LINEWIDTH CALCULATION
                     DUE TO GREATER BROADENING EFFICIENCY OF H2O)
    FREQ    GHZ      FREQUENCY          0 TO 900
    #
    REFERENCES FOR EQUATIONS AND COEFFICIENTS:
    P.W. Rosenkranz, CHAP. 2 and appendix, in ATMOSPHERIC REMOTE SENSING
     BY MICROWAVE RADIOMETRY (M.A. Janssen, ed., 1993).
    H.J. Liebe et al, JQSRT V.48, PP.629-643 (1992).
    M.J. Schwartz, Ph.D. thesis, M.I.T. (1997).
    SUBMILLIMETER LINE INTENSITIES FROM HITRAN96.
    This version differs from Liebe's MPM92 in two significant respects:
    1. It uses the modification of the 1- line width temperature dependence
    recommended by Schwartz: (1/T).
    2. It uses the same temperature dependence (X) for submillimeter
    line widths as in the 60 GHz band: (1/T)^0.8

    LINES ARE ARRANGED 1-,1+,3-,3+,ETC. IN SPIN-ROTATION SPECTRUM
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/o2_list.json"
    CF = loadCoeffsJSON(path)

    WB300 = 0.56
    X = 0.754
    TH = 300.0 / TEMP
    TH1 = TH - 1.0
    B = TH**X
    PRESWV = VAPDEN * TEMP / 216.68
    PRESDA = PRES - PRESWV
    DEN = 0.001 * (PRESDA * B + 1.2 * PRESWV * TH)
    DFNR = WB300 * DEN
    PE2 = DEN * DEN

    SUM = 1.584e-17 * FREQ * FREQ * DFNR / (TH * (FREQ * FREQ + DFNR * DFNR))

    for K in range(len(CF["F"])):
        Y = DEN * (CF["Y0"][K] + CF["Y1"][K] * TH1)
        DNU = PE2 * (CF["DNU0"][K] + CF["DNU1"][K])
        GFAC = 1.0 + PE2 * (CF["G0"][K] + CF["G1"][K] * TH1)
        DF = CF["W300"][K] * DEN
        STR = CF["S300"][K] * np.exp(-CF["BE"][K] * TH1)
        DEL1 = FREQ - CF["F"][K] - DNU
        DEL2 = FREQ + CF["F"][K] + DNU
        D1 = DEL1 * DEL1 + DF * DF
        D2 = DEL2 * DEL2 + DF * DF
        SF1 = (DF * GFAC + DEL1 * Y) / D1
        SF2 = (DF * GFAC - DEL2 * Y) / D2

        SUM = SUM + STR * (SF1 + SF2) * (FREQ / CF["F"][K]) ** 2

    ALPHA = 1.6097e11 * SUM * PRESDA * TH**3
    ALPHA[ALPHA < 0] = 0
    ALPHA = ALPHA * 1.004  # increase absorption to match Koshelev2017

    return ALPHA


def ABN2_R22(T, P, F):
    """
    ****ABSN2 = ABSORPTION COEFFICIENT DUE TO NITROGEN IN AIR (NEPER/KM)
    T = TEMPERATURE (K)
    P = PRESSURE (MB)
    F = FREQUENCY (GHZ)
    """
    TH = 300.0 / T
    FDEPEN = 0.5 + 0.5 / (1.0 + (F / 450.0) ** 2.0)
    ALPHA = 9.95e-14 * FDEPEN * P * P * F * F * TH**3.22

    return ALPHA


def ABLIQ_R22(LWC, F, T):
    """COMPUTES POWER ABSORPTION COEFFICIENT IN NEPERS/KM
    BY SUSPENDED CLOUD LIQUID WATER DROPLETS."""

    ALPHA = np.zeros(len(F), np.float32)
    if T >= 233.0:
        Tc = T - con.T0
        theta = 300.0 / T
        z = 0.0 + F * 1j

        kappa = (
            -43.7527 * theta**0.05
            + 299.504 * theta**1.47
            - 399.364 * theta**2.11
            + 221.327 * theta**2.31
        )

        delta = 80.69715 * np.exp(-Tc / 226.45)
        sd = 1164.023 * np.exp(-651.4728 / (Tc + 133.07))
        kappa = kappa - delta * z / (sd + z)

        delta = 4.008724 * np.exp(-Tc / 103.05)
        hdelta = delta / 2.0
        f1 = (
            10.46012
            + 0.1454962 * Tc
            + 6.3267156e-02 * Tc**2
            + 9.3786645e-04 * Tc**3
        )
        z1 = (-0.75 + 1 * 1j) * f1
        z2 = -4500.0 + 2000.0 * 1j
        cnorm = np.log10(z2 / z1)
        chip = hdelta * np.log10((z - z2) / (z - z1)) / cnorm
        chij = hdelta * np.log10((z - np.conj(z2)) / (z - np.conj(z1))) / np.conj(cnorm)
        dchi = chip + chij - delta
        kappa = kappa + dchi

        RE = (kappa - 1.0) / (kappa + 2.0)
        ALPHA = -0.06286 * np.imag(RE) * F * LWC

    return ALPHA


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
                + coeff[2] * (e_top / (T_top**2))
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
                    + coeff[2] * (e_bot / (T_bot**2))
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
            r_top**2 + r_bot**2 * (np.cos(alpha) ** 2 - 1.0)
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
        valid = 1
        if i > 0:
            tau_top = tau[kmax - 2 - i + 1, :]
            tau_bot = tau[kmax - 2 - i, :]
        if n_f > 1:
            for ii in range(n_f):
                if tau_bot[ii] == tau_top[ii]:
                    valid = 0
                if tau_bot[ii] < tau_top[ii]:
                    valid = -1
        else:
            if tau_bot == tau_top:
                valid = 0
            if tau_bot < tau_top:
                valid = -1

        if valid == 1:
            delta_tau = tau_bot - tau_top
            A = np.ones(n_f, dtype=np.float64) - np.exp(-1 * delta_tau / mu[i])
            B = delta_tau - mu[i] + mu[i] * np.exp(-1 * delta_tau / mu[i])

            T_pl2 = (
                (2.0 * con.h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(con.h * freq_si / (con.kB * T[kmax - 2 - i])) - 1)
            )
            T_pl1 = (
                (2.0 * con.h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(con.h * freq_si / (con.kB * T[kmax - 1 - i])) - 1)
            )
            diff = (T_pl2 - T_pl1) / delta_tau
            IN = IN * np.exp(-1 * delta_tau / mu[i]) + T_pl1 * A + diff * B

    TB = (
        (con.h * freq_si / con.kB)
        * 1.0
        / np.log((2 * con.h * freq_si / (IN * lamda_si**2.0)) + 1.0)
    )

    return TB
