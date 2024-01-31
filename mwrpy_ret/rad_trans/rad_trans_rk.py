import os

import numpy as np

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import abshum_to_vap
from mwrpy_ret.utils import GAUSS, dcerror, loadCoeffsJSON


def RT_RK(
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
    non-scattering microwave radiative transfer using Rosenkranz 2023 gas
    absorption
    """
    inst_char = 1
    if inst_char == 0:
        # Calculate optical thickness
        if tau_k is None:
            tau_k = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f[:7])
        if tau_v is None:
            tau_v = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f[7:])

        mu = MU_CALC(z_final, T_final, p_final, q_final, theta)
        # Calculate TB
        TB = np.empty(len(f), np.float32)
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
    LWC,  # LWC [g m^-3]
    f,  # freq. [GHz]
):
    """
    subroutine to determine optical thickness tau
    at height k (index counting from bottom of zgrid)
    on the basis of Rosenkranz water vapor and absorption model
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
        # p_mean = np.sqrt(p[kmax - 1 - ii] * p[kmax - 1 - ii - 1])
        rhow_mean = (rhow[kmax - 1 - ii] + rhow[kmax - 1 - ii - 1]) / 2.0

        # ****gas absorption
        # water vapor
        AWV = ABWV_R23(rhow_mean * 1000.0, T_mean, p_mean / 100.0, f) / 1000.0

        # oxygen
        AO2 = ABO2_R23(T_mean, p_mean / 100.0, rhow_mean * 1000.0, f) / 1000.0

        # nitrogen
        AN2 = ABN2_R22(T_mean, p_mean / 100.0, f) / 1000.0

        # liquid water
        ABLIQ = ABLIQ_R22(LWC[kmax - 2 - ii], f, T_mean) / 1000.0

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
    ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR
      T       KELVIN    I   TEMPERATURE
      P       MILLIBAR  I   PRESSURE              .1 TO 1000
      RHO     G/M**3    I   WATER VAPOR DENSITY
      F       GHZ       I   FREQUENCY             0 TO 800
      ALPHA   NEPERS/KM O   ABSORPTION COEFFICIENT
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/h2o_list.json"
    CF = loadCoeffsJSON(path)

    # ****number of frequencies
    n_f = len(F)

    # ****LOCAL VARIABLES:
    NLINES = len(CF["FL"])
    DF = np.zeros((2, n_f))

    if RHO.all() <= 0:
        ALPHA = np.zeros(n_f)
    else:
        PVAP = RHO * T * 4.615228e-3
        PDA = P - PVAP
        TI = CF["Trefcon"] / T

        CON = (
            (CF["Cf"] * PDA * TI ** CF["Xcf"] + CF["Cs"] * PVAP * TI ** CF["Xcs"])
            * PVAP
            * F**2
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
                    WIDTH2 = (CF["W2"][I] / 1000.0) * PDA * TI ** CF["XW2"][I] + (
                        CF["W2S"][I] / 1000.0
                    ) * PVAP * TI ** CF["XW2S"][I]
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
                    (CF["SHS"][I] / 1000.0)
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
                    if (J == 0) & (WIDTH2 > 0.0):
                        INDEX1 = np.abs(DF[J, :]) < 10.0 * WIDTH0
                        INDEX2 = (np.abs(DF[J, :]) < 750) & ~INDEX1
                        XC = (WIDTH0 - 1.5 * WIDTH2 + (DF[J] + 1.5 * DELTA2) * 1j) / (
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


def ABWV_R23(
    RHO,  # abs. humidity in gm-3
    T,  # temp. in K
    P,  # pressure in hPa
    F,  # freqeuncy in GHz
):
    """
    ABSORPTION COEF IN ATMOSPHERE DUE TO WATER VAPOR
      T       KELVIN    I   TEMPERATURE
      P       MILLIBAR  I   PRESSURE              .1 TO 1000
      RHO     G/M**3    I   WATER VAPOR DENSITY
      F       GHZ       I   FREQUENCY             0 TO 800
      ALPHA   NEPERS/KM O   ABSORPTION COEFFICIENT
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r23/h2o_list.json"
    CF = loadCoeffsJSON(path)
    CF["XW2"][CF["XW2"] <= 0.0] = CF["X"][CF["XW2"] <= 0.0]
    CF["XW2S"][CF["XW2S"] <= 0.0] = CF["XS"][CF["XW2S"] <= 0.0]

    PVAP = RHO * T * 4.615228e-3
    PDA = P - PVAP
    TI = 296.0 / T
    TILN = np.log(TI)
    TI2 = np.exp(2.5 * TILN)

    NLINES = len(CF["FL"])
    WIDTH0, WIDTH2 = np.zeros(NLINES, np.float32), np.zeros(NLINES, np.float32)
    DELTA2, SHIFT = np.zeros(NLINES, np.float32), np.zeros(NLINES, np.float32)
    S = np.zeros(NLINES, np.float32)
    for i in range(NLINES):
        WIDTH0[i] = (
            CF["W0"][i] / 1000.0 * PDA * TI ** CF["X"][i]
            + CF["W0S"][i] / 1000.0 * PVAP * TI ** CF["XS"][i]
        )
        if CF["W2"][i] > 0.0:
            WIDTH2[i] = (
                CF["W2"][i] / 1000.0 * PDA * TI ** CF["XW2"][i]
                + CF["W2S"][i] / 1000.0 * PVAP * TI ** CF["XW2S"][i]
            )
        else:
            WIDTH2[i] = 0.0
        DELTA2[i] = CF["D2"][i] / 1000.0 * PDA + CF["D2S"][i] / 1000.0 * PVAP
        SHIFTF = (
            CF["SH"][i]
            / 1000.0
            * PDA
            * (1.0 - CF["AAIR"][i] * TILN)
            * TI ** CF["XH"][i]
        )
        SHIFTS = (
            CF["SHS"][i]
            / 1000.0
            * PVAP
            * (1.0 - CF["ASELF"][i] * TILN)
            * TI ** CF["XHS"][i]
        )
        SHIFT[i] = SHIFTF + SHIFTS
        S[i] = CF["S1"][i] * TI2 * np.exp(CF["B2"][i] * (1.0 - TI))

    TI = CF["Trefcon"] / T
    ALPHA = H2OCON(F, T)
    # CON = 1.4162E-8 * TI ** 7.5
    # ALPHA = np.ones(len(F), np.float32) * CON
    CONF = CF["Cf"] * TI ** CF["Xcf"]
    for k, FREQ in enumerate(F):
        SUMM = 0.0
        for i in range(NLINES):
            if np.abs(FREQ - CF["FL"][i]) <= 750.1:
                DF = np.zeros(2, np.float32)
                DF[0] = FREQ - CF["FL"][i] - SHIFT[i]
                DF[1] = FREQ + CF["FL"][i] + SHIFT[i]
                WSQ = WIDTH0[i] ** 2.0
                BASE = WIDTH0[i] / (562500.0 + WSQ)
                RES = 0.0
                for j in range(2):
                    if WIDTH2[i] > 0.0 and j == 0 and np.abs(DF[j]) < 10.0 * WIDTH0[i]:
                        Xc = (
                            (WIDTH0[i] - 1.5 * WIDTH2[i])
                            + ((DF[j] + 1.5 * DELTA2[i]) * 1j)
                        ) / (WIDTH2[i] - DELTA2[i] * 1j)
                        Xrt = np.sqrt(Xc)
                        pxw = (
                            1.77245385090551603
                            * Xrt
                            * dcerror(-np.imag(Xrt), np.real(Xrt))
                        )
                        SD = 2.0 * (1.0 - pxw) / (WIDTH2[i] - DELTA2[i] * 1j)
                        RES = RES + np.real(SD) - BASE
                    elif np.abs(DF[j]) < 750.0:
                        RES = RES + WIDTH0[i] / (DF[j] ** 2 + WSQ) - BASE

                SUMM = SUMM + S[i] * RES * (FREQ / CF["FL"][i]) ** 2.0

        CON = (CONF * PDA + ALPHA[k] * PVAP) * PVAP * FREQ**2.0
        ALPHA[k] = 1.0e-10 * RHO * SUMM / (np.pi * 2.9915075e-23) + CON

    return ALPHA


def H2OCON(F, T):
    SELFCON = [2.877e-21, 2.855e-21, 2.731e-21, 2.49e-21, 2.178e-21, 1.863e-21]
    SELFTEXP = [6.413, 6.414, 6.275, 6.049, 5.789, 5.557]

    THETA = 296.0 / T
    A = np.zeros(7, np.float32)
    for j in np.linspace(1, 6, 6, dtype=int):
        A[j] = 6.532e12 * SELFCON[j - 1] * THETA ** (SELFTEXP[j - 1] + 3.0)
    A[0] = A[2]

    CS = np.zeros(len(F), np.float32)
    for i, FREQ in enumerate(F):
        FJ = FREQ / 299.792458
        J = np.int32(np.min((FJ, 3)))
        P = FJ - np.float32(J)
        C = (3.0 - 2.0 * P) * P**2.0
        B = 0.5 * P * (1.0 - P)
        B1 = B * (1.0 - P)
        B2 = B * P
        CS[i] = (
            -A[J] * B1 + A[J + 1] * (1.0 - C + B2) + A[J + 2] * (C + B1) - A[J + 3] * B2
        )

    return CS


def ABO2_R22(TEMP, PRES, VAPDEN, FREQ):
    """
    ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR
    TEMP    KELVIN   TEMPERATURE        UNCERTAIN, but believed to be
                                         valid for atmosphere
    PRES   MILLIBARS PRESSURE           3 TO 1000
    VAPDEN  G/M^3    WV DENSITY         (ENTERS LINEWIDTH CALCULATION
                     DUE TO GREATER BROADENING EFFICIENCY OF H2O)
    FREQ    GHZ      FREQUENCY          0 TO 900
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/o2_list.json"
    CF = loadCoeffsJSON(path)

    TH = 300.0 / TEMP
    TH1 = TH - 1.0
    B = TH ** CF["X"]
    PRESWV = VAPDEN * TEMP / 216.68
    PRESDA = PRES - PRESWV
    DEN = 0.001 * (PRESDA * B + 1.2 * PRESWV * TH)
    DFNR = CF["WB300"] * DEN
    PE2 = DEN**2

    SUM = 1.584e-17 * FREQ**2 * DFNR / (TH * (FREQ**2 + DFNR**2))

    for K in range(len(CF["F"])):
        Y = DEN * (CF["Y0"][K] + CF["Y1"][K] * TH1)
        DNU = PE2 * (CF["DNU0"][K] + CF["DNU1"][K] * TH1)
        GFAC = 1.0 + PE2 * (CF["G0"][K] + CF["G1"][K] * TH1)
        DF = CF["W300"][K] * DEN
        STR = CF["S300"][K] * np.exp(-CF["BE"][K] * TH1)
        DEL1 = FREQ - CF["F"][K] - DNU
        DEL2 = FREQ + CF["F"][K] + DNU
        D1 = DEL1**2 + DF**2
        D2 = DEL2**2 + DF**2
        SF1 = (DF * GFAC + DEL1 * Y) / D1
        SF2 = (DF * GFAC - DEL2 * Y) / D2

        SUM = SUM + STR * (SF1 + SF2) * (FREQ / CF["F"][K]) ** 2

    ALPHA = 1.6097e11 * SUM * PRESDA * TH**3
    ALPHA[ALPHA < 0] = 0
    ALPHA = ALPHA * 1.004  # increase absorption to match Koshelev2017

    return ALPHA


def ABO2_R23(TEMP, PRES, VAPDEN, FREQ):
    """
    ABSORPTION COEFFICIENT DUE TO OXYGEN IN AIR
    TEMP    KELVIN   TEMPERATURE        UNCERTAIN, but believed to be
                                         valid for atmosphere
    PRES   MILLIBARS PRESSURE           3 TO 1000
    VAPDEN  G/M^3    WV DENSITY         (ENTERS LINEWIDTH CALCULATION
                     DUE TO GREATER BROADENING EFFICIENCY OF H2O)
    FREQ    GHZ      FREQUENCY          0 TO 900
    """

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r23/o2_list.json"
    CF = loadCoeffsJSON(path)

    TH = 300.0 / TEMP
    TH1 = TH - 1.0
    PRESWV = 4.615228e-3 * VAPDEN * TEMP
    PRESDA = PRES - PRESWV
    DEN = 0.001 * (PRESDA * TH ** CF["X"] + 1.2 * PRESWV * TH)
    PE1 = 0.99 * DEN
    PE2 = PE1**2
    WNR = CF["WB300"] * DEN

    SUMM, ANORM = 0.0, 0.0
    A, G = np.zeros(len(CF["F"]), np.float32), np.zeros(len(CF["F"]), np.float32)
    for k, F in enumerate(CF["F"]):
        A[k] = CF["S300"][k] * np.exp(-CF["BE"][k] * TH1) / F**2
        G[k] = CF["G0"][k] + CF["G1"][k] * TH1
        if k in range(2, 39):
            SUMM = SUMM + A[k] * G[k]
            ANORM = ANORM + A[k]
    OFF = SUMM / ANORM

    ALPHA = np.zeros(len(FREQ))
    for j, FJ in enumerate(FREQ):
        SUMM = (1.584e-17 / TH) * WNR / (FJ**2.0 + WNR**2.0)
        for k, F in enumerate(CF["F"]):
            WIDTH = CF["W300"][k] * DEN
            Y = PE1 * (CF["Y300"][k] + CF["Y1"][k] * TH1)
            if k in range(2, 39):
                GFAC = 1.0 + PE2 * (G[k] - OFF)
            else:
                GFAC = 1.0
            FCEN = F + PE2 * (CF["DNU0"][k] + CF["DNU1"][k] * TH1)

            if k == 1 and np.abs(FJ - FCEN) < 10.0 * WIDTH:
                WIDTH2 = 0.076 * WIDTH
                Xc = ((WIDTH - 1.5 * WIDTH2) + (FJ - FCEN) * 1j) / WIDTH2
                Xrt = np.sqrt(Xc)
                pxw = 1.77245385090551603 * Xrt * dcerror(-np.imag(Xrt), np.real(Xrt))
                ASD = (1.0 + Y * 1j) * 2.0 * (1.0 - pxw) / WIDTH2
                SF1 = np.real(ASD)
            else:
                SF1 = (WIDTH * GFAC + (FJ - FCEN) * Y) / (
                    (FJ - FCEN) ** 2.0 + WIDTH**2.0
                )

            SF2 = (WIDTH * GFAC - (FJ + FCEN) * Y) / ((FJ + FCEN) ** 2.0 + WIDTH**2.0)
            SUMM = SUMM + A[k] * (SF1 + SF2)
            if k == 37:
                SUMM = np.max((SUMM, 0.0))
        ALPHA[j] = 1.6097e11 * SUMM * PRESDA * FJ**2.0 * TH**3.0

    return ALPHA


def ABN2_R22(T, P, F):
    """
    ABSORPTION COEFFICIENT DUE TO NITROGEN IN AIR
    T = TEMPERATURE (K)
    P = PRESSURE (MB)
    F = FREQUENCY (GHZ)
    """
    TH = 300.0 / T
    FDEPEN = 0.5 + 0.5 / (1.0 + (F / 450.0) ** 2.0)
    ALPHA = 9.95e-14 * FDEPEN * P**2.0 * F**2.0 * TH**3.22

    return ALPHA


def ABLIQ_R22(LWC, F, T):
    """POWER ABSORPTION COEFFICIENT
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
            + 6.3267156e-02 * Tc**2.0
            + 9.3786645e-04 * Tc**3.0
        )
        z1 = (-0.75 + 1 * 1j) * f1
        z2 = -4500.0 + 2000.0 * 1j
        cnorm = np.log10(z2 / z1)
        chip = hdelta * np.log10((z - z2) / (z - z1)) / cnorm
        chij = hdelta * np.log10((z - np.conj(z2)) / (z - np.conj(z1))) / np.conj(cnorm)
        dchi = chip + chij - delta
        kappa = kappa + dchi

        # THETA1 = 1.-300./T
        # EPS0 = 77.66 - 103.3*THETA1
        # EPS1 = .0671*EPS0
        # EPS2 = 3.52
        # FP = 20.1*np.exp(7.88*THETA1)
        # FS = 39.8*FP
        # kappa = (EPS0-EPS1)/(1. + F/FP * 1j) + (EPS1-EPS2)/(1. + F/FS * 1j) + EPS2

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

    IN = np.ones(n_f, dtype=np.float64) * 2.73
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
