import os

import numpy as np

import mwrpy_ret.constants as con
from mwrpy_ret.utils import dcerror, loadCoeffsJSON


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

    # ****number of frequencies
    n_f = len(F)
    if RHO.all() <= 0:
        return np.zeros(n_f, np.float32)

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r22/h2o_list.json"
    CF = loadCoeffsJSON(path)
    CF["W0"] = CF["Wair"] / 1000.0
    CF["W0S"] = CF["Wself"] / 1000.0
    CF["SH"] = CF["Sair"] / 1000.0
    CF["SHS"] = CF["Sself"] / 1000.0
    CF["W2"] = CF["W2air"] / 1000.0
    CF["W2S"] = CF["W2self"] / 1000.0
    CF["D2"] = CF["D2air"] / 1000.0
    CF["D2S"] = CF["D2self"] / 1000.0
    CF["XW2"][CF["XW2"] <= 0.0] = CF["X"][CF["XW2"] <= 0.0]
    CF["XW2S"][CF["XW2S"] <= 0.0] = CF["XS"][CF["XW2S"] <= 0.0]
    CF["XH"][CF["XH"] <= 0.0] = CF["X"][CF["XH"] <= 0.0]
    CF["XHS"][CF["XHS"] <= 0.0] = CF["XS"][CF["XHS"] <= 0.0]

    PVAP = RHO * T * 4.615228e-3
    PDA = P - PVAP
    TI = CF["Trefcon"] / T
    CON = (
        (CF["Cf"] * PDA * TI ** CF["Xcf"] + CF["Cs"] * PVAP * TI ** CF["Xcs"])
        * PVAP
        * F**2.0
    )
    REFTLINE = 296.0
    TI = REFTLINE / T
    TILN = np.log(TI)
    TI2 = np.exp(2.5 * TILN)

    # ****ADD RESONANCES
    SUM = np.zeros(n_f, np.float32)
    for I, FL in enumerate(CF["FL"]):
        WIDTH0 = (
            CF["W0"][I] * PDA * TI ** CF["X"][I]
            + CF["W0S"][I] * PVAP * TI ** CF["XS"][I]
        )
        if CF["W2"][I] > 0.0:
            WIDTH2 = (
                CF["W2"][I] * PDA * TI ** CF["XW2"][I]
                + CF["W2S"][I] * PVAP * TI ** CF["XW2S"][I]
            )
        else:
            WIDTH2 = 0.0  # SUM = 0

        DELTA2 = CF["D2"][I] * PDA + CF["D2S"][I] * PVAP
        SHIFTF = CF["SH"][I] * PDA * (1.0 - CF["AAIR"][I] * TILN) * TI ** CF["XH"][I]
        SHIFTS = (
            CF["SHS"][I] * PVAP * (1.0 - CF["ASELF"][I] * TILN) * TI ** CF["XHS"][I]
        )
        SHIFT = SHIFTF + SHIFTS
        WSQ = WIDTH0**2

        S = CF["S1"][I] * TI2 * np.exp(CF["B2"][I] * (1.0 - TI))
        DF = np.zeros((2, n_f), np.float32)
        DF[0, :] = F - FL - SHIFT
        DF[1, :] = F + FL + SHIFT

        # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
        BASE = WIDTH0 / (562500.0 + WSQ)

        # DO FOR POSITIVE AND NEGATIVE RESONANCES
        RES = np.zeros(n_f, np.float32)
        for J in range(2):
            INDEX1 = np.zeros(n_f, dtype=bool)
            if (J == 0) & (WIDTH2 > 0.0):
                INDEX1 = np.abs(DF[J, :]) < 10.0 * WIDTH0
                XC = (WIDTH0 - 1.5 * WIDTH2 + (DF[J, :] + 1.5 * DELTA2) * 1j) / (
                    WIDTH2 - DELTA2 * 1j
                )
                XRT = np.sqrt(XC)
                PXW = 1.77245385090551603 * XRT * dcerror(-np.imag(XRT), np.real(XRT))
                SD = 2.0 * (1.0 - PXW) / (WIDTH2 - DELTA2 * 1j)
                RES[INDEX1] = (RES + np.real(SD) - BASE)[INDEX1]
            INDEX2 = (np.abs(DF[J, :]) < 750.0) & ~INDEX1
            if np.any(INDEX2):
                RES[INDEX2] = (RES + WIDTH0 / (DF[J, :] ** 2 + WSQ) - BASE)[INDEX2]

        SUM = SUM + S * RES * (F / FL) ** 2.0

    return 1.0e-10 * RHO * SUM / (np.pi * 2.9915075e-23) + CON


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

    # ****number of frequencies
    n_f = len(F)
    if RHO.all() <= 0:
        return np.zeros(n_f, np.float32)

    path = os.path.dirname(os.path.realpath(__file__)) + "/coeff/r23/h2o_list.json"
    CF = loadCoeffsJSON(path)
    CF["XW2"][CF["XW2"] <= 0.0] = CF["X"][CF["XW2"] <= 0.0]
    CF["XW2S"][CF["XW2S"] <= 0.0] = CF["XS"][CF["XW2S"] <= 0.0]

    PVAP = RHO * T * 4.615228e-3
    PDA = P - PVAP
    TI = 296.0 / T
    TILN = np.log(TI)
    TI2 = np.exp(2.5 * TILN)
    TI = CF["Trefcon"] / T

    ALPHA = H2OCON(F, T)
    CON = (CF["Cf"] * TI ** CF["Xcf"] * PDA + ALPHA * PVAP) * PVAP * F**2.0

    SUM = np.zeros(n_f, np.float32)
    for i, FL in enumerate(CF["FL"]):
        WIDTH0 = (
            CF["W0"][i] / 1000.0 * PDA * TI ** CF["X"][i]
            + CF["W0S"][i] / 1000.0 * PVAP * TI ** CF["XS"][i]
        )
        if CF["W2"][i] > 0.0:
            WIDTH2 = (
                CF["W2"][i] / 1000.0 * PDA * TI ** CF["XW2"][i]
                + CF["W2S"][i] / 1000.0 * PVAP * TI ** CF["XW2S"][i]
            )
        else:
            WIDTH2 = 0.0
        DELTA2 = CF["D2"][i] / 1000.0 * PDA + CF["D2S"][i] / 1000.0 * PVAP
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
        SHIFT = SHIFTF + SHIFTS
        S = CF["S1"][i] * TI2 * np.exp(CF["B2"][i] * (1.0 - TI))

        if np.abs(F - FL).any() < 750.1:
            DF = np.zeros((2, n_f), np.float32)
            DF[0, :] = F - FL - SHIFT
            DF[1, :] = F + FL + SHIFT
            WSQ = WIDTH0**2.0
            BASE = WIDTH0 / (562500.0 + WSQ)
            RES = np.zeros(n_f, np.float32)
            for j in range(2):
                INDEX1 = np.abs(DF[j, :]) < 10.0 * WIDTH0
                INDEX2 = (np.abs(DF[j, :]) < 750.0) & ~INDEX1
                if WIDTH2 > 0.0 and j == 0:
                    Xc = (
                        (WIDTH0 - 1.5 * WIDTH2) + ((DF[j, :] + 1.5 * DELTA2) * 1j)
                    ) / (WIDTH2 - DELTA2 * 1j)
                    Xrt = np.sqrt(Xc)
                    pxw = (
                        1.77245385090551603 * Xrt * dcerror(-np.imag(Xrt), np.real(Xrt))
                    )
                    SD = 2.0 * (1.0 - pxw) / (WIDTH2 - DELTA2 * 1j)
                    RES[INDEX1] = (RES + np.real(SD) - BASE)[INDEX1]
                else:
                    RES[INDEX2] = (RES + WIDTH0 / (DF[j, :] ** 2.0 + WSQ) - BASE)[
                        INDEX2
                    ]

            SUM = SUM + S * RES * (F / FL) ** 2.0

    return 1.0e-10 * RHO * SUM / (np.pi * 2.9915075e-23) + CON


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
    PE2 = DEN**2.0

    SUM = 1.584e-17 * FREQ**2.0 * DFNR / (TH * (FREQ**2.0 + DFNR**2.0))

    for K, FL in enumerate(CF["F"]):
        Y = DEN * (CF["Y0"][K] + CF["Y1"][K] * TH1)
        DNU = PE2 * (CF["DNU0"][K] + CF["DNU1"][K] * TH1)
        GFAC = 1.0 + PE2 * (CF["G0"][K] + CF["G1"][K] * TH1)
        DF = CF["W300"][K] * DEN
        STR = CF["S300"][K] * np.exp(-CF["BE"][K] * TH1)
        DEL1 = FREQ - FL - DNU
        DEL2 = FREQ + FL + DNU
        D1 = DEL1**2.0 + DF**2.0
        D2 = DEL2**2.0 + DF**2.0
        SF1 = (DF * GFAC + DEL1 * Y) / D1
        SF2 = (DF * GFAC - DEL2 * Y) / D2

        SUM = SUM + STR * (SF1 + SF2) * (FREQ / FL) ** 2.0

    ALPHA = 1.6097e11 * SUM * PRESDA * TH**3.0
    ALPHA[ALPHA < 0.0] = 0.0

    return ALPHA * 1.004  # increase absorption to match Koshelev2017


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
    PE2 = PE1**2.0
    WNR = CF["WB300"] * DEN

    A = CF["S300"] * np.exp(-CF["BE"] * TH1) / CF["F"] ** 2.0
    G = CF["G0"] + CF["G1"] * TH1
    OFF = np.sum(A[1:37] * G[1:37]) / np.sum(A[1:37])

    SUMM = (1.584e-17 / TH) * WNR / (FREQ**2.0 + WNR**2.0)
    for k, F in enumerate(CF["F"]):
        WIDTH = CF["W300"][k] * DEN
        Y = PE1 * (CF["Y300"][k] + CF["Y1"][k] * TH1)
        if k in range(1, 38):
            GFAC = 1.0 + PE2 * (G[k] - OFF)
        else:
            GFAC = 1.0
        FCEN = F + PE2 * (CF["DNU0"][k] + CF["DNU1"][k] * TH1)

        INDEX1 = np.zeros(len(FREQ), dtype=bool)
        if k == 0:
            INDEX1 = np.abs(FREQ - FCEN) < 10.0 * WIDTH
        INDEX2 = ~INDEX1
        SF1 = np.zeros(len(FREQ), np.float32)
        if np.any(INDEX1):
            WIDTH2 = 0.076 * WIDTH
            Xc = ((WIDTH - 1.5 * WIDTH2) + (FREQ[INDEX1] - FCEN) * 1j) / WIDTH2
            Xrt = np.sqrt(Xc)
            pxw = 1.77245385090551603 * Xrt * dcerror(-np.imag(Xrt), np.real(Xrt))
            ASD = (1.0 + Y * 1j) * 2.0 * (1.0 - pxw) / WIDTH2
            SF1[INDEX1] = np.real(ASD)
        else:
            SF1[INDEX2] = (WIDTH * GFAC + (FREQ[INDEX2] - FCEN) * Y) / (
                (FREQ[INDEX2] - FCEN) ** 2.0 + WIDTH**2.0
            )
        SF2 = (WIDTH * GFAC - (FREQ + FCEN) * Y) / ((FREQ + FCEN) ** 2.0 + WIDTH**2.0)

        SUMM = SUMM + A[k] * (SF1 + SF2)
        if k == 37:
            SUMM[SUMM < 0.0] = 0.0

    return 1.6097e11 * SUMM * PRESDA * FREQ**2.0 * TH**3.0


def ABN2_R(T, P, F):
    """
    ABSORPTION COEFFICIENT DUE TO NITROGEN IN AIR
    T = TEMPERATURE (K)
    P = PRESSURE (MB)
    F = FREQUENCY (GHZ)
    """
    TH = 300.0 / T
    FDEPEN = 0.5 + 0.5 / (1.0 + (F / 450.0) ** 2.0)

    return 9.95e-14 * FDEPEN * P**2.0 * F**2.0 * TH**3.22


def ABLIQ_R(LWC, F, T):
    """POWER ABSORPTION COEFFICIENT
    BY SUSPENDED CLOUD LIQUID WATER DROPLETS."""

    if LWC <= 0.0 or T < 233.0:
        return np.zeros(len(F), np.float32)

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
    f1 = 10.46012 + 0.1454962 * Tc + 6.3267156e-2 * Tc**2.0 + 9.3786645e-4 * Tc**3.0
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

    return -0.06286 * np.imag(RE) * F * LWC
