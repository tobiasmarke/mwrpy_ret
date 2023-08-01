import json
import os
import warnings

import numpy as np

import mwrpy_ret.constants as con


def STP_IM10(
    # [m] states grid of T_final [K], p_final [Pa], q_final [kgm^-3]
    z_final,
    T_final,
    p_final,
    q_final,
    LWC,
    theta,  # zenith angle of observation in deg.
    f,  # frequency vector in GHz
    # re-calculate opt. depth for every angle =0: no! (=1: yes is default),
    # tau_calc=True,
    # can save some time when calc. Jacobians
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

    theta = np.deg2rad(theta)
    mu = np.cos(theta) + 0.025 * np.exp(-11.0 * np.cos(theta))

    # ****radiative transfer
    tau, tau_wv, tau_o2 = TAU_CALC(z_final, T_final, p_final, q_final, LWC, f)

    TB = TB_CALC(T_final, tau, mu, f)

    return (
        TB,  # [K] brightness temperature array of f grid
        tau,  # total optical depth
        tau_wv,  # WV optical depth
        tau_o2,
    )


def loadCoeffsJSON(path) -> dict:
    """Load coefficients required for O2 absorption."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            try:
                var_all = dict(**json.load(f))
                for key in var_all.keys():
                    var_all[key] = np.asarray(var_all[key])
            except json.decoder.JSONDecodeError:
                print(path)
                raise
    return dict(**var_all)


def dcerror(x, y):
    """SIXTH-ORDER APPROX TO THE COMPLEX ERROR FUNCTION OF z=X+iY."""
    a = [
        122.607931777104326,
        214.382388694706425,
        181.928533092181549,
        93.155580458138441,
        30.180142196210589,
        5.912626209773153,
        0.564189583562615,
    ]
    b = [
        122.607931773875350,
        352.730625110963558,
        457.334478783897737,
        348.703917719495792,
        170.354001821091472,
        53.992906912940207,
        10.479857114260399,
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ZH = np.abs(y) - x * 1j
        ASUM = (
            ((((a[6] * ZH + a[5]) * ZH + a[4]) * ZH + a[3]) * ZH + a[2]) * ZH + a[1]
        ) * ZH + a[0]
        BSUM = (
            (((((ZH + b[6]) * ZH + b[5]) * ZH + b[4]) * ZH + b[3]) * ZH + b[2]) * ZH
            + b[1]
        ) * ZH + b[0]
        w = ASUM / BSUM
        w2 = 2.0 * np.exp(-((x + y * 1j) ** 2)) - np.conj(w)
        DCERROR = w
        DCERROR[y < 0] = w2[y < 0]
        return DCERROR


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
    abs_wv = np.zeros((kmax - 1, n_f))
    abs_o2 = np.zeros((kmax - 1, n_f))

    tau = np.zeros((kmax - 1, n_f))
    tau_wv = np.zeros((kmax - 1, n_f))
    tau_o2 = np.zeros((kmax - 1, n_f))

    for ii in range(kmax - 1):
        # alles SI!!
        deltaz = z[kmax - 1 - ii] - z[kmax - 1 - ii - 1]
        T_mean = (T[kmax - 1 - ii] + T[kmax - 1 - ii - 1]) / 2.0
        deltap = p[kmax - 1 - ii] - p[kmax - 1 - ii - 1]

        if deltap >= 0:
            p[kmax - 1 - ii] = p[kmax - 1 - ii] - 0.1
            if deltap >= 1:
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
        AWV = ABWV_R22(rhow_mean * 1000.0, T_mean, p_mean / 100, f)
        AWV = AWV / 1000.0

        # oxygen
        AO2 = ABO2_R22(T_mean, p_mean / 100.0, rhow_mean * 1000.0, f)
        AO2 = AO2 / 1000.0

        # nitrogen (Rosenkranz O2)
        AN2 = ABN2_R22(T_mean, p_mean / 100, f)
        AN2 = AN2 / 1000.0

        ALIQ = rewat_ellison(T_mean, f, LWC[kmax - 2 - ii])
        # ALIQ = abliq(LWC[kmax - 1 - ii], f, T_mean)
        # if W_mean > 0:
        #     import pdb
        #     pdb.set_trace()

        absg = AWV + AO2 + AN2 + ALIQ

        abs_all[kmax - 2 - ii, :] = absg
        abs_wv[kmax - 2 - ii, :] = AWV
        abs_o2[kmax - 2 - ii, :] = AO2

        tau_x = np.zeros(n_f)
        tau_x1 = np.zeros(n_f)
        tau_x2 = np.zeros(n_f)

        for jj in range(ii + 1):
            deltaz = z[kmax - 1 - jj] - z[kmax - 2 - jj]
            tau_x = (abs_all[kmax - 2 - jj, :]) * deltaz + tau_x
            tau_x1 = (abs_wv[kmax - 2 - jj, :]) * deltaz + tau_x1
            tau_x2 = (abs_o2[kmax - 2 - jj, :]) * deltaz + tau_x2

        tau[kmax - 2 - ii, :] = tau_x
        tau_wv[kmax - 2 - ii, :] = tau_x1
        tau_o2[kmax - 2 - ii, :] = tau_x2

    return tau, tau_wv, tau_o2  # total opt. depth  # WV opt. depth  # O2 opt. depth


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

    O2ABS = 1.6097e11 * SUM * PRESDA * TH**3
    O2ABS[O2ABS < 0] = 0
    O2ABS = O2ABS * 1.004  # increase absorption to match Koshelev2017

    return O2ABS


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


def abliq(LWC, F, T):
    """COMPUTES POWER ABSORPTION COEFFICIENT IN NEPERS/KM
    BY SUSPENDED CLOUD LIQUID WATER DROPLETS."""

    ABLIQ = np.zeros(len(F), np.float32)
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
        ABLIQ = -0.06286 * np.imag(RE) * F * LWC * 1000.0 * 1e-3 / con.c

    return ABLIQ


def rewat_ellison(T, F, LWC):
    """Return liquid water absoption coefficient according to ELLISON 2006.

    REFERENCES
    BOOK ARTICLE FROM WILLIAM ELLISON IN MAETZLER 2006 (p.431-455):
    THERMAL MICROWAVE RADIATION:
    APPLICATIONS FOR REMOTE SENSING IET ELECTROMAGNETIC WAVES SERIES 52
    ISBN: 978-086341-573-9"""
    # *** Convert Salinity from parts per thousand to SI
    salinity = 0.0
    Temp = T - con.T0

    # --------------------------------------------------------------------------
    # COEFFS AND CALCULATION OF eps(FREQ, Temp, SAL) according to (5.21, p.445)
    # --------------------------------------------------------------------------

    # *** Coefficients a_i (Table 5.5 or p. 454):
    a_1 = 0.46606917e-2
    a_2 = -0.26087876e-4
    a_3 = -0.63926782e-5
    a_4 = 0.63000075e1
    a_5 = 0.26242021e-2
    a_6 = -0.42984155e-2
    a_7 = 0.34414691e-4
    a_8 = 0.17667420e-3
    a_9 = -0.20491560e-6
    a_10 = 0.58366888e3
    a_11 = 0.12634992e3
    a_12 = 0.69227972e-4
    a_13 = 0.38957681e-6
    a_14 = 0.30742330e3
    a_15 = 0.12634992e3
    a_16 = 0.37245044e1
    a_17 = 0.92609781e-2
    a_18 = -0.26093754e-1

    # *** Calculate parameter functions (5.24)-(5.28), p.447
    EPS_S = 87.85306 * np.exp(
        -0.00456992 * Temp
        - a_1 * salinity
        - a_2 * salinity**2.0
        - a_3 * salinity * Temp
    )
    EPS_1 = a_4 * np.exp(-a_5 * Temp - a_6 * salinity - a_7 * salinity * Temp)
    tau_1 = (a_8 + a_9 * salinity) * np.exp(a_10 / (Temp + a_11)) * 1e-9
    tau_2 = (a_12 + a_13 * salinity) * np.exp(a_14 / (Temp + a_15)) * 1e-9
    EPS_INF = a_16 + a_17 * Temp + a_18 * salinity

    # *** Finally apply the interpolation formula (5.21)
    first_term = (EPS_S - EPS_1) / (1.0 + (-2.0 * np.pi * F * tau_1) * 1j)
    second_term = (EPS_1 - EPS_INF) / (1.0 + (-2.0 * np.pi * F * tau_2) * 1j)
    third_term = EPS_INF

    # third_term = EPS_INF
    EPS = first_term + second_term + third_term

    # *** compute absorption coefficients
    RE = (EPS - 1) / (EPS + 2)
    MASS_ABSCOF = 6.0 * np.pi * np.imag(RE) * F * 1e-3 / con.c
    ALIQ = MASS_ABSCOF * LWC * 1000.0

    return ALIQ


def TB_CALC(T, tau, mu_s, freq):
    """
    calculate brightness temperatures without scattering
    according to Simmer (94) pp. 87 - 91 (alpha = 1, no scattering)
    Planck/thermodynamic conform (28.05.03) # UL
    """
    kmax = len(T)
    n_f = len(freq)

    mu = np.zeros(n_f) + mu_s
    freq_si = freq * 1e9
    lamda_si = con.c / freq_si

    IN = np.zeros(n_f, dtype=np.float64) + 2.73
    IN = (
        (2.0 * con.h * freq_si / (lamda_si**2.0))
        * 1.0
        / (np.exp(con.h * freq_si / (con.kB * IN)) - 1.0)
    )

    tau_top = np.zeros(n_f, dtype=np.float64)
    tau_bot = tau[kmax - 2]
    for i in range(kmax - 1):
        valid = 1
        if i > 0:
            tau_top = tau[kmax - 2 - i + 1]
            tau_bot = tau[kmax - 2 - i]

        for ii in range(n_f):
            if tau_bot[ii] == tau_top[ii]:
                valid = 0
            if tau_bot[ii] < tau_top[ii]:
                valid = -1
        if valid == 0:
            print("warning, zero absorption coefficient")
        if valid == -1:
            print("warning, negative absorption coefficient")

        if valid == 1:
            delta_tau = tau_bot - tau_top
            A = np.ones(n_f, dtype=np.float64) - np.exp(-1 * delta_tau / mu)
            B = delta_tau - mu + mu * np.exp(-1 * delta_tau / mu)

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
            IN = IN * np.exp(-1 * delta_tau / mu) + T_pl1 * A + diff * B

    TB = (
        (con.h * freq_si / con.kB)
        * 1.0
        / np.log((2 * con.h * freq_si / (IN * lamda_si**2.0)) + 1.0)
    )

    return TB
