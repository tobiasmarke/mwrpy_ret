# coding: utf-8
import numba
import numpy as np


def STP_IM10(
    # [m] states grid of T_final [K], p_final [Pa], q_final [kgm^-3]
    z_final,
    T_final,
    p_final,
    q_final,
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
    tau, tau_wv, tau_o2 = TAU_CALC_IM10(z_final, T_final, p_final, q_final, f)

    TB = TB_CALC_PL_IM10(T_final, tau, mu, f)

    return (
        TB,  # [K] brightness temperature array of f grid
        tau,  # total optical depth
        tau_wv,  # WV optical depth
        tau_o2,
    )


@numba.jit(cache=True, nopython=True)
def TAU_CALC_IM10(
    z,  # height [m]
    T,  # Temp. [K]
    p,  # press. [Pa]
    rhow,  # abs. hum. [kg m^-3]
    f,  # freq. [GHz]
):
    """
    Abstract:
    subroutine to determine optical thichkness tau
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
        AWV = ABWVR98_IM10(rhow_mean * 1000.0, T_mean, p_mean / 100, f)
        AWV = AWV / 1000.0

        # oxygen
        AO2 = ABO2R98_IM10(T_mean, p_mean / 100.0, rhow_mean * 1000.0, f)
        AO2 = AO2 / 1000.0

        # nitrogen (Rosenkranz O2)
        AN2 = ABSN2_IM10(T_mean, p_mean / 100, f)
        AN2 = AN2 / 1000.0

        absg = AWV + AO2 + AN2

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


@numba.jit(cache=True, nopython=True)
def ABWVR98_IM10(
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

    # ****number of frequencies
    n_f = len(F)

    # ****LOCAL VARIABLES:
    NLINES = 15
    DF = np.zeros((2, n_f))

    # ****LINE FREQUENCIES:
    FL = [
        22.2351,
        183.3101,
        321.2256,
        325.1529,
        380.1974,
        439.1508,
        443.0183,
        448.0011,
        470.8890,
        474.6891,
        488.4911,
        556.9360,
        620.7008,
        752.0332,
        916.1712,
    ]

    # ****LINE INTENSITIES AT 300K:
    S1 = [
        0.1310e-13,
        0.2273e-11,
        0.8036e-13,
        0.2694e-11,
        0.2438e-10,
        0.2179e-11,
        0.4624e-12,
        0.2562e-10,
        0.8369e-12,
        0.3263e-11,
        0.6659e-12,
        0.1531e-08,
        0.1707e-10,
        0.1011e-08,
        0.4227e-10,
    ]

    # ****T COEFF. OF INTENSITIES:
    B2 = [
        2.144,
        0.668,
        6.179,
        1.541,
        1.048,
        3.595,
        5.048,
        1.405,
        3.597,
        2.379,
        2.852,
        0.159,
        2.391,
        0.396,
        1.441,
    ]

    # ****T-EXPONENT OF AIR-BROADENING:
    X = [
        0.69,
        0.64,
        0.67,
        0.68,
        0.54,
        0.63,
        0.60,
        0.66,
        0.66,
        0.65,
        0.69,
        0.69,
        0.71,
        0.68,
        0.70,
    ]

    # ****AIR-BROADENED WIDTH PARAMETERS AT 300K:
    W3 = [
        0.002656,
        0.00281,
        0.0023,
        0.00278,
        0.00287,
        0.0021,
        0.00186,
        0.00263,
        0.00215,
        0.00236,
        0.0026,
        0.00321,
        0.00244,
        0.00306,
        0.00267,
    ]

    # ****SELF-BROADENED WIDTH PARAMETERS AT 300K
    WS = [
        0.0127488,
        0.01491,
        0.0108,
        0.0135,
        0.01541,
        0.0090,
        0.00788,
        0.01275,
        0.00983,
        0.01095,
        0.01313,
        0.01320,
        0.01140,
        0.01253,
        0.01275,
    ]

    # ****T-EXPONENT OF SELF-BROADENING:
    XS = [
        0.61,
        0.85,
        0.54,
        0.74,
        0.89,
        0.52,
        0.50,
        0.67,
        0.65,
        0.64,
        0.72,
        1.0,
        0.68,
        0.84,
        0.78,
    ]

    if RHO <= 0:
        ALPHA = np.zeros(n_f)
    else:
        PVAP = RHO * T / 217.0
        PDA = P - PVAP
        DEN = 3.335e16 * RHO
        TI = 300.0 / T
        TI2 = TI**2.5

        # ****CONTINUUM TERMS
        bf_org = 5.43e-10
        bs_org = 1.8e-8

        bf_mult = 1.105
        bs_mult = 0.79

        bf = bf_org * bf_mult
        bs = bs_org * bs_mult

        CON = (bf * PDA * TI**3 + bs * PVAP * TI**7.5) * PVAP * F * F

        # ****ADD RESONANCES
        SUM = np.zeros(n_f)

        for I in range(NLINES):
            WIDTH = W3[I] * PDA * TI ** X[I] + WS[I] * PVAP * TI ** XS[I]
            WSQ = WIDTH * WIDTH
            S = S1[I] * TI2 * np.exp(B2[I] * (1.0 - TI))
            DF[0, :] = F - FL[I]
            DF[1, :] = F + FL[I]

            # USE CLOUGH'S DEFINITION OF LOCAL LINE CONTRIBUTION
            BASE = WIDTH / (562500.0 + WSQ)

            # DO FOR POSITIVE AND NEGATIVE RESONANCES
            RES = np.zeros(n_f)

            for i_n_f in range(n_f):
                for J in range(2):
                    if np.abs(DF[J, i_n_f]) < 750.0:
                        RES[i_n_f] = (
                            RES[i_n_f] + WIDTH / (DF[J, i_n_f] ** 2 + WSQ) - BASE
                        )

            SUM = SUM + S * RES * (F / FL[I]) ** 2

        ALPHA = 0.3183e-4 * DEN * SUM + CON

    return ALPHA


@numba.jit(cache=True, nopython=True)
def ABO2R98_IM10(TEMP, PRES, VAPDEN, FREQ):
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
    F = [
        118.7503,
        56.2648,
        62.4863,
        58.4466,
        60.3061,
        59.5910,
        59.1642,
        60.4348,
        58.3239,
        61.1506,
        57.6125,
        61.8002,
        56.9682,
        62.4112,
        56.3634,
        62.9980,
        55.7838,
        63.5685,
        55.2214,
        64.1278,
        54.6712,
        64.6789,
        54.1300,
        65.2241,
        53.5957,
        65.7648,
        53.0669,
        66.3021,
        52.5424,
        66.8368,
        52.0214,
        67.3696,
        51.5034,
        67.9009,
        368.4984,
        424.7632,
        487.2494,
        715.3931,
        773.8397,
        834.1458,
    ]

    S300 = [
        0.2936e-14,
        0.8079e-15,
        0.2480e-14,
        0.2228e-14,
        0.3351e-14,
        0.3292e-14,
        0.3721e-14,
        0.3891e-14,
        0.3640e-14,
        0.4005e-14,
        0.3227e-14,
        0.3715e-14,
        0.2627e-14,
        0.3156e-14,
        0.1982e-14,
        0.2477e-14,
        0.1391e-14,
        0.1808e-14,
        0.9124e-15,
        0.1230e-14,
        0.5603e-15,
        0.7842e-15,
        0.3228e-15,
        0.4689e-15,
        0.1748e-15,
        0.2632e-15,
        0.8898e-16,
        0.1389e-15,
        0.4264e-16,
        0.6899e-16,
        0.1924e-16,
        0.3229e-16,
        0.8191e-17,
        0.1423e-16,
        0.6494e-15,
        0.7083e-14,
        0.3025e-14,
        0.1835e-14,
        0.1158e-13,
        0.3993e-14,
    ]

    BE = [
        0.009,
        0.015,
        0.083,
        0.084,
        0.212,
        0.212,
        0.391,
        0.391,
        0.626,
        0.626,
        0.915,
        0.915,
        1.260,
        1.260,
        1.660,
        1.665,
        2.119,
        2.115,
        2.624,
        2.625,
        3.194,
        3.194,
        3.814,
        3.814,
        4.484,
        4.484,
        5.224,
        5.224,
        6.004,
        6.004,
        6.844,
        6.844,
        7.744,
        7.744,
        0.048,
        0.044,
        0.049,
        0.145,
        0.141,
        0.145,
    ]
    # WIDTHS IN MHZ/MB

    WB300 = 0.56
    X = 0.8
    W300 = [
        1.63,
        1.646,
        1.468,
        1.449,
        1.382,
        1.360,
        1.319,
        1.297,
        1.266,
        1.248,
        1.221,
        1.207,
        1.181,
        1.171,
        1.144,
        1.139,
        1.110,
        1.108,
        1.079,
        1.078,
        1.05,
        1.05,
        1.02,
        1.02,
        1.00,
        1.00,
        0.97,
        0.97,
        0.94,
        0.94,
        0.92,
        0.92,
        0.89,
        0.89,
        1.92,
        1.92,
        1.92,
        1.81,
        1.81,
        1.81,
    ]

    Y300 = [
        -0.0233,
        0.2408,
        -0.3486,
        0.5227,
        -0.5430,
        0.5877,
        -0.3970,
        0.3237,
        -0.1348,
        0.0311,
        0.0725,
        -0.1663,
        0.2832,
        -0.3629,
        0.3970,
        -0.4599,
        0.4695,
        -0.5199,
        0.5187,
        -0.5597,
        0.5903,
        -0.6246,
        0.6656,
        -0.6942,
        0.7086,
        -0.7325,
        0.7348,
        -0.7546,
        0.7702,
        -0.7864,
        0.8083,
        -0.8210,
        0.8439,
        -0.8529,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    V = [
        0.0079,
        -0.0978,
        0.0844,
        -0.1273,
        0.0699,
        -0.0776,
        0.2309,
        -0.2825,
        0.0436,
        -0.0584,
        0.6056,
        -0.6619,
        0.6451,
        -0.6759,
        0.6547,
        -0.6675,
        0.6135,
        -0.6139,
        0.2952,
        -0.2895,
        0.2654,
        -0.2590,
        0.3750,
        -0.3680,
        0.5085,
        -0.5002,
        0.6206,
        -0.6091,
        0.6526,
        -0.6393,
        0.6640,
        -0.6475,
        0.6729,
        -0.6545,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    TH = 300.0 / TEMP
    TH1 = TH - 1.0
    B = TH**X
    PRESWV = VAPDEN * TEMP / 217.0
    PRESDA = PRES - PRESWV
    DEN = 0.001 * (PRESDA * B + 1.1 * PRESWV * TH)
    DENS = 0.001 * (PRESDA + 1.1 * PRESWV) * TH
    DFNR = WB300 * DEN
    SUM = 1.6e-17 * FREQ * FREQ * DFNR / (TH * (FREQ * FREQ + DFNR * DFNR))

    for K in range(40):
        if K == 0:
            DF = W300[0] * DENS
        else:
            DF = W300[K] * DEN
        Y = 0.001 * PRES * B * (Y300[K] + V[K] * TH1)
        STR = S300[K] * np.exp(-BE[K] * TH1)
        SF1 = (DF + (FREQ - F[K]) * Y) / ((FREQ - F[K]) ** 2 + DF * DF)
        SF2 = (DF - (FREQ + F[K]) * Y) / ((FREQ + F[K]) ** 2 + DF * DF)
        SUM = SUM + STR * (SF1 + SF2) * (FREQ / F[K]) ** 2

    O2ABS = 0.5034e12 * SUM * PRESDA * TH**3 / 3.14159

    return O2ABS


@numba.jit(cache=True, nopython=True)
def ABSN2_IM10(T, P, F):
    """
    ****ABSN2 = ABSORPTION COEFFICIENT DUE TO NITROGEN IN AIR (NEPER/KM)
    T = TEMPERATURE (K)
    P = PRESSURE (MB)
    F = FREQUENCY (GHZ)
    """
    TH = 300.0 / T
    ALPHA = 6.4e-14 * P * P * F * F * TH**3.55

    return ALPHA


@numba.jit(cache=True, nopython=True)
# https://github.com/numba/numba/issues/2518
def TB_CALC_PL_IM10(T, tau, mu_s, freq):
    """
    calculate brightness temperatures without scattering
    according to Simmer (94) pp. 87 - 91 (alpha = 1, no scattering)
    Planck/thermodynamic conform (28.05.03) # UL
    """
    h = 6.6262e-34  # Planck constant
    kB = 1.3806e-23  # Boltzmann constant
    c_li = 2.997925 * 1e8  # spped of light

    kmax = len(T)
    n_f = len(freq)

    mu = np.zeros(n_f) + mu_s
    freq_si = freq * 1e9
    lamda_si = c_li / freq_si

    IN = np.zeros(n_f, dtype=np.float64) + 2.73
    IN = (
        (2.0 * h * freq_si / (lamda_si**2.0))
        * 1.0
        / (np.exp(h * freq_si / (kB * IN)) - 1.0)
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
                (2.0 * h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(h * freq_si / (kB * T[kmax - 2 - i])) - 1)
            )
            T_pl1 = (
                (2.0 * h * freq_si / (lamda_si**2.0))
                * 1.0
                / (np.exp(h * freq_si / (kB * T[kmax - 1 - i])) - 1)
            )
            diff = (T_pl2 - T_pl1) / delta_tau
            IN = IN * np.exp(-1 * delta_tau / mu) + T_pl1 * A + diff * B

    TB = (
        (h * freq_si / kB)
        * 1.0
        / np.log((2 * h * freq_si / (IN * lamda_si**2.0)) + 1.0)
    )

    return TB
