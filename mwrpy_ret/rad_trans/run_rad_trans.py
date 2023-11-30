import numpy as np

from mwrpy_ret.atmos import (
    abs_hum,
    detect_cloud_mod,
    detect_liq_cloud,
    interp_log_p,
    mod_ad,
    rh_to_iwv,
)
from mwrpy_ret.rad_trans import STP_IM10


def rad_trans(
    input_dat: dict,
    height_int: np.ndarray,
    freq: np.ndarray,
    theta: np.ndarray,
    coeff_bdw: dict,
    ape_ang: np.ndarray,
) -> dict:
    # Integrated water vapor [kg/m²]
    iwv = rh_to_iwv(
        input_dat["air_temperature"][:],
        input_dat["relative_humidity"][:],
        input_dat["height"][:],
    )

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    if "lwc" in input_dat:
        top, base = detect_cloud_mod(input_dat["height"][:], input_dat["lwc"][:])
    else:
        top, base = detect_liq_cloud(
            input_dat["height"][:],
            input_dat["air_temperature"][:],
            input_dat["relative_humidity"][:],
            input_dat["air_pressure"][:],
        )

    lwc, lwp, height_new, cloud_new = (
        np.empty(0, np.float64),
        0.0,
        np.empty(0, np.float64),
        np.empty(0, np.float64),
    )
    if len(top) in np.linspace(1, 15, 15):
        for icl, _ in enumerate(top):
            xcl = np.where(
                (input_dat["height"][:] >= base[icl])
                & (input_dat["height"][:] <= top[icl])
            )[0]
            if len(xcl) > 1:
                if "lwc" in input_dat:
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
                cloud_new = np.hstack((cloud_new, cloudx))
                lwc = np.hstack((lwc, lwcx))
                lwp = lwp + np.sum(lwcx * np.diff(input_dat["height"][xcl]))

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

    else:
        height_new = height_int
        lwc_new = np.zeros(len(height_new) - 1, np.float32)

    # Interpolate to new grid
    pressure_new = interp_log_p(
        input_dat["air_pressure"][:], input_dat["height"][:], height_new
    )
    temperature_new = np.interp(
        height_new, input_dat["height"][:], input_dat["air_temperature"][:]
    )
    relhum_new = np.interp(
        height_new,
        input_dat["height"][:],
        input_dat["relative_humidity"][:],
    )
    abshum_new = abs_hum(temperature_new, relhum_new)

    # Radiative transport
    tb = np.empty((1, len(freq), len(theta)), np.float32)
    tb[0, :, 0], tau_k, tau_v = STP_IM10(
        height_new,
        temperature_new,
        pressure_new,
        abshum_new,
        lwc_new,
        theta[0],
        freq,
        coeff_bdw,
        ape_ang,
    )
    if len(theta) > 1:
        for i_ang in range(len(theta) - 1):
            tb[0, :, i_ang + 1], _, _ = STP_IM10(
                height_new,
                temperature_new,
                pressure_new,
                abshum_new,
                lwc_new,
                theta[i_ang + 1],
                freq,
                coeff_bdw,
                ape_ang,
                tau_k,
                tau_v,
            )

    # Interpolate to final grid
    pressure_int = interp_log_p(
        input_dat["air_pressure"][:], input_dat["height"][:], height_int
    )
    temperature_int = np.interp(
        height_int, input_dat["height"][:], input_dat["air_temperature"][:]
    )
    relhum_int = np.interp(
        height_int, input_dat["height"][:], input_dat["relative_humidity"][:]
    )
    abshum_int = abs_hum(temperature_int, relhum_int)

    output = {
        "time": np.asarray([input_dat["time"]]),
        "tb": tb,
        "air_temperature": np.expand_dims(temperature_int, 0),
        "air_pressure": np.expand_dims(pressure_int, 0),
        "absolute_humidity": np.expand_dims(abshum_int, 0),
        "lwp": np.asarray([lwp]),
        "iwv": np.asarray([iwv]),
    }

    return output
