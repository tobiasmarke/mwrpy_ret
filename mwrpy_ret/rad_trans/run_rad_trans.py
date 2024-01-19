import numpy as np

from mwrpy_ret.atmos import (
    abs_hum,
    detect_cloud_mod,
    detect_liq_cloud,
    get_cloud_prop,
    interp_log_p,
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
    tb = np.ones((1, len(freq), len(theta)), np.float32) * -999.0
    tb_pro = np.ones((1, len(freq), len(theta)), np.float32) * -999.0
    lwp, lwp_pro = -999.0, -999.0

    # Integrated water vapor [kg/m²]
    iwv = rh_to_iwv(
        input_dat["air_temperature"][:],
        input_dat["relative_humidity"][:],
        input_dat["air_pressure"][:],
        input_dat["height"][:],
    )

    # Cloud geometry [m] / cloud water content (LWC, LWP)
    cloud_methods = ("prognostic", "detected") if "lwc" in input_dat else "detected"
    for method in cloud_methods:
        if method == "prognostic":
            top, base = detect_cloud_mod(input_dat["height"][:], input_dat["lwc"][:])
        else:
            top, base = detect_liq_cloud(
                input_dat["height"][:],
                input_dat["air_temperature"][:],
                input_dat["relative_humidity"][:],
                input_dat["air_pressure"][:],
            )
        if len(top) in np.linspace(1, 15, 15):
            height_new, lwc_new, lwp = get_cloud_prop(
                base, top, height_int, input_dat, method
            )
        else:
            height_new = height_int
            lwc_new = np.zeros(len(height_new) - 1, np.float32)
            lwp = 0.0

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
        if method == "prognostic":
            lwp_pro, tb_pro = lwp, tb

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
        "tb_pro": tb_pro,
        "air_temperature": np.expand_dims(temperature_int, 0),
        "air_pressure": np.expand_dims(pressure_int, 0),
        "absolute_humidity": np.expand_dims(abshum_int, 0),
        "lwp": np.asarray([lwp]),
        "lwp_pro": np.asarray([lwp_pro]),
        "iwv": np.asarray([iwv]),
    }

    return output
