import numpy as np

from mwrpy_ret.atmos import (
    abs_hum,
    detect_cloud_mod,
    detect_liq_cloud,
    get_cloud_prop,
    hum_to_iwv,
    interp_log_p,
)
from mwrpy_ret.rad_trans import calc_mw_rt


def rad_trans(
    input_dat: dict,
    params: dict,
    coeff_bdw: dict,
    ape_ang: np.ndarray,
) -> dict:
    theta = 90.0 - np.array(params["elevation_angle"])
    tb = np.ones((1, len(params["frequency"]), len(theta)), np.float32) * -999.0
    tb_pro = np.ones((1, len(params["frequency"]), len(theta)), np.float32) * -999.0
    tb_tmp = np.ones((1, len(params["frequency"]), len(theta)), np.float32) * -999.0
    lwp, lwp_pro = -999.0, -999.0

    # Integrated water vapor [kg/m²]
    if "absolute_humidity" not in input_dat:
        input_dat["absolute_humidity"] = abs_hum(
            input_dat["air_temperature"][:], input_dat["relative_humidity"][:]
        )
    if np.any(input_dat["absolute_humidity"].mask):
        iwv = -999.0
    else:
        iwv = hum_to_iwv(
            input_dat["absolute_humidity"][:],
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
            height_new, lwc_new, lwp_tmp = get_cloud_prop(
                base, top, input_dat["height"][:], input_dat, method
            )
            # Interpolate to new grid
            pressure_new = interp_log_p(
                input_dat["air_pressure"][:], input_dat["height"][:], height_new
            )
            temperature_new = np.interp(
                height_new, input_dat["height"][:], input_dat["air_temperature"][:]
            )
            abshum_new = np.interp(
                height_new, input_dat["height"][:], input_dat["absolute_humidity"][:]
            )
        else:
            height_new = input_dat["height"][:]
            lwc_new = np.zeros(len(height_new), np.float32)
            lwp_tmp = 0.0
            temperature_new = input_dat["air_temperature"][:]
            pressure_new = input_dat["air_pressure"][:]
            abshum_new = input_dat["absolute_humidity"][:]

        # Radiative transport
        tb_tmp[0, :, 0], tau_k, tau_v = calc_mw_rt(
            height_new,
            temperature_new,
            pressure_new,
            abshum_new,
            lwc_new,
            theta[0],
            np.array(params["frequency"]),
            coeff_bdw,
            ape_ang,
        )
        if len(theta) > 1:
            for i_ang in range(len(theta) - 1):
                tb_tmp[0, :, i_ang + 1], _, _ = calc_mw_rt(
                    height_new,
                    temperature_new,
                    pressure_new,
                    abshum_new,
                    lwc_new,
                    theta[i_ang + 1],
                    np.array(params["frequency"]),
                    coeff_bdw,
                    ape_ang,
                    tau_k,
                    tau_v,
                )
        if method == "prognostic":
            lwp_pro, tb_pro = np.copy(lwp_tmp), np.copy(tb_tmp)
        else:
            lwp, tb = np.copy(lwp_tmp), np.copy(tb_tmp)

    # Interpolate to final grid
    pressure_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["air_pressure"][:],
    )
    temperature_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["air_temperature"][:],
    )
    abshum_int = np.interp(
        params["height"],
        input_dat["height"][:] - input_dat["height"][0],
        input_dat["absolute_humidity"][:],
    )

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
        "height_in": np.expand_dims(input_dat["height"][:], 0),
        "air_temperature_in": np.expand_dims(input_dat["air_temperature"][:], 0),
        "air_pressure_in": np.expand_dims(input_dat["air_pressure"][:], 0),
        "absolute_humidity_in": np.expand_dims(input_dat["absolute_humidity"][:], 0),
    }

    return output
