import metpy.calc
import netCDF4 as nc
import numpy as np
from metpy.units import units

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import abs_hum, detect_liq_cloud, interp_log_p, mod_ad, rh_to_iwv
from mwrpy_ret.rad_trans import STP_IM10


def rad_trans_rs(
    file_name: str,
    height_int: np.ndarray,
    freq: np.ndarray,
    theta: np.ndarray,
) -> dict:
    output: dict = {}
    with nc.Dataset(file_name) as rs_data:
        # GPM to m
        geopot = units.Quantity(
            rs_data.variables["geopotential_height"][:] * con.g0, "m^2/s^2"
        )
        height = metpy.calc.geopotential_to_height(geopot).magnitude
        height = height - height[0] + height_int[0]

        # Integrated water vapor
        iwv = rh_to_iwv(
            rs_data.variables["air_temperature"][:] + con.T0,
            rs_data.variables["relative_humidity"][:] / 100.0,
            height,
        )

        # Cloud model / water
        top, base = detect_liq_cloud(
            height,
            rs_data.variables["air_temperature"][:] + con.T0,
            rs_data.variables["relative_humidity"][:] / 100.0,
            rs_data.variables["air_pressure"][:] * 100.0,
        )

        lwc, lwp, height_new, cloud_new = (
            np.empty(0, np.float64),
            0.0,
            np.empty(0, np.float64),
            np.empty(0, np.float64),
        )
        if len(top) > 0:
            for icl, _ in enumerate(top):
                xcl = np.where((height >= base[icl]) & (height <= top[icl]))[0]
                if len(xcl) > 1:
                    lwcx, cloudx = mod_ad(
                        rs_data.variables["air_temperature"][xcl] + con.T0,
                        rs_data.variables["air_pressure"][xcl] * 100.0,
                        height[xcl],
                    )
                    cloud_new = np.hstack((cloud_new, cloudx))
                    lwc = np.hstack((lwc, lwcx))
                    lwp = lwp + np.sum(lwcx * np.diff(height[xcl]))

                    if len(height_new) == 0:
                        height_new = height_int[height_int < base[0]]
                    else:
                        height_new = np.hstack(
                            (
                                height_new,
                                height_int[
                                    (height_int > top[icl - 1])
                                    & (height_int < base[icl])
                                ],
                            )
                        )

            # New vertical grid
            height_new = np.hstack((height_new, height_int[height_int > top[-1]]))
            height_new = np.sort(np.hstack((height_new, cloud_new)))
        else:
            height_new = height_int

        # Interpolate to new grid
        pressure_new = interp_log_p(
            rs_data.variables["air_pressure"][:], height, height_new
        )
        if np.min(pressure_new) >= 0.0:
            temperature_new = np.interp(
                height_new, height, rs_data.variables["air_temperature"][:] + con.T0
            )

            relhum_new = np.interp(
                height_new,
                height,
                rs_data.variables["relative_humidity"][:] / 100.0,
            )

            abshum_new = abs_hum(temperature_new, relhum_new)

            _, xx, _ = np.intersect1d(
                height_new, cloud_new, assume_unique=True, return_indices=True
            )
            lwc_new = np.zeros(len(height_new) - 1, np.float32)
            lwc_new[xx] = lwc

            # Radiative transport
            tb = np.empty((1, len(freq), len(theta)), np.float32)
            tb[0, :, 0], tau = STP_IM10(
                height_new,
                temperature_new,
                pressure_new,
                abshum_new,
                lwc_new,
                theta[0],
                freq,
            )
            if len(theta) > 1:
                for i_ang in range(len(theta) - 1):
                    tb[0, :, i_ang + 1], _ = STP_IM10(
                        height_new,
                        temperature_new,
                        pressure_new,
                        abshum_new,
                        lwc_new,
                        theta[i_ang + 1],
                        freq,
                        tau,
                    )

        # Interpolate to final grid
        pressure_int = interp_log_p(
            rs_data.variables["air_pressure"][:], height, height_int
        )
        if np.min(pressure_int) >= 0.0:
            temperature_int = np.interp(
                height_int, height, rs_data.variables["air_temperature"][:] + con.T0
            )

            relhum_int = np.interp(
                height_int, height, rs_data.variables["relative_humidity"][:] / 100.0
            )

            abshum_int = abs_hum(temperature_int, relhum_int)

            output = {
                "time": np.asarray([rs_data.variables["BEZUGSDATUM_SYNOP"][-1].data]),
                "tb": tb,
                "temperature": np.expand_dims(temperature_int, 0),
                "pressure": np.expand_dims(pressure_int, 0),
                "absolute_humidity": np.expand_dims(abshum_int, 0),
                "lwp": np.asarray([lwp]),
                "iwv": np.asarray([iwv]),
            }
    return output
