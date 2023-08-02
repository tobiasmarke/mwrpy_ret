import metpy.calc
import netCDF4 as nc
import numpy as np
from metpy.units import units

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import (
    abs_hum,
    detect_liq_cloud,
    interp_log_p,
    lwp_from_lwc,
    mod_ad,
    rh_to_iwv,
)
from mwrpy_ret.rad_trans import STP_IM10


def rad_trans_rs(
    file_name: str,
    height_int: np.ndarray,
    freq: np.ndarray,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32, np.float32, np.ndarray
]:
    with nc.Dataset(file_name) as rs_data:
        # GPM to m
        geopot = units.Quantity(
            rs_data.variables["geopotential_height"][:] * con.g0, "m^2/s^2"
        )
        height = metpy.calc.geopotential_to_height(geopot).magnitude
        height = height - height[0]

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

            # Integrated water vapor
            iwv = rh_to_iwv(
                rs_data.variables["relative_humidity"][:] / 100.0,
                rs_data.variables["air_temperature"][:] + con.T0,
                rs_data.variables["air_pressure"][:],
                height,
            )
            # iwv = 0.0
            # absh = abs_hum(
            #     rs_data.variables["air_temperature"][:] + con.T0,
            #     rs_data.variables["relative_humidity"][:] / 100.0,
            # )
            # for ix in range(len(absh) - 1):
            #     iwv = iwv + ((absh[ix] + absh[ix + 1]) / 2.0) * (
            #         height[ix + 1] - height[ix]
            #     )

            # Cloud model / water
            top, base, cloud = detect_liq_cloud(
                height,
                rs_data.variables["air_temperature"][:] + con.T0,
                rs_data.variables["relative_humidity"][:] / 100.0,
                rs_data.variables["air_pressure"][:] * 100.0,
            )

            lwc, lwp = np.empty(0), 0.0
            if len(top) > 0:
                for icl, _ in enumerate(top):
                    xcl = np.where((height >= base[icl]) & (height <= top[icl]))[0]
                    lwcx = mod_ad(
                        rs_data.variables["air_temperature"][xcl] + con.T0,
                        rs_data.variables["air_pressure"][xcl] * 100.0,
                        height[xcl],
                    )
                    lwc = np.hstack((lwc, lwcx))
                    lwp = lwp + lwp_from_lwc(lwcx, height[xcl])

            _, xx, _ = np.intersect1d(height, cloud, return_indices=True)
            lwc = mod_ad(
                rs_data.variables["air_temperature"][xx] + con.T0,
                rs_data.variables["air_pressure"][xx] * 100.0,
                height[xx],
            )

            # New vertical grid
            height_new = np.sort(np.hstack((height_int, cloud)))

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

                _, xx, _ = np.intersect1d(height_new, cloud, return_indices=True)
                lwc_new = np.zeros(len(height_new) - 1, np.float32)
                lwc_new[xx[0:-1]] = lwc

            # Radiative transport
            tb, _, _, _ = STP_IM10(
                height_new,
                temperature_new,
                pressure_new,
                abshum_new,
                lwc_new,
                0.0,
                freq,
            )

        return (
            np.asarray(tb),
            np.asarray(temperature_int),
            np.asarray(pressure_int),
            abshum_int,
            np.float32(lwp),
            np.float32(iwv),
            height_int,
        )
