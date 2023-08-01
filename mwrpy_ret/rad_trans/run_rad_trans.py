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

freq = np.array(
    [
        22.240,
        23.040,
        23.840,
        25.440,
        26.240,
        27.840,
        31.400,
        51.260,
        52.280,
        53.860,
        54.940,
        56.660,
        57.300,
        58.00,
    ]
)
height_int = np.array(
    [
        0.0000000,
        50.000000,
        100.00000,
        150.00000,
        200.00000,
        250.00000,
        325.00000,
        400.00000,
        475.00000,
        550.00000,
        625.00000,
        700.00000,
        800.00000,
        900.00000,
        1000.0000,
        1150.0000,
        1300.0000,
        1450.0000,
        1600.0000,
        1800.0000,
        2000.0000,
        2250.0000,
        2500.0000,
        2750.0000,
        3000.0000,
        3250.0000,
        3500.0000,
        3750.0000,
        4000.0000,
        4250.0000,
        4500.0000,
        4750.0000,
        5000.0000,
        5500.0000,
        6000.0000,
        6500.0000,
        7000.0000,
        7500.0000,
        8000.0000,
        8500.0000,
        9000.0000,
        9500.0000,
        10000.000,
        15000.000,
        20000.000,
        25000.000,
        30000.000,
    ]
)


def rad_trans_rs(file_name: str) -> tuple[np.ndarray, np.ndarray]:
    with nc.Dataset(file_name) as rs_data:
        geopot = units.Quantity(
            rs_data.variables["geopotential_height"][:] * con.g0, "m^2/s^2"
        )
        height = metpy.calc.geopotential_to_height(geopot).magnitude
        height = height - height[0]

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

            iwv = rh_to_iwv(
                rs_data.variables["relative_humidity"][:] / 100.0,
                rs_data.variables["air_temperature"][:] + con.T0,
                rs_data.variables["air_pressure"][:],
                height,
            )
            print(iwv)
            top, base, _ = detect_liq_cloud(
                height,
                rs_data.variables["air_temperature"][:] + con.T0,
                rs_data.variables["relative_humidity"][:] / 100.0,
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
                print(lwp)

            tb, _, _, _ = STP_IM10(
                height_int,
                temperature_int,
                pressure_int,
                abshum_int,
                lwc,
                0.0,
                freq,
            )
        return tb, temperature_int
