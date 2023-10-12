import metpy.calc
import netCDF4 as nc
import numpy as np
from metpy.units import units

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import era5_geopot


def prepare_rs(file: str) -> dict:
    input_rs: dict = {}
    with nc.Dataset(file) as rs_data:
        geopotential = units.Quantity(
            rs_data.variables["geopotential_height"][:] * con.g0, "m^2/s^2"
        )
        input_rs["height"] = metpy.calc.geopotential_to_height(
            geopotential[:]
        ).magnitude
        input_rs["air_temperature"] = rs_data.variables["air_temperature"][:] + con.T0
        input_rs["relative_humidity"] = (
            rs_data.variables["relative_humidity"][:] / 100.0
        )
        input_rs["air_pressure"] = rs_data.variables["air_pressure"][:] * 100.0
        input_rs["time"] = rs_data.variables["BEZUGSDATUM_SYNOP"][-1].data

    return input_rs


def prepare_mod(mod_data: dict, index: int, date_i: int) -> dict:
    input_mod: dict = {}

    geopotential, input_mod["air_pressure"] = era5_geopot(
        mod_data["level"][:],
        np.mean(np.exp(mod_data["lnsp"][index, 0, :, :]), axis=(0, 1)),
        np.mean(mod_data["z"][index, 0, :, :], axis=(0, 1)),
        np.mean(mod_data["t"][index, :, :, :], axis=(1, 2)),
        np.mean(mod_data["q"][index, :, :, :], axis=(1, 2)),
    )
    geopotential = units.Quantity(geopotential, "m^2/s^2")
    input_mod["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_mod["air_temperature"] = np.flip(
        np.mean(mod_data["t"][index, :, :, :], axis=(1, 2))
    )
    humidity = np.flip(np.mean(mod_data["q"][index, :, :, :], axis=(1, 2)))
    input_mod[
        "relative_humidity"
    ] = metpy.calc.relative_humidity_from_specific_humidity(
        input_mod["air_pressure"] * units.Pa,
        units.Quantity(input_mod["air_temperature"], "K"),
        humidity,
    ).magnitude
    input_mod["time"] = date_i

    return input_mod
