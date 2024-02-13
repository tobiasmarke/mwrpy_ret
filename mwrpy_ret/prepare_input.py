import metpy.calc
import netCDF4 as nc
import numpy as np
from metpy.units import units

import mwrpy_ret.constants as con
from mwrpy_ret.atmos import era5_geopot, moist_rho_rh, q2rh
from mwrpy_ret.utils import seconds_since_epoch


def prepare_ifs(ifs_data: nc.Dataset, index: int, date_i: str) -> dict:
    input_ifs: dict = dict(
        height=ifs_data["height"][index, :],
        air_temperature=ifs_data["temperature"][index, :],
        air_pressure=ifs_data["pressure"][index, :],
        relative_humidity=ifs_data["rh"][index, :],
    )
    input_ifs["lwc"] = ifs_data["ql"][index, :] * moist_rho_rh(
        input_ifs["air_pressure"],
        input_ifs["air_temperature"],
        input_ifs["relative_humidity"],
    )
    input_ifs["time"] = seconds_since_epoch(date_i)

    return input_ifs


def prepare_standard_atmosphere(sa_data: nc.Dataset) -> dict:
    input_sa: dict = dict(
        height=sa_data.variables["height"][:28] * 1000.0,
        air_temperature=sa_data.variables["t_atmo"][:28, 0],
        air_pressure=sa_data.variables["p_atmo"][:28, 0] * 100.0,
        absolute_humidity=sa_data.variables["a_atmo"][:28, 0],
    )
    input_sa["relative_humidity"] = q2rh(
        sa_data.variables["q_atmo"][:28, 0],
        input_sa["air_temperature"],
        input_sa["air_pressure"],
    )
    input_sa["time"] = 0

    return input_sa


def prepare_radiosonde(rs_data: nc.Dataset) -> dict:
    input_rs: dict = {}
    geopotential = units.Quantity(
        rs_data.variables["geopotential_height"][:] * con.g0, "m^2/s^2"
    )
    input_rs["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_rs["height"] = input_rs["height"][:]
    input_rs["air_temperature"] = rs_data.variables["air_temperature"][:] + con.T0
    input_rs["relative_humidity"] = rs_data.variables["relative_humidity"][:] / 100.0
    input_rs["air_pressure"] = rs_data.variables["air_pressure"][:] * 100.0
    input_rs["time"] = seconds_since_epoch(
        str(rs_data.variables["BEZUGSDATUM_SYNOP"][-1].data)
    )

    return input_rs


def prepare_era5(mod_data: dict, index: int, date_i: str) -> dict:
    input_era5: dict = {}
    geopotential, input_era5["air_pressure"] = era5_geopot(
        mod_data["level"][:],
        np.mean(np.exp(mod_data["lnsp"][index, 0, :, :]), axis=(0, 1)),
        np.mean(mod_data["z"][index, 0, :, :], axis=(0, 1)),
        np.mean(mod_data["t"][index, :, :, :], axis=(1, 2)),
        np.mean(mod_data["q"][index, :, :, :], axis=(1, 2)),
    )
    geopotential = units.Quantity(geopotential, "m^2/s^2")
    input_era5["height"] = metpy.calc.geopotential_to_height(geopotential[:]).magnitude
    input_era5["height"] = input_era5["height"][:]
    input_era5["air_pressure"] = input_era5["air_pressure"][:]
    input_era5["air_temperature"] = np.flip(
        np.mean(mod_data["t"][index, :, :, :], axis=(1, 2))
    )
    input_era5[
        "relative_humidity"
    ] = metpy.calc.relative_humidity_from_specific_humidity(
        input_era5["air_pressure"] * units.Pa,
        units.Quantity(input_era5["air_temperature"], "K"),
        np.flip(np.mean(mod_data["q"][index, :, :, :], axis=(1, 2))),
    ).magnitude
    clwc = np.flip(np.mean(mod_data["clwc"][index, :, :, :], axis=(1, 2)))
    mxr = metpy.calc.mixing_ratio_from_relative_humidity(
        units.Quantity(input_era5["air_pressure"], "Pa"),
        units.Quantity(input_era5["air_temperature"], "K"),
        units.Quantity(input_era5["relative_humidity"], "dimensionless"),
    )
    rho = metpy.calc.density(
        units.Quantity(input_era5["air_pressure"], "Pa"),
        units.Quantity(input_era5["air_temperature"], "K"),
        mxr,
    )
    input_era5["lwc"] = clwc * rho.magnitude
    input_era5["time"] = seconds_since_epoch(date_i)

    return input_era5
