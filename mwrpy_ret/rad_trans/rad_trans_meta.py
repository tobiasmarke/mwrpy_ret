"""Module for Radiative Transfer Metadata"""
from collections.abc import Callable
from typing import TypeAlias

from mwrpy_ret.utils import MetaData


def get_data_attributes(ret_variables: dict, source: str) -> dict:
    """Adds Metadata for Ret MWR variables for NetCDF file writing.
    Args:
        ret_variables: RetArray instances.
        source: Data type of the netCDF file.

    Returns:
        Dictionary

    Raises:
        RuntimeError: Specified data type is not supported.

    Example:
        from rad_trans.rad_trans_meta import get_data_attributes
        att = get_data_attributes('data','data_type')
    """

    if source not in (
        "radiosonde",
        "model",
    ):
        raise RuntimeError([source + " not supported for file writing."])

    read_att = att_reader[source]
    attributes = dict(ATTRIBUTES_COM, **read_att)

    for key in list(ret_variables):
        if key in attributes:
            ret_variables[key].set_attributes(attributes[key])
        else:
            del ret_variables[key]

    index_map = {v: i for i, v in enumerate(attributes)}
    ret_variables = dict(
        sorted(ret_variables.items(), key=lambda pair: index_map[pair[0]])
    )

    return ret_variables


ATTRIBUTES_COM = {
    "time": MetaData(
        long_name="Time (UTC) of the measurement",
        units="seconds since 1970-01-01 00:00:00.000",
    ),
    "height": MetaData(
        long_name="Height above mean sea level",
        standard_name="height_above_mean_sea_level",
        units="m",
    ),
    "frequency": MetaData(
        long_name="Nominal centre frequency of microwave channels",
        standard_name="radiation_frequency",
        units="GHz",
    ),
    "elevation_angle": MetaData(
        long_name="Sensor elevation angle",
        units="degree",
        comment="0=horizon, 90=zenith",
    ),
}


ATTRIBUTES_RS = {
    "tb": MetaData(
        long_name="Microwave brightness temperature simulated from radiosonde",
        standard_name="brightness_temperature",
        units="K",
    ),
    "air_temperature": MetaData(
        long_name="Temperature profile from radiosonde",
        standard_name="air_temperature",
        units="K",
    ),
    "air_pressure": MetaData(
        long_name="Pressure profile from radiosonde",
        standard_name="air_pressure",
        units="Pa",
    ),
    "absolute_humidity": MetaData(
        long_name="Absolute humidity profile from radiosonde",
        units="kg m-3",
    ),
    "lwp": MetaData(
        long_name="Column-integrated liquid water path derived from radiosonde",
        standard_name="atmosphere_cloud_liquid_water_content",
        units="kg m-2",
    ),
    "iwv": MetaData(
        long_name="Column-integrated water vapour derived from radiosonde",
        standard_name="atmosphere_mass_content_of_water_vapor",
        units="kg m-2",
    ),
}


ATTRIBUTES_MOD = {
    "tb": MetaData(
        long_name="Microwave brightness temperature simulated from model",
        standard_name="brightness_temperature",
        units="K",
    ),
    "air_temperature": MetaData(
        long_name="Temperature profile from model",
        standard_name="air_temperature",
        units="K",
    ),
    "air_pressure": MetaData(
        long_name="Pressure profile from model",
        standard_name="air_pressure",
        units="Pa",
    ),
    "absolute_humidity": MetaData(
        long_name="Absolute humidity profile from model",
        units="kg m-3",
    ),
    "lwp": MetaData(
        long_name="Column-integrated liquid water path derived from model",
        standard_name="atmosphere_cloud_liquid_water_content",
        units="kg m-2",
    ),
    "iwv": MetaData(
        long_name="Column-integrated water vapour derived from model",
        standard_name="atmosphere_mass_content_of_water_vapor",
        units="kg m-2",
    ),
}


FuncType: TypeAlias = Callable[[str], dict]
att_reader: dict[str, dict] = {
    "radiosonde": ATTRIBUTES_RS,
    "model": ATTRIBUTES_MOD,
}
