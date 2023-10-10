import datetime

import cdsapi
import numpy as np

from mwrpy_ret.utils import _get_filename


def era5_request(
    site: str, params: dict, start_date: datetime.date, stop_date: datetime.date
):
    """Function to download ERA5 data from CDS API for specified site and dates
    Args:
        site: Name of site
        params: config dictionary
        start_date: first day of request
        stop_date: last day of request
    """
    output_file = _get_filename("era5", start_date, stop_date, site)

    lat_box = get_corner_coord(
        params["latitude"], params["lat_offset"], params["lat_res"]
    )
    lon_box = get_corner_coord(
        params["longitude"], params["lon_offset"], params["lon_res"]
    )
    area_str = f"{lat_box[1]:.3f}/{lon_box[0]:.3f}/{lat_box[0]:.3f}/{lon_box[1]:.3f}"
    lat_res, lon_res = params["lat_res"], params["lon_res"]
    grid_str = f"{lat_res:.3f}/{lon_res:.3f}"

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-complete",
        {
            "class": "ea",
            "dataset": "era5",
            "date": str(start_date) + "/to/" + str(stop_date),
            "expver": "1",
            "levelist": "1/to/137",
            "levtype": "ml",
            "param": "129/130/133/152/246/248",
            "stream": "oper",
            "time": "00/to/23/by/1",
            "type": "an",
            "grid": grid_str,
            "area": area_str,
            "format": "netcdf",
        },
        output_file,
    )


def get_corner_coord(stn_coord, offset, resol):
    """get corners of a coordinate box around station coordinates
    which match model grid points"""
    stn_coord_rounded = (
        round(stn_coord / resol) * resol
    )  # round centre coordinate to model resolution
    return stn_coord_rounded + np.array(offset)
