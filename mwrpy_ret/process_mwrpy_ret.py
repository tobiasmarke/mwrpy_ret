import datetime
import logging
import os

import netCDF4 as nc
import numpy as np

from mwrpy_ret import ret_mwr
from mwrpy_ret.era5_download.get_era5 import era5_request
from mwrpy_ret.prepare_input import (
    prepare_era5,
    prepare_ifs,
    prepare_radiosonde,
    prepare_standard_atmosphere,
)
from mwrpy_ret.rad_trans.rad_trans_meta import get_data_attributes
from mwrpy_ret.rad_trans.run_rad_trans import rad_trans
from mwrpy_ret.utils import (
    _get_filename,
    append_data,
    date_range,
    get_file_list,
    get_processing_dates,
    isodate2date,
    read_bandwidth_coefficients,
    read_beamwidth_coefficients,
    read_config,
    seconds2date,
)


def main(args):
    logging.basicConfig(level="INFO")
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    global_specs = read_config(args.site, "global_specs")
    params = read_config(args.site, "params")
    if args.command == "get_era5":
        era5_request(args.site, params, start_date, stop_date)
    else:
        data_nc = process_input(args.command, args.site, start_date, stop_date, params)
        output_file = _get_filename(args.command, start_date, stop_date, args.site)
        output_dir = os.path.dirname(output_file)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if ("time" in data_nc) & (output_file is not None):
            ret_in = ret_mwr.Ret(data_nc)
            ret_in.data = get_data_attributes(ret_in.data, args.command)
            logging.info(f"Saving output file {output_file}")
            ret_mwr.save_rpg(ret_in, output_file, global_specs, args.command)


def process_input(
    source: str,
    site: str,
    start_date: datetime.date,
    stop_date: datetime.date,
    params: dict,
) -> dict:
    data_nc: dict = {}
    if site == "standard_atmosphere":
        source = "standard_atmosphere"
    if source == "ifs":
        for date in date_range(start_date, stop_date):
            data_in = str(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__))
                    + params["data_ifs"]
                    + date.strftime("%Y/")
                    + date.strftime("%Y%m%d")
                )
            )
            file_name = get_file_list(data_in, "ecmwf")
            with nc.Dataset(file_name[0]) as ifs_data:
                for index, hour in enumerate(ifs_data["time"][:-1]):
                    output_hour = None
                    date_i = datetime.datetime.combine(
                        date, datetime.time(int(hour))
                    ).strftime("%Y%m%d%H")
                    input_ifs = prepare_ifs(ifs_data, index, date_i)
                    try:
                        output_hour = call_rad_trans(input_ifs, params)
                    except ValueError:
                        logging.info(f"Skipping time {date_i}")
                    if output_hour is not None:
                        if date_i[-2:] == "00":
                            logging.info(
                                f"Radiative transfer using {source} data "
                                f"for {site}, {date_i[:-2]}"
                            )
                        for key, array in output_hour.items():
                            data_nc = append_data(data_nc, key, array)

    elif source == "radiosonde":
        for date in date_range(start_date, stop_date):
            output_day: dict = {}
            data_in = str(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__))
                    + params["data_rs"]
                    + date.strftime("%Y/%m/%d/")
                )
            )
            file_names = get_file_list(data_in, "radiosonde")
            for file in file_names:
                output_hour = None
                with nc.Dataset(file) as rs_data:
                    input_rs = prepare_radiosonde(rs_data)
                try:
                    output_hour = call_rad_trans(input_rs, params)
                except ValueError:
                    logging.info(f"Skipping file {file}")
                if output_hour is not None:
                    for key, array in output_hour.items():
                        output_day = append_data(output_day, key, array)
            if len(output_day) > 0:
                logging.info(
                    f"Radiative transfer using {source} data for {site}, {date}"
                )
                for key, array in output_day:
                    data_nc = append_data(data_nc, key, array)

    elif source == "era5":
        file_name = str(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__))
                + params["data_era5"]
                + site
                + "_era5_input_"
                + start_date.strftime("%Y%m%d")
                + "_"
                + stop_date.strftime("%Y%m%d")
                + ".nc"
            )
        )
        with nc.Dataset(file_name) as era5_data:
            for index, hour in enumerate(era5_data["time"]):
                date_i = seconds2date(hour * 3600.0, (1900, 1, 1))
                output_hour = None
                input_era5 = prepare_era5(era5_data, index, date_i)
                try:
                    output_hour = call_rad_trans(input_era5, params)
                except ValueError:
                    logging.info(f"Skipping time {date_i}")
                if output_hour is not None:
                    if date_i[-2:] == "00":
                        logging.info(
                            f"Radiative transfer using {source} data "
                            f"for {site}, {date_i[:-2]}"
                        )
                    for key, array in output_hour.items():
                        data_nc = append_data(data_nc, key, array)

    elif source == "standard_atmosphere":
        data_in = str(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__))
                + params["data_std"]
                + "standard_atmospheres.nc"
            )
        )
        with nc.Dataset(data_in) as sa_data:
            input_sa = prepare_standard_atmosphere(sa_data)
            data_nc = call_rad_trans(input_sa, params)

    data_nc["height"] = np.array(params["height"]) + params["altitude"]
    data_nc["height_in"] = data_nc["height_in"][0, :]
    data_nc["frequency"] = np.array(params["frequency"])
    data_nc["elevation_angle"] = np.array(params["elevation_angle"])
    return data_nc


def call_rad_trans(data_in: dict, params: dict) -> dict:
    # Channel bandwidth
    coeff_bdw = read_bandwidth_coefficients()
    # Antenna beamwidth
    ape_ang = read_beamwidth_coefficients()

    data_nc = rad_trans(
        data_in,
        np.array(params["height"]) + params["altitude"],
        np.array(params["frequency"]),
        90.0 - np.array(params["elevation_angle"]),
        coeff_bdw,
        ape_ang,
    )
    return data_nc


class RetIn:
    """Class for retrieval input files"""

    def __init__(self, data_in: dict):
        self.data: dict = {}
        self._init_data(data_in)

    def _init_data(self, data_in: dict):
        for key, data in data_in.items():
            self.data[key] = data
