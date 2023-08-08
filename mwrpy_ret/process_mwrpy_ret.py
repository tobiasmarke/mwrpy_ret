import datetime
import logging
import os

import numpy as np

from mwrpy_ret import ret_mwr
from mwrpy_ret.rad_trans.rad_trans_meta import get_data_attributes
from mwrpy_ret.rad_trans.run_rad_trans import rad_trans_rs
from mwrpy_ret.utils import (
    _get_filename,
    append_data,
    date_range,
    get_file_list,
    get_processing_dates,
    isodate2date,
    read_yaml_config,
)


def main(args):
    logging.basicConfig(level="INFO")
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    global_attributes, params = read_yaml_config(args.site)
    data_nc: dict = {}
    for date in date_range(start_date, stop_date):
        output_day = process_input(args.command, date, params)
        logging.info(f"Radiative transfer using {args.command} for {args.site}, {date}")
        if output_day is not None:
            for key in output_day:
                data_nc = append_data(data_nc, key, output_day[key])

    data_nc["height"] = np.array(params["height"]) + params["altitude"]
    data_nc["frequency"] = np.array(params["frequency"])
    data_nc["elevation_angle"] = np.array(params["elevation_angle"])

    output_file = _get_filename(args.command, start_date, stop_date, args.site)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if ("time" in data_nc) & (output_file is not None):
        ret_in = ret_mwr.Ret(data_nc)
        ret_in.data = get_data_attributes(ret_in.data, args.command)
        logging.info(f"Saving output file {output_file}")
        ret_mwr.save_rpg(ret_in, output_file, global_attributes, args.command)


def process_input(source: str, date: datetime.date, params: dict) -> dict:
    output_day: dict = {}
    if source == "radiosonde":
        data_in = os.path.join(params["data_rs"], date.strftime("%Y/%m/%d"))
        file_names = get_file_list(data_in)
        for file in file_names:
            output_hour = rad_trans_rs(
                file,
                np.array(params["height"]) + params["altitude"],
                np.array(params["frequency"]),
                np.array(params["elevation_angle"]) - 90.0,
            )
            if output_hour is not None:
                for key, array in output_hour.items():
                    output_day = append_data(output_day, key, array)
    return output_day


class RetIn:
    """Class for retrieval input files"""

    def __init__(self, data_in: dict):
        self.data: dict = {}
        self._init_data(data_in)

    def _init_data(self, data_in: dict):
        for key, data in data_in.items():
            self.data[key] = data
