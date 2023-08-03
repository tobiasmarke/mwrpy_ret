import datetime
import logging
import os

import numpy as np

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
    output_all: dict = {}
    for date in date_range(start_date, stop_date):
        output_day = process_input(args.command, date, args.site)
        logging.info(f"Radiative transfer using {args.command} for {args.site}, {date}")
        for key, array in output_day.items():
            output_all = append_data(output_all, key, array)

    output_file = _get_filename(args.command, start_date, stop_date, args.site)
    output_dir = os.path.dirname(output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def process_input(source: str, date: datetime.date, site: str) -> dict:
    _, params = read_yaml_config(site)
    output_day: dict = {}
    if source == "radiosonde":
        data_in = os.path.join(params["data_rs"], date.strftime("%Y/%m/%d"))
        file_names = get_file_list(data_in)
        for file in file_names:
            output_hour = rad_trans_rs(
                file,
                np.array(params["height"]),
                np.array(params["frequency"]),
                np.array(params["elevation"]) - 90.0,
            )
            for key, array in output_hour.items():
                output_day = append_data(output_day, key, array)
    return output_day
