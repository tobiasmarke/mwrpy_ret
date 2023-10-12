import datetime
import logging
import os

import netCDF4 as nc
import numpy as np

from mwrpy_ret import ret_mwr
from mwrpy_ret.era5_download.get_era5 import era5_request
from mwrpy_ret.rad_trans.prepare_input import prepare_mod, prepare_rs
from mwrpy_ret.rad_trans.rad_trans_meta import get_data_attributes
from mwrpy_ret.rad_trans.run_rad_trans import rad_trans
from mwrpy_ret.utils import (
    GAUSS,
    _get_filename,
    append_data,
    date_range,
    get_file_list,
    get_processing_dates,
    isodate2date,
    loadCoeffsJSON,
    read_yaml_config,
    seconds2date,
)


def main(args):
    logging.basicConfig(level="INFO")
    _start_date, _stop_date = get_processing_dates(args)
    start_date = isodate2date(_start_date)
    stop_date = isodate2date(_stop_date)
    global_attributes, params = read_yaml_config(args.site)
    if args.command == "era5":
        era5_request(args.site, params, start_date, stop_date)
    else:
        data_nc = process_input(args.command, args.site, start_date, stop_date, params)

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


def process_input(
    source: str,
    site: str,
    start_date: datetime.date,
    stop_date: datetime.date,
    params: dict,
) -> dict:
    # Channel bandwidth
    path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/rad_trans/coeff/o2_bandpass_interp_freqs.json"
    )
    FFI = loadCoeffsJSON(path)
    bdw_fre = FFI["FFI"].T
    path = (
        os.path.dirname(os.path.realpath(__file__))
        + "/rad_trans/coeff/o2_bandpass_interp_norm_resp.json"
    )
    FRIN = loadCoeffsJSON(path)
    bdw_wgh = FRIN["FRIN"].T
    f_all, ind1 = np.empty(0, np.float32), np.zeros(1, np.int32)
    for ff in range(7):
        ifr = np.where(bdw_fre[ff, :] >= 0.0)[0]
        f_all = np.hstack((f_all, bdw_fre[ff, ifr]))
        ind1 = np.hstack((ind1, ind1[len(ind1) - 1] + len(ifr)))

    # Antenna beamwidth
    ape_ini = np.linspace(-9.9, 9.9, 199)
    ape_ang = ape_ini[GAUSS(ape_ini, 0.0) > 1e-3]
    ape_ang = ape_ang[ape_ang >= 0.0]

    data_nc: dict = {}
    if source == "radiosonde":
        for date in date_range(start_date, stop_date):
            output_day: dict = {}
            data_in = os.path.join(params["data_rs"], date.strftime("%Y/%m/%d"))
            file_names = get_file_list(data_in)
            for file in file_names:
                output_hour = None
                input_rs = prepare_rs(file)
                try:
                    output_hour = rad_trans(
                        input_rs,
                        np.array(params["height"]) + params["altitude"],
                        np.array(params["frequency"]),
                        90.0 - np.array(params["elevation_angle"]),
                        bdw_fre,
                        bdw_wgh,
                        f_all,
                        ind1,
                        ape_ang,
                    )
                except ValueError:
                    logging.info(f"Skipping file {file}")
                if output_hour is not None:
                    for key, array in output_hour.items():
                        output_day = append_data(output_day, key, array)

            if output_day is not None:
                logging.info(f"Radiative transfer using {source} for {site}, {date}")
                for key in output_day:
                    data_nc = append_data(data_nc, key, output_day[key])

    else:
        file_name = get_file_list(
            params["data_mod"]
            + site
            + "_era5_"
            + start_date.strftime("%Y%m%d")
            + "_"
            + stop_date.strftime("%Y%m%d")
        )
        if len(file_name) == 1:
            with nc.Dataset(file_name[0]) as mod_data:
                for index, hour in enumerate(mod_data["time"]):
                    date_i = seconds2date(hour * 3600.0, (1900, 1, 1))
                    output_hour = None
                    input_mod = prepare_mod(mod_data, index, int(date_i))
                    try:
                        output_hour = rad_trans(
                            input_mod,
                            np.array(params["height"]) + params["altitude"],
                            np.array(params["frequency"]),
                            90.0 - np.array(params["elevation_angle"]),
                            bdw_fre,
                            bdw_wgh,
                            f_all,
                            ind1,
                            ape_ang,
                        )
                    except ValueError:
                        logging.info(f"Skipping time {date_i}")
                    if output_hour is not None:
                        for key, array in output_hour.items():
                            output_day = append_data(output_day, key, array)

                if output_day is not None:
                    logging.info(
                        f"Radiative transfer using {source} for {site}, {date_i[:-3]}"
                    )
                    for key in output_day:
                        data_nc = append_data(data_nc, key, output_day[key])

    return data_nc


class RetIn:
    """Class for retrieval input files"""

    def __init__(self, data_in: dict):
        self.data: dict = {}
        self._init_data(data_in)

    def _init_data(self, data_in: dict):
        for key, data in data_in.items():
            self.data[key] = data
