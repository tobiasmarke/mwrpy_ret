import datetime
import glob
import json
import logging
import os
import warnings
from typing import Any, Iterator, NamedTuple

import numpy as np
import yaml
from numpy import ma
from yaml.loader import SafeLoader


class MetaData(NamedTuple):
    long_name: str
    units: str
    standard_name: str | None = None
    definition: str | None = None
    comment: str | None = None


def append_data(data_in: dict, key: str, array: np.ndarray) -> dict:
    """Appends data to a dictionary field (creates the field if not yet present).

    Args:
        data_in: Dictionary where data will be appended.
        key: Key of the field.
        array: Numpy array to be appended to data_in[key].

    """
    data = data_in.copy()
    if key not in data:
        data[key] = array
    else:
        data[key] = ma.concatenate((data[key], array))
    return data


def get_file_list(path_to_files: str):
    """Returns file list for specified path."""
    f_list = sorted(glob.glob(path_to_files + "/*.nc"))
    if len(f_list) == 0:
        logging.warning("Error: no files found in directory %s", path_to_files)
    return f_list


def get_processing_dates(args) -> tuple[str, str]:
    """Returns processing dates."""
    if args.date is not None:
        start_date = args.date
        stop_date = get_date_from_past(-1, start_date)
    else:
        start_date = args.start
        stop_date = args.stop
    start_date = str(date_string_to_date(start_date))
    stop_date = str(date_string_to_date(stop_date))
    return start_date, stop_date


def _get_filename(
    source: str, start: datetime.date, stop: datetime.date, site: str
) -> str:
    _, params = read_yaml_config(site)
    filename = (
        f"{site}_{source}_{start.strftime('%Y%m%d')}_{stop.strftime('%Y%m%d')}.nc"
    )

    return os.path.join(params["data_out"], filename)


def isodate2date(date_str: str) -> datetime.date:
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()


def date_range(
    start_date: datetime.date, end_date: datetime.date
) -> Iterator[datetime.date]:
    """Returns range between two dates (datetimes)."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def str_to_numeric(value: str) -> int | float:
    """Converts string to number (int or float)."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def isscalar(array: Any) -> bool:
    """Tests if input is scalar.
    By "scalar" we mean that array has a single value.
    Examples:
        >>> isscalar(1)
            True
        >>> isscalar([1])
            True
        >>> isscalar(np.array(1))
            True
        >>> isscalar(np.array([1]))
            True
    """

    arr = ma.array(array)
    if not hasattr(arr, "__len__") or arr.shape == () or len(arr) == 1:
        return True
    return False


def read_yaml_config(site: str) -> tuple[dict, dict]:
    """Reads config yaml files."""
    dir_name = os.path.dirname(os.path.realpath(__file__))

    site_file = os.path.join(dir_name, "site_config", f"{site}.yaml")
    if not os.path.isfile(site_file):
        raise NotImplementedError(f"Error: site config file {site_file} not found")
    with open(site_file, "r", encoding="utf8") as f:
        site_config = yaml.load(f, Loader=SafeLoader)

    conf_file = os.path.join(dir_name, "site_config", "config.yaml")
    if not os.path.isfile(conf_file):
        raise NotImplementedError(
            f"Error: instrument config file {conf_file} not found"
        )
    with open(conf_file, "r", encoding="utf8") as f:
        inst_config = yaml.load(f, Loader=SafeLoader)

    for name in inst_config["params"].keys():
        site_config["params"][name] = inst_config["params"][name]

    return site_config["global_specs"], site_config["params"]


def date_string_to_date(date_string: str) -> datetime.date:
    """Convert YYYY-MM-DD to Python date."""
    date_arr = [int(x) for x in date_string.split("-")]
    return datetime.date(*date_arr)


def get_time() -> str:
    """Returns current UTC-time."""
    return f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} +00:00"


def get_date_from_past(n: int, reference_date: str | None = None) -> str:
    """Return date N-days ago.
    Args:
        n: Number of days to skip (can be negative, when it means the future).
        reference_date: Date as "YYYY-MM-DD". Default is the current date.
    Returns:
        str: Date as "YYYY-MM-DD".
    """
    reference = reference_date or get_time().split()[0]
    the_date = date_string_to_date(reference) - datetime.timedelta(n)
    return str(the_date)


def loadCoeffsJSON(path) -> dict:
    """Load coefficients required for O2 absorption."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            try:
                var_all = dict(**json.load(f))
                for key in var_all.keys():
                    var_all[key] = np.asarray(var_all[key])
            except json.decoder.JSONDecodeError:
                print(path)
                raise
    return dict(**var_all)


def dcerror(x, y):
    """SIXTH-ORDER APPROX TO THE COMPLEX ERROR FUNCTION OF z=X+iY."""
    a = [
        122.607931777104326,
        214.382388694706425,
        181.928533092181549,
        93.155580458138441,
        30.180142196210589,
        5.912626209773153,
        0.564189583562615,
    ]
    b = [
        122.607931773875350,
        352.730625110963558,
        457.334478783897737,
        348.703917719495792,
        170.354001821091472,
        53.992906912940207,
        10.479857114260399,
    ]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        ZH = np.abs(y) - x * 1j
        ASUM = (
            ((((a[6] * ZH + a[5]) * ZH + a[4]) * ZH + a[3]) * ZH + a[2]) * ZH + a[1]
        ) * ZH + a[0]
        BSUM = (
            (((((ZH + b[6]) * ZH + b[5]) * ZH + b[4]) * ZH + b[3]) * ZH + b[2]) * ZH
            + b[1]
        ) * ZH + b[0]
        w = ASUM / BSUM
        w2 = 2.0 * np.exp(-((x + y * 1j) ** 2)) - np.conj(w)
        DCERROR = w
        DCERROR[y < 0] = w2[y < 0]
        return DCERROR