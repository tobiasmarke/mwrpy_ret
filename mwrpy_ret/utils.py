import glob
import json
import logging
import os
import warnings

import numpy as np


def get_file_list(path_to_files: str):
    """Returns file list for specified path."""
    f_list = sorted(glob.glob(path_to_files + "/*.nc"))
    if len(f_list) == 0:
        logging.warning("Error: no files found in directory %s", path_to_files)
    return f_list


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
