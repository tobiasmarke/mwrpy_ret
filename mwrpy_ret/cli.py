#!/usr/bin/env python3
"""A wrapper script for calling data processing functions.
All modules MUST have an add_arguments function
which adds the subcommand to the subparser.
"""

import argparse
import sys

from mwrpy_ret import process_mwrpy_ret, utils


def main(args):
    args = _parse_args(args)
    process_mwrpy_ret.main(args)


def _parse_args(args):
    parser = argparse.ArgumentParser(description="MWRpy_ret processing main wrapper.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["ifs", "radiosonde", "era5", "get_era5", "standard_atmosphere"],
        default="ifs",
        help="Command to execute.",
    )
    group = parser.add_argument_group(title="General options")
    group.add_argument(
        "-s",
        "--site",
        required=True,
        help="Site to process data from, e.g. juelich",
        type=str,
    )
    group.add_argument(
        "--start",
        type=str,
        metavar="YYYY-MM-DD",
        help="Starting date. Default is current day - 2 (included).",
        default=utils.get_date_from_past(0),
    )
    group.add_argument(
        "--stop",
        type=str,
        metavar="YYYY-MM-DD",
        help="Stopping date. Default is current day - 1 (excluded).",
        default=utils.get_date_from_past(-1),
    )
    group.add_argument(
        "-d",
        "--date",
        type=str,
        metavar="YYYY-MM-DD",
        help="Single date to be processed.",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    main(sys.argv[1:])
