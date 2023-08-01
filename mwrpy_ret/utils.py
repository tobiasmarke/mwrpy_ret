import glob
import logging


def get_file_list(path_to_files: str):
    """Returns file list for specified path."""
    f_list = sorted(glob.glob(path_to_files + "/*.nc"))
    if len(f_list) == 0:
        logging.warning("Error: no files found in directory %s", path_to_files)
    return f_list
