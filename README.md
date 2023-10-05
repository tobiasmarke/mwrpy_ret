# MWRpy_ret

Repository for Microwave Radiometer Retrieval Derivation

## Installation

From GitHub:

```shell
git clone https://github.com/tobiasmarke/mwrpy_ret.git
cd mwrpy_ret
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install .
```

MWRpy_ret requires Python 3.10 or newer.


## Configuration

The folder `mwrpy_ret/site_config/` contains site-specific configuration files, 
defining the input and output data paths etc., and the file `config.yaml`, which
defines the elevation angles, frequencies and height grid.

## Command line usage

MWRpy_ret can be run using the command line tool `mwrpy_ret/cli.py`:

    usage: mwrpy_ret/cli.py [-h] -s SITE [-d YYYY-MM-DD] [--start YYYY-MM-DD]
                           [--stop YYYY-MM-DD] [{radiosonde}]

Arguments:

| Short | Long         | Default           | Description                                                                          |
| :---- | :----------- | :---------------- | :----------------------------------------------------------------------------------- |
| `-s`  | `--site`     |                   | Site to process data from, e.g, `lindenberg`. Required.                              |
| `-d`  | `--date`     |                   | Single date to be processed. Alternatively, `--start` and `--stop` can be defined.   |
|       | `--start`    | `current day - 1` | Starting date.                                                                       |
|       | `--stop`     | `current day `    | Stopping date.                                                                       |

Commands:

| Command      | Description              |
| :--------    | :----------------------- |
| `radiosonde` | Process radiosonde data. |
