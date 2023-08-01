from mwrpy_ret.rad_trans.run_rad_trans import rad_trans_rs
from mwrpy_ret.utils import get_file_list

file_list = get_file_list("/home/tmarke/Dokumente/GitHub/mwrpy_ret/tests/data")

for file in file_list:
    tb, T = rad_trans_rs(file)
    print(tb)
