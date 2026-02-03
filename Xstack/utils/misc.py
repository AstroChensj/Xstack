#!/usr/bin/env python3
"""

"""
import os
import re


def get_nh(RA,DEC):
    """
    Get the Galactic NH from NASA"s HEASARC tool `NH` (https://heasarc.gsfc.nasa.gov/Tools/w3nh_help.html).
    Please ensure the HEASOFT env has been set up.
    NOTE: this is deprecated. Please use `gdpyc.GasMap` from Github instead.
    
    Parameters
    ----------
    RA : float

    DEC : float

    
    Returns
    -------
    nh_val : float
        nh values in units of 1 cm^-2
    """
    # write sh
    log_file = "nh.log"
    os.system("rm -rf %s"%log_file)
    shell_file = open("run_nh.sh","w",newline="")
    shell_file.writelines("(\n")
    shell_file.writelines("echo 2000\n") # Equinox (d/f 2000)
    shell_file.writelines("echo %f\n"%RA) # RA in hh mm ss.s or degrees
    shell_file.writelines("echo %f\n"%DEC) # DEC in hh mm ss.s or degrees
    shell_file.writelines(") | nh\n")
    shell_file.close()
    # run sh
    os.system("bash run_nh.sh > %s 2>&1"%log_file)
    # read sh
    with open(log_file,"r") as file:
        text = file.read()
    # Use regular expression to find the line with "Weighted" and capture the value
    match1 = re.search(r"Weighted average nH \(cm\*\*-2\)\s+([0-9.E+-]+)", text)
    match2 = re.search(r"h1_nh_HI4PI.fits >> nH \(cm\*\*-2\)\s+([0-9.E+-]+)", text) # in case when the given RA/DEC falls outside the allowed range
    if match1:
        nh_val = match1.group(1)
    elif match2:
        nh_val = match2.group(1)
    else:
        raise Exception("Invalid RA (%.4f), DEC(%.4f) for nh!"%(RA,DEC))
    nh_val = float(nh_val)

    return nh_val


def pygrppha(src_name,grp_name,grpmin=25):
    with open("grppha.sh","w") as shell_file:
        shell_file.writelines("rm -rf %s\n"%(grp_name))
        shell_file.writelines("(\n")
        shell_file.writelines("echo %s\n"%(src_name))
        shell_file.writelines("echo %s\n"%(grp_name))
        shell_file.writelines("echo group min %d\n"%(grpmin))
        shell_file.writelines("echo exit\n")
        shell_file.writelines(") | grppha\n")
    os.system("bash grppha.sh > grppha.log 2>&1")
    return