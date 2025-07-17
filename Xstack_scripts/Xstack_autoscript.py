#!/usr/bin/env python3
"""A comprehensive standalone pipeline code for X-ray spectral (rest-frame) shifting and stacking.

X-ray spectral stacking is non-trivial compared to optical spectral stacking. The difficulties arise from two facts: 
1) X-ray has much fewer photon counts (Poisson), meaning that spectral counts and uncertainties cannot be scaled simultaneously (as compared to optical); 
2) X-ray has non-diagonal, complex response, meaning that the response needs to be taken into account when stacking.

To tackle these issues, we develop Xstack: a open-source, and comprehensive standalone pipeline code for X-ray spectral stacking. The methodology is to first sum all (rest-frame) PI spectra, without any scaling; and then sum the response files (ARFs and RMFs), each with appropriate weighting factors to preserve the overall spectral shape.

The key features of Xstack are: 
1) properly account account for individual spectral contribution to the final stack, by assigning data-driven ARF weighting factors; 
2) preserve Poisson statistics; 
3) support Galactic absorption correction.


Examples
--------
Calling Xstack is simple. For this command line version, it is only a single-line task:

```shell
runXstack your_filelist.txt --prefix ./results/stacked_
```

And you will get the stacked PI spectrum `./results/stacked_pi.fits`, stacked background PI spectrum `./results/stacked_bkgpi.fits`, stacked response files `./results/stacked_arf.fits`, `./results/stacked_rmf.fits`, and `./results/stacked_fene.fits` which stores the first contributing energy of each individual source. Or more sophisticatedly:

```shell
runXstack your_filelist.txt --prefix ./results/stacked_ --rsp_weight_method SHP --rsp_proj_gamma 2.0 --flux_energy_lo 1.0 --flux_energy_hi 2.3 --nthreads 20 --ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf
```

If you want to do bootstrap, that is also easy:

```shell
runXstack your_filelist.txt --prefix ./results/stacked_ --rsp_weight_method SHP --rsp_project_gamma 2.0 --flux_energy_lo 1.0 --flux_energy_hi 2.3 --nthreads 20 --ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf --resample_method bootstrap --num_bootstrap 100
```

Please see below for the documentation of each argument:

"""

# the main module
from Xstack import Xstack,rsXstack
# the usual packages
import numpy as np
from astropy.io import fits as pyfits
import os
import shutil
import sys
import argparse

nh_file = Xstack.default_nh_file
script_path = os.path.abspath(__file__) # the absolute path of Xstack_autoscript.py

class HelpfulParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write("error: %s\n" % message)
		self.print_help()
		sys.exit(2)

parser = HelpfulParser(description=__doc__,
	epilog="""Shi-Jiang Chen, Johannes Buchner and Teng Liu (C) 2025 <JohnnyCsj666@gmail.com>""",
    formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument("filelist", type=str, help="text file containing the file names")
parser.add_argument("--prefix", type=str, default="./results/stacked_", help="prefix for output stacked PI, BKGPI, ARF, and RMF files; defaults to './results/stacked_'")
parser.add_argument("--rsp_weight_method", type=str, default="SHP", help="method to calculate RSP weighting factor for each source; 'SHP': assuming all sources have same spectral shape (only this mode would require flux_energy_lo and flux_energy_hi), 'FLX': assuming all sources have same shape and energy flux (weigh by exposure time), 'LMN': assuming all sources have same shape and luminosity (weigh by exposure/dist^2); defaults to 'SHP'")
parser.add_argument("--rsp_project_gamma", type=float, default=2.0, help="prior photon index value for projecting RSP matrix onto the output energy channel. This is used in the `SHP` method, to calculate the weight of each response; defaults to 2.0 (typical for AGN).")
parser.add_argument("--flux_energy_lo", type=float, default=1.0, help="lower end of the energy range in keV for computing flux, used only when `rsp_weight_method`=`SHP`; defaults to 1.0")
parser.add_argument("--flux_energy_hi", type=float, default=2.3, help="upper end of the energy range in keV for computing flux; used only when `rsp_weight_method`=`SHP`; defaults to 2.3")
parser.add_argument("--nthreads", type=int, default=10, help="number of cpus used for RMF shifting")
parser.add_argument("--num_bkg_groups", type=int, default=10, help="number of background groups")
parser.add_argument("--ene_trc", type=float, default=0.0, help="energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)")
parser.add_argument("--same_rmf", type=str, default=None, help="specify the name of common rmf, if all sources are to use the same rmf")
# below are for bootstrap (either bootstrap or KFold)
parser.add_argument("--resample_method", type=str, default="None", help="method for performing resampling; 'None': no resampling, 'bootstrap': use bootstrap, 'KFold': use KFold)")
parser.add_argument("--num_bootstrap", type=int, default=10, help="number of bootstrap experiments")
parser.add_argument("--bootstrap_portion", type=float, default=1.0, help="portion of sources to resample in each bootstrap experiment")
parser.add_argument("--Ksort_filelist", type=str, default="Ksort_filelist.txt", help="name of file storing the sorting value for each source in `filelist`")
parser.add_argument("--K", type=int, default=4, help="number of groups for KFold")

args = parser.parse_args()


def main():
    pifile_lst = []
    bkgpifile_lst = []
    arffile_lst = []
    rmffile_lst = []
    z_lst = []
    nh_lst = []

    # read files
    for line in open(args.filelist):
        filename = line.rstrip()
        print("checking ...", filename)
        
        path = os.path.dirname(filename)
        a = pyfits.open(filename)
        header = a["SPECTRUM"].header
        backfile = os.path.join(path, header["BACKFILE"])
        rmffile = os.path.join(path, header["RESPFILE"])
        # if all sources share the same RMF, you can manually point the `rmffile` to the common RMF"s path via --same_rmf argument:
        if args.same_rmf is not None:
              rmffile = args.same_rmf
        arffile = os.path.join(path, header["ANCRFILE"])

        b = pyfits.open(backfile)
        bheader = b["SPECTRUM"].header
        assert "ANCRFILE" not in bheader or bheader["ANCRFILE"] == header["ANCRFILE"], "background must have same ARF"
        
        z = float(open(filename + ".z").read())
        z_lst.append(z)
        nh = float(open(filename + ".nh").read())
        nh_lst.append(nh)
        pifile_lst.append(filename)
        bkgpifile_lst.append(backfile)
        arffile_lst.append(arffile)
        rmffile_lst.append(rmffile)


    if args.num_bkg_groups > len(pifile_lst):
          print("Warning! `Nbkggrp` must be smaller than the number of spectra loaded. `Nbkggrp` is now set to 1.")
          args.num_bkg_groups = 1


    check_your_output_dir = os.path.join(os.path.dirname(args.prefix),"check_your_output")
    os.makedirs(check_your_output_dir,exist_ok=True)


    if args.resample_method == "None":    # no resampling: single stack
        data = Xstack.XstackRunner(
            pifile_lst=pifile_lst,                          # the PI spectrum list
            arffile_lst=arffile_lst,                        # the ARF list
            rmffile_lst=rmffile_lst,                        # the RMF list
            z_lst=z_lst,                                    # the redshift list
            bkgpifile_lst=bkgpifile_lst,                    # the bkg PI files list
            nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                 # the source id list (optional)
            rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
            rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            sample_rmf=None,                                # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                                # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                                # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                    # the number of background groups to calculate uncertainty of background
            ene_trc=args.ene_trc,                           # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
            nthreads=args.nthreads,                         # number of cpus used for RMF shifting
            prefix=args.prefix,                             # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
        ).run()


    elif args.resample_method == "bootstrap":     # resampling, using bootstrap
        data = rsXstack.resample_XstackRunner(
            pifile_lst=pifile_lst,                          # the PI spectrum list
            arffile_lst=arffile_lst,                        # the ARF list
            rmffile_lst=rmffile_lst,                        # the RMF list
            z_lst=z_lst,                                    # the redshift list
            bkgpifile_lst=bkgpifile_lst,                    # the bkg PI spectrum list
            nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                 # the source id list (optional)
            rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
            rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            sample_rmf=None,                                # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                                # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                                # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                    # the number of background groups to calculate uncertainty of background
            ene_trc=args.ene_trc,                           # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
            nthreads=args.nthreads,                         # number of cpus used for RMF shifting
            resample_method=args.resample_method,           # resample method: `bootstrap` or `KFold`
            num_bootstrap=args.num_bootstrap,               # number of bootstrap experiments in `bootstrap` method
            bootstrap_portion=args.bootstrap_portion,       # portion to resample in `bootstrap` method
            prefix=args.prefix,                             # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
        ).run()


    elif args.resample_method == "KFold":     # resampling, using K-Fold
        Ksort_lst = []
        with open(args.Ksort_filelist) as f:
                lines = f.readlines()
        for line in lines:
                Ksort_lst.append(float(line.strip("\n")))
        Ksort_lst = np.array(Ksort_lst)
        assert len(Ksort_lst)==len(pifile_lst), "`Ksort_filelist` must have same length as `pifile_lst`!"

        data = rsXstack.resample_XstackRunner(
            pifile_lst=pifile_lst,                          # the PI spectrum list
            arffile_lst=arffile_lst,                        # the ARF list
            rmffile_lst=rmffile_lst,                        # the RMF list
            z_lst=z_lst,                                    # the redshift list
            bkgpifile_lst=bkgpifile_lst,                    # the bkg PI spectrum list
            nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
            srcid_lst=None,                                 # the source id list (optional)
            rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
            rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
            int_rng=(args.flux_energy_lo, args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
            sample_rmf=None,                                # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            sample_arf=None,                                # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
            nh_file=nh_file,                                # the Galactic absorption profile (absorption factor vs. energy)
            Nbkggrp=args.num_bkg_groups,                    # the number of background groups to calculate uncertainty of background
            ene_trc=args.ene_trc,                           # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
            nthreads=args.nthreads,                         # number of cpus used for RMF shifting
            resample_method=args.resample_method,           # resample method: `bootstrap` or `KFold`
            K=args.K,                                       # number of subgroups to divide the original sample into in `KFold` method
            Ksort_lst=Ksort_lst,                            # value list  used to sort the original sample in `KFold` method
            prefix=args.prefix,                             # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
        ).run()


    else:
        raise Exception("Available `resample_method`: `None`, `bootstrap`, or `KFold`!")


if __name__ == "__main__":
    main()

