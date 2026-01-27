#!/usr/bin/env python3
"""
A standalone pipeline code for X-ray spectral shifting and stacking.

X-ray spectral stacking is non-trivial compared to optical spectral 
stacking. The difficulties arise from two facts: 

1) X-ray has much fewer photon counts (Poisson), meaning that spectral 
   counts and uncertainties cannot be scaled simultaneously (as compared 
   to optical); 
2) X-ray has non-diagonal, complex response, meaning that the response 
   needs to be taken into account when stacking.

To tackle these issues, we develop Xstack: a standalone pipeline code 
for X-ray spectral stacking. The methodology is to first sum all (rest-
frame) PI spectra, without any scaling; and then sum the response files 
(full response, ARFs*RMFs), with appropriate weighting factors to 
preserve the overall spectral shape. 

The key features of Xstack are: 
1) properly account account for individual spectral contribution to the 
   final stack, by assigning data-driven weighting factors for the 
   responses; 
2) preserve Poisson statistics; 
3) support Galactic absorption correction.


Examples
--------
Calling Xstack is simple. For this command line version, it is only a 
single-line task:

.. code-block::

	runXstack your_filelist.txt --prefix ./results/stacked_


And you will get: 
- stacked PI spectrum `./results/stacked_pi.fits`
- stacked background PI spectrum `./results/stacked_bkgpi.fits`
- stacked response files
	+ `./results/stacked_arf.fits`
	+ `./results/stacked_rmf.fits`
- and `./results/stacked_fene.fits`, which stores the first contributing 
  energy of each individual source. 

Or more sophisticatedly:

.. code-block::

	runXstack your_filelist.txt --prefix ./results/stacked_ \
	--rsp_weight_method SHP --rsp_proj_gamma 2.0 \
	--flux_energy_lo 1.0 --flux_energy_hi 2.3 --nthreads 20 \
	--ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf


If you want to do bootstrap, that is also easy:

.. code-block::

	runXstack your_filelist.txt --prefix ./results/stacked_ \
	--rsp_weight_method SHP --rsp_project_gamma 2.0 \
	--flux_energy_lo 1.0 --flux_energy_hi 2.3 --nthreads 20 \
	--ene_trc 0.2 --same_rmf AllSourcesUseSameRMF.rmf \
	--resample_method bootstrap --num_bootstrap 100


Please see below for the documentation of each argument:

"""
# Xstack main module
from Xstack.Xstack import XstackRunner
from Xstack.config import default_nh_file
# usual packages
import numpy as np
from astropy.io import fits as pyfits
import fitsio
import os
import sys
from pathlib import Path
from joblib import Parallel,delayed
import argparse
import warnings
from tqdm import tqdm


class HelpfulParser(argparse.ArgumentParser):
	def error(self, message):
		sys.stderr.write(f"error: {message}\n")
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
parser.add_argument("--extended", action="store_true", help="whether or not this is an extended source")
parser.add_argument("--same_rmf", type=str, default=None, help="specify the name of common rmf, if all sources are to use the same rmf")
parser.add_argument("--do_cache", action="store_true", help="save and load individual rest-frame files")
# below are for bootstrap
parser.add_argument("--bootstrap", action="store_true", help="activate bootstrap mode")
parser.add_argument("--num_bootstrap", type=int, default=10, help="number of bootstrap experiments")
parser.add_argument("--bootstrap_portion", type=float, default=1.0, help="portion of sources to resample in each bootstrap experiment")

args = parser.parse_args()


def read_entry(filename,same_rmf=None,check_bkg_arf=None):
	"""
	
	"""
	filename = filename.strip()
	if not filename:
		return None
	
	path = os.path.dirname(filename)

	try:
		with fitsio.FITS(filename) as ff:
			hdr = ff["SPECTRUM"].read_header()
	except Exception as err:
		warnings.warn(f"Cannot read {filename}: {err}")
		return 

	backfile = hdr.get("BACKFILE","")
	rmffile = hdr.get("RESPFILE","")
	arffile = hdr.get("ANCRFILE","")

	backfile = os.path.join(path,backfile) if backfile else None
	rmffile = os.path.join(path,rmffile)  if rmffile  else ""
	arffile = os.path.join(path,arffile)  if arffile  else ""
			
	#--- if all sources share the same RMF
	#--- you can manually point the `rmffile` to the common RMF"s path via --same_rmf argument:
	if same_rmf is not None:
		rmffile = same_rmf

	#--- read backfile
	if os.path.isfile(backfile):
		if check_bkg_arf:
			try:
				with fitsio.FITS(backfile) as bb:
					bhdr = bb["SPECTRUM"].read_header()
				if bhdr.get("ANCRFILE","none") not in ("none",hdr.get("ANCRFILE","")):
					warnings.warn(
						f"Background must have same ARF; but got {bhdr.get("ANCRFILE")} instead of {hdr.get("ANCRFILE")}"
					)
			except Exception as err:
				warnings.warn(f"Cannot read background {backfile}: {err}")
	else:
		backfile = None

	#--- read z, nh
	pz = Path(filename + ".z")
	pnh = Path(filename + ".nh")
	z = float(pz.read_text().strip()) if pz.exists() else 0.0
	nh = float(pnh.read_text().strip()) if pnh.exists() else 0.0

	return filename,backfile,arffile,rmffile,z,nh


def main():
	#--- parse input files
	with open(args.filelist,"r") as f:
		lines = f.readlines()
	filename_lst = [line.strip() for line in lines if line.strip()]
	results = Parallel(
		n_jobs=min(10,args.nthreads),
		backend="loky",
	)(
		delayed(read_entry)(filename,same_rmf=args.same_rmf,check_bkg_arf=True) 
		for filename in tqdm(filename_lst)
	)

	pifile_lst = []
	bkgpifile_lst = []
	arffile_lst = []
	rmffile_lst = []
	z_lst = []
	nh_lst = []

	for result in results:
		if result is None:
			continue
		filename,backfile,arffile,rmffile,z,nh = result
		if filename is not None:
			pifile_lst.append(filename)
			bkgpifile_lst.append(backfile)
			arffile_lst.append(arffile)
			rmffile_lst.append(rmffile)
			z_lst.append(z)
			nh_lst.append(nh)

	if np.all([bkgfile is None for bkgfile in bkgpifile_lst]):
		bkgpifile_lst = None

	XstackRunner(
		pifile_lst=pifile_lst,                          # the PI spectrum list
		arffile_lst=arffile_lst,                        # the ARF list
		rmffile_lst=rmffile_lst,                        # the RMF list
		z_lst=z_lst,                                    # the redshift list
		bkgpifile_lst=bkgpifile_lst,                    # the bkg PI files list
		nh_lst=nh_lst,                                  # the Galactic absorption list (optional, in units of 1 cm^{-2})
		srcid_lst=None,                                 # the source id list (optional)
		rspwt_method=args.rsp_weight_method,            # method to calculate response weighting factor for each source (recommended: SHP)
		rspproj_gamma=args.rsp_project_gamma,           # prior photon index for projecting RSP matrix onto the output energy channel
		int_rng=(args.flux_energy_lo,args.flux_energy_hi), # if `arfscal_method`=`SHP`, choose the range to calculate flux
		sample_rmf=None,                                # the sample RMF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
		sample_arf=None,                                # the sample ARF to read input/output energy bin edge (if not specified, the first RMF in `rmffile_lst` will be used)
		nh_file=default_nh_file,                        # the Galactic absorption profile (absorption factor vs. energy)
		Nbkggrp=args.num_bkg_groups,                    # the number of background groups to calculate uncertainty of background
		ene_trc=args.ene_trc,                           # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
		extended=args.extended,                         # extended sources?
		nthreads=args.nthreads,                         # number of cpus used for RMF shifting
		bootstrap=True if args.bootstrap else False,	# do single stacking or bootstrap stacking?
		num_bootstrap=args.num_bootstrap,               # number of bootstrap experiments in `bootstrap` method
		bootstrap_portion=args.bootstrap_portion,       # portion to resample in `bootstrap` method
		prefix=args.prefix,                             # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
		do_cache=True if args.do_cache else False,		# save and load individual rest-frame files
	).run()



if __name__ == "__main__":
	main()

