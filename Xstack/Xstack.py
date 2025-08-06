#!/usr/bin/env python3
"""
This is the main wrapper module for all spectral shifting+stacking procedures. 

Authors: Shi-Jiang Chen (MPE, USTC), Johannes Buchner (MPE), Teng Liu (USTC)
Contact: JohnnyCsj666@gmail.com

"""
from .shift_pi import *
from .shift_rsp import *
from .misc import fene_fits

import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed
from tqdm import tqdm
import gc
import os
default_nh_file = os.path.join(os.path.dirname(__file__), "tbabs_1e20.txt")
version_file = os.path.join(os.path.dirname(__file__), "VERSION")

with open(version_file) as f:
    lines = f.readlines()
    version = lines[0].strip()
    lastupdate = lines[1].strip()

##############################################
############# MAIN FUNCTION ##################
##############################################
class XstackRunner:
    """
    X-ray Spectral Shifting & Stacking.

    Example usage
    -------------
    ```python
    data = XstackRunner(
        pifile_lst = your_pifile_lst,
        arffile_lst = your_arffile_lst,
        rmffile_lst = your_rmffile_lst,
        z_lst = your_z_lst,
        bkgpifile_lst = your_bkgpifile_lst,
        prefix = './results/stacked_',
    )
    data.run()  # this will produce the stacked PI, bkgPI, ARF, RMF in one go
    ```
    """
    def __init__(
            self,pifile_lst,arffile_lst,rmffile_lst,z_lst,bkgpifile_lst=None,nh_lst=None,srcid_lst=None,rspwt_method="SHP",rspproj_gamma=2.0,int_rng=(1.0,2.3),sample_rmf=None,sample_arf=None,nh_file=None,Nbkggrp=10,ene_trc=None,nthreads=1,prefix="./results/stacked_",
        ):
        """
        Parameters
        ----------
        pifile_lst : list or numpy.ndarray
            The input PI spectrum file list.
        arffile_lst : list or numpy.ndarray
            The input ARF file list.
        rmffile_lst : list or numpy.ndarray
            The input RMF file list.
        z_lst : list or numpy.ndarray
            The redshift list.
        bkgpifile_lst : list or numpy.ndarray, optional
            The input background PI spectrum list. Defaults to None.
        nh_lst : list or numpy.ndarray, optional
            The Galactic absorption column density list (in units of 1 cm^{-2}). Defaults to None.
        srcid_lst : list or numpy.ndarray, optional
            The source ID list. Defaults to None.
        rspwt_method : str, optional
            Method for calculating ARFSCAL. Defaults to `SHP`. Available methods are:
            - `SHP`: assuming all sources have same spectral shape, recommended
            - `FLX`: assuming all sources have same flux (erg/s/cm^2)
            - `LMN`: assuming all sources have same luminosity (erg/s)
        rspproj_gamma : float, optional
            The prior photon index value for projecting RSP matrix onto the output energy channel. This is used in the `SHP` method, to calculate the weight of each response. Defaults to 2.0 (typical for AGN).
        int_rng : tuple of (float,float), optional
            The energy (keV) range for computing flux. Defaults to (1.0,2.3).
        sample_rmf : str, optional
            Name of sample RMF. Defaults to None.
        sample_arf : str, optional
            Name of sample ARF. Defaults to None.
        nh_file : str, optional
            Galactic absorption profile (absorption factor vs. energy) at 1e20 cm^{-2}. If specified, galactic absorption correction will be applied on the ARF before shifting.
            - Should be in txt format. 
            - Should also contain the following columns in the first extension: `nhene_ce`, `nhene_wd`, `factor`.
            - `factor` should indicate the absorption factor when nh=1e20.
            - An easy way to obtain the `nh_file`: iplot `tbabs*powerlaw` with `Nh`=1e20 and `PhoIndex`=0.0, `Norm`=1 in Xspec.
        Nbkggrp : int, optional
            Number of groups with similar background-to-source scaling ratio. Defaults to 10.
        ene_trc : float, optional
            Truncate energy below which manually set ARF and PI counts to zero. For eROSITA, `ene_trc` is typically set as 0.2 keV. Defaults to None.
        nthreads : int, optional
            Number of CPUs used in shifting RSP.
        prefix : str, optional
            Prefix for output stacked PI, BKGPI, ARF, and RMF files. Defaults to './results/stacked_'
        """
        self.pifile_lst = pifile_lst
        self.arffile_lst = arffile_lst
        self.rmffile_lst = rmffile_lst
        self.z_lst = z_lst
        self.bkgpifile_lst = bkgpifile_lst
        if nh_lst is not None:
            self.nh_lst = nh_lst
        else:
            self.nh_lst = np.zeros(len(pifile_lst))
        if srcid_lst is not None:
            self.srcid_lst = srcid_lst
        else:
            self.srcid_lst = np.arange(len(pifile_lst))
        self.rspwt_method = rspwt_method
        self.rspproj_gamma = rspproj_gamma
        self.int_rng = int_rng
        if sample_rmf is None:
            self.sample_rmf = rmffile_lst[0]
        else:
            self.sample_rmf = sample_rmf
        if sample_arf is None:
            self.sample_arf = arffile_lst[0]
        else:
            self.sample_arf = sample_arf
        self.nh_file = nh_file
        if Nbkggrp > len(pifile_lst):
            print("Warning! `Nbkggrp` must be smaller than the number of spectra loaded. `Nbkggrp` is now set to 1.")
            self.Nbkggrp = 1
        else:
            self.Nbkggrp = Nbkggrp
        self.ene_trc = ene_trc
        self.nthreads = nthreads

        # creating output directory
        self.outdir = os.path.dirname(prefix)
        if self.outdir != "":
            os.makedirs(self.outdir,exist_ok=True)

        prefix_base = os.path.basename(prefix)
        self.o_pi_name = f"{prefix_base}pi.fits"
        self.o_bkgpi_name = f"{prefix_base}bkgpi.fits"
        self.o_arf_name = f"{prefix_base}arf.fits"
        self.o_rmf_name = f"{prefix_base}rmf.fits"
        self.o_fene_name = f"{prefix_base}fene.fits"

        # creating empty lists to store results
        self.pi_sft_lst = []
        self.rspmat_sft_lst = []
        self.bkgpi_sft_lst = []

        self.bkgscal_lst = []
        self.expo_lst = []
        self.arffene_lst = []
        self.fene_lst = []

    def run(self):
        """
        Shift all PIs + bkgPIs + ARFs + RMFs to rest-frame in one go.
        """
        print("#######################################################")
        print("################ Welcome to Xstack! ###################")
        print("#######################################################")
        print(f"Version: {version}")
        print(f"Last updated: {lastupdate}")
        print("******************* Input Summary *********************")
        print(f"Number of sources: {len(self.pifile_lst)}")
        print(f"Redshift range: {np.min(self.z_lst):.3f} -- {np.max(self.z_lst):.3f}")
        print(f"NH range: {np.min(self.nh_lst)} -- {np.max(self.nh_lst)}")
        print(f"NH file: {self.nh_file if self.nh_file is not None else 'None'}")
        print(f"RSP weighting method: {self.rspwt_method}")
        print(f"RSP projection gamma: {self.rspproj_gamma}")
        print(f"Flux calculation range: {self.int_rng[0]} -- {self.int_rng[1]} keV (used only in `SHP` mode)")
        print(f"ARF Truncation energy: {self.ene_trc} keV")
        print(f"Number of CPUs used for shifting RMF: {self.nthreads}")
        print(f"Number of background groups: {self.Nbkggrp}")
        print(f"Output directory: {self.outdir}")
        print(f"Output PI spectrum (base)name: {os.path.join(self.outdir,self.o_pi_name)}")
        print(f"Output bkg PI spectrum (base)name: {os.path.join(self.outdir,self.o_bkgpi_name)}")
        print(f"Output ARF (base)name: {os.path.join(self.outdir,self.o_arf_name)}")
        print(f"Output RMF (base)name: {os.path.join(self.outdir,self.o_rmf_name)}")
        print(f"Output FENE (base)name: {os.path.join(self.outdir,self.o_fene_name)}")
        print("*******************************************************")
        
        with fits.open(self.sample_rmf) as hdu:
            mat = hdu["MATRIX"].data
            ebo = hdu["EBOUNDS"].data
        ene_lo = ebo["E_MIN"]
        ene_hi = ebo["E_MAX"]
        iene_lo = mat["ENERG_LO"]
        iene_hi = mat["ENERG_HI"]

        del hdu["MATRIX"].data,hdu["EBOUNDS"].data  # to clear memory
        
        # SHIFTING
        print("")
        print("******************* Shifting ... **********************")
        ## use backend="loky" to avoid memory leakage
        results = Parallel(n_jobs=self.nthreads,backend="loky")(delayed(self.process_entry)(i) for i in tqdm(range(len(self.srcid_lst))))
        for result in results:
            pi_sft, bkgpi_sft, rspmat_sft, bkgscal, expo, arffene, fene = result
            self.pi_sft_lst.append(pi_sft)
            self.bkgpi_sft_lst.append(bkgpi_sft)
            self.rspmat_sft_lst.append(rspmat_sft)
            self.bkgscal_lst.append(bkgscal)
            self.expo_lst.append(expo)
            self.arffene_lst.append(arffene)
            self.fene_lst.append(fene)
        del results

        # STACKING
        print("")
        print("******************* Stacking ... **********************")
        expo = np.sum(self.expo_lst)
        pi_stk,pierr_stk = add_pi(
            self.pi_sft_lst,fits_name=self.o_pi_name,expo=expo,bkg_file=self.o_bkgpi_name,rmf_file=self.o_rmf_name,arf_file=self.o_arf_name,
        )
        bkgpi_stk,bkgpierr_stk = add_bkgpi(
            self.bkgpi_sft_lst,bkgscal_lst=self.bkgscal_lst,Ngrp=self.Nbkggrp,fits_name=self.o_bkgpi_name,expo=expo,
        )
        arf_stk, rmf_stk = add_rsp(
            self.rspmat_sft_lst,self.pi_sft_lst,self.z_lst,bkgpi_lst=self.bkgpi_sft_lst,bkgscal_lst=self.bkgscal_lst,ene_lo=ene_lo,ene_hi=ene_hi,arfene_lo=iene_lo,arfene_hi=iene_hi,expo_lst=self.expo_lst,int_rng=self.int_rng,rspwt_method=self.rspwt_method,rspproj_gamma=self.rspproj_gamma,outarf_name=self.o_arf_name,sample_arf=self.sample_arf,srcid_lst=self.srcid_lst,outrmf_name=self.o_rmf_name,sample_rmf=self.sample_rmf
        )
        
        if self.o_fene_name is not None:
            fene_fits(self.srcid_lst,self.arffene_lst,self.fene_lst,self.o_fene_name)

        # Move all output files to outdir
        if self.outdir != "":
            os.system(f"mv {self.o_pi_name} {self.o_bkgpi_name} {self.o_arf_name} {self.o_rmf_name} {self.o_fene_name} {self.outdir}")

        del self.rspmat_sft_lst # to clear memory

        print("")
        print(f"#######################################################")
        print(f"########## Stacking {len(self.srcid_lst)} spectra completed! ###########")
        print(f"#######################################################")
        print(f"Stacked PI spectrum saved to: {os.path.join(self.outdir,self.o_pi_name)}")
        print(f"Stacked BKGPI spectrum saved to: {os.path.join(self.outdir,self.o_bkgpi_name)}")
        print(f"Stacked ARF saved to: {os.path.join(self.outdir,self.o_arf_name)}")
        print(f"Stacked RMF saved to: {os.path.join(self.outdir,self.o_rmf_name)}")
        if self.o_fene_name is not None:
            print(f"Stacked FENE saved to: {os.path.join(self.outdir,self.o_fene_name)}")
        print("")
        print(f"# NOTE: the output stacked spectra have {{BACK,AREA,CORR}}SCAL=1, even though the inputs have different ratios. This is because these information have already gone into the background spectrum by scaling it.")
        print("")
        
        return pi_stk, pierr_stk, bkgpi_stk, bkgpierr_stk, arf_stk, rmf_stk
    

    ###### internal function #####
    def process_entry(self,i):
        pifile = self.pifile_lst[i]
        bkgpifile = self.bkgpifile_lst[i]
        arffile = self.arffile_lst[i]
        rmffile = self.rmffile_lst[i]
        z = self.z_lst[i]
        nh = self.nh_lst[i]

        # pi shifting
        (pi_chan_sft,pi_coun_sft,pi_chan,pi_coun) = shift_pi(pifile,self.sample_rmf,z,self.ene_trc)
        pi_sft = pi_coun_sft.astype("float64")
        # BKGpi shifting
        if self.bkgpifile_lst is None:
            bkgpi_coun_sft = np.zeros(len(pi_coun_sft))
        else:
            (bkgpi_chan_sft,bkgpi_coun_sft,bkgpi_chan,bkgpi_coun) = shift_pi(bkgpifile,self.sample_rmf,z,self.ene_trc)
        bkgpi_sft = bkgpi_coun_sft.astype("float64")
        # RSP shifting
        rspmat_sft = shift_rsp(arffile,rmffile,z,self.nh_file,nh=nh,ene_trc=self.ene_trc)
        
        # first energy
        with fits.open(rmffile) as hdu:
            mat = hdu["MATRIX"].data
            ebo = hdu["EBOUNDS"].data
        arfene_lo = mat["ENERG_LO"]
        arfene_hi = mat["ENERG_HI"]
        arfene_ce = (arfene_lo + arfene_hi) / 2
        arfene_wd = arfene_hi - arfene_lo
        ene_lo = ebo["E_MIN"]
        ene_hi = ebo["E_MAX"]
        ene_ce = (ene_lo + ene_hi) / 2
        ene_wd = ene_hi - ene_lo
        arf_sft = project_rspmat(rspmat_sft,ene_lo,ene_hi,arfene_lo,arfene_hi,proj_axis="MODEL")

        arf_nonzero_mask = (arf_sft!=0)
        arffene = arfene_ce[arf_nonzero_mask][0]
        pi_nonzero_mask = (pi_sft!=0)
        fene = ene_ce[pi_nonzero_mask][0] if pi_nonzero_mask.any() else -1
        
        del hdu["MATRIX"].data, hdu["EBOUNDS"].data  # to clear memory

        # EXPO & BKGSCAL
        bkgscal = get_bkgscal(pifile,bkgpifile)
        expo = get_expo(pifile)
        
        gc.collect()

        return pi_sft, bkgpi_sft, rspmat_sft, bkgscal, expo, arffene, fene
    ##############################
