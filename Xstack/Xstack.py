#!/usr/bin/env python3
"""
=================================================================
Main wrapper module for all spectral shifting+stacking procedures
=================================================================
:Authors:   Shi-Jiang Chen (MPE, USTC)
            Johannes Buchner (MPE)
            Teng Liu (USTC)
:Email:     JohnnyCsj666@gmail.com

"""
import numpy as np
from astropy.io import fits
from joblib import Parallel,delayed
from tqdm import tqdm
import os
from pathlib import Path
import time
from Xstack.utils.pi import read_pi,shift_pi,get_bkgscal,get_expo,get_rega,calc_pi_error,calc_bkgpi_error,write_pi,write_bkgpi
from Xstack.utils.rsp import read_rsp,shift_rsp,get_prob,compute_rspwt,rescale_rspmat,project_rspmat,extract_arf_rmf_from_rspmat,get_tlmin_from_header,write_arf,write_rmf
from Xstack.utils.fene import write_fene
from Xstack.utils.random import calc_bootstrap_weights
from Xstack.utils.logger import get_logger,get_ram_gb
from Xstack.config import VERSION,LASTUPDATE


class XstackRunner:
    """
    X-ray Spectral Shifting & Stacking.

    Example usage
    -------------
    .. code-block::

        data = XstackRunner(
            pifile_lst = your_pifile_lst,
            arffile_lst = your_arffile_lst,
            rmffile_lst = your_rmffile_lst,
            z_lst = your_z_lst,
            bkgpifile_lst = your_bkgpifile_lst,
            prefix = './results/stacked_',
        )
        data.run()  
        # this will produce the stacked PI, bkgPI, ARF, RMF in one go

    """
    def __init__(
            self,pifile_lst,arffile_lst,rmffile_lst,z_lst,
            bkgpifile_lst=None,nh_lst=None,srcid_lst=None,
            rspwt_method="SHP",rspproj_gamma=2.0,int_rng=(1.0,2.3),
            sample_rmf=None,sample_arf=None,nh_file=None,
            Nbkggrp=10,ene_trc=None,extended=False,nthreads=1,
            bootstrap=False,num_bootstrap=10,bootstrap_portion=1.0,
            prefix="./results/stacked_",
            do_cache=False,
        ):
        """
        Initialize Xstack.

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
            The Galactic absorption column density list in units of 
            1 cm^{-2}. Defaults to None.

        srcid_lst : list or numpy.ndarray, optional
            The source ID list. Defaults to None.

        rspwt_method : str, optional
            Method for calculating ARFSCAL. Defaults to `SHP`. Available 
            methods are:
            - `SHP`: assuming all sources have same spectral shape, 
              recommended
            - `FLX`: assuming all sources have same spectral shape and 
              flux ([erg/cm^2/s] for point sources while 
              [erg/cm^2/s/deg^2] for extended sources)
            - `LMN`: assuming all sources have same spectral shape and 
              luminosity ([erg/s] for point sources while [erg/s/deg^2] 
              for extended sources)

        rspproj_gamma : float, optional
            The prior photon index value for projecting RSP matrix onto 
            the output energy channel. This is used in the `SHP` method, 
            to calculate the weight of each response. Defaults to 2.0 
            (typical for AGN).

        int_rng : tuple of (float,float), optional
            The energy (keV) range for computing flux under `SHP` mode. 
            Defaults to (1.0,2.3).

        sample_rmf : str, optional
            Name of sample RMF. Defaults to None.

        sample_arf : str, optional
            Name of sample ARF. Defaults to None.

        nh_file : str, optional
            Galactic absorption profile (absorption factor vs. energy) 
            at 1e20 cm^{-2}. If specified, galactic absorption correction 
            will be applied on the ARF before rest-frame shifting.
            - Should be in .txt format. 
            - Should also contain the following columns in the first 
              extension: `nhene_ce`, `nhene_wd`, `factor`.
            - `factor` should indicate the absorption factor when nh=1e20.
            - An easy way to obtain the `nh_file`: iplot `tbabs*powerlaw` 
              with `Nh`=1e20 and `PhoIndex`=0.0, `Norm`=1 in Xspec.

        Nbkggrp : int, optional
            Number of groups with similar background-to-source scaling 
            ratio. Defaults to 10.

        ene_trc : float, optional
            Truncate energy below which manually set ARF and PI counts to 
            zero. For eROSITA, `ene_trc` is typically set as 0.2 keV. 
            Defaults to None.

        extended : bool, optional
            Whether or not are the sources to be stacked extended sources. 
            The calculation of response weights would be affected. 
            Defaults to False, i.e., they are point sources.

        nthreads : int, optional
            Number of CPUs used in shifting RSP.

        bootstrap : bool, optional
            Whether or not to do bootstrap. Defaults to False.

        num_bootstrap : int, optional
            Number of bootstrap realizations when `bootstrap` mode is activated.
            Defaults to 10.

        bootstrap_portion : float, optional
            Fraction of sources to be sampled in each bootstrap realization.
            Defaults to 1.

        prefix : str, optional
            Prefix for output stacked PI, BKGPI, ARF, and RMF files. 
            Defaults to './results/stacked_'.

        do_cache : bool, optional
            Whether or not to store and read the intermediate rest-frame files 
            for each individual source. If true, source-wise rest-frame files 
            will be saved under the same path as `pifile`. This will save a lot 
            of running time, at the cost of additional disk space (and burden on
            I/O). Defaults to False.
        """
        #--- create output directory and define logger
        self.outdir = os.path.dirname(prefix)
        if self.outdir != "":
            os.makedirs(self.outdir,exist_ok=True)
        self.logger_fname = f"{prefix}runXstack.log"
        self.main_logger = get_logger(self.logger_fname)

        #--- basic setup
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
            self.main_logger.warning("Warning! `Nbkggrp` must be smaller than the number of spectra loaded. `Nbkggrp` is now set to 1.")
            self.Nbkggrp = 1
        else:
            self.Nbkggrp = Nbkggrp
        self.ene_trc = ene_trc
        self.extended = extended
        self.nthreads = nthreads
        self.Nsrc = len(pifile_lst)
        self.do_cache = do_cache

        #--- read ARF energy edges and RMF energy edges
        ##--- NOTE: this assumes RMF energies are identical across the sample (i.e., from the same instrument) !!!
        self.main_logger.info("Please ensure all spectra share the same energy grids!")
        with fits.open(self.sample_rmf) as hdu:
            mat = hdu["MATRIX"].data
            ebo = hdu["EBOUNDS"].data
        self.ENE_LO = ebo["E_MIN"]
        self.ENE_HI = ebo["E_MAX"]
        self.ENE_CE = (self.ENE_LO + self.ENE_HI) / 2
        self.ENE_WD = self.ENE_HI - self.ENE_LO
        self.IENE_LO = mat["ENERG_LO"]
        self.IENE_HI = mat["ENERG_HI"]
        self.IENE_CE = (self.IENE_LO + self.IENE_HI) / 2
        self.IENE_WD = self.IENE_HI - self.IENE_LO
        self.F_CHAN_0 = get_tlmin_from_header(rmf_fname=self.sample_rmf)
        self.PROB = get_prob(mat=mat,ebo=ebo,f_chan_0=self.F_CHAN_0)
        self.int_flg = (self.ENE_CE>self.int_rng[0]) & (self.ENE_CE<self.int_rng[-1])
        # TODO: all energy edges merge into a single dict?

        #--- read PI for channel
        with fits.open(self.pifile_lst[0]) as hdu:
            data = hdu["SPECTRUM"].data
        self.CHANNEL = data["CHANNEL"]

        #--- bootstrap setting
        self.bootstrap = bootstrap
        if self.bootstrap:  # if you prefer to do bootstrap
            self.num_bootstrap = num_bootstrap
            self.bootstrap_portion = bootstrap_portion
            rng = np.random.default_rng(seed=self.num_bootstrap)
            self.bwt_lst_rlz = [
                calc_bootstrap_weights(Nsrc=self.Nsrc,bootstrap_portion=self.bootstrap_portion,rng=rng)
                for _ in range(self.num_bootstrap)
            ]   # realizations of bootstrap weight list, (num_bootstrap, Nsrc) 
        else:   # if you prefer not to do bootstrap, just direct sum
            self.num_bootstrap = 1  # set num_bootstrap to 1
            self.bwt_lst_rlz = [np.ones(self.Nsrc,dtype=np.int64)]  # and bootstrap weight list to 1s
        
        #--- source id list
        self.srcid_lst_rlz = [np.repeat(self.srcid_lst,self.bwt_lst_rlz[k]) for k in range(self.num_bootstrap)]

        #--- create empty arrays to store results
        ##--- NOTE: `lst` means list for sources (Nsrc), `rlz` means realization for bootstrap
        self.bkgscal_lst_rlz = [[] for _ in range(self.num_bootstrap)]      # realizations of bkgscal list, (num_bootstrap, Nsrc)
        self.rspwt_lst_rlz = [[] for _ in range(self.num_bootstrap)]        # realizations of response weight list, (same)
        self.expo_lst_rlz = [[] for _ in range(self.num_bootstrap)]         # realizations of exposure list, (same)
        self.rega_lst_rlz = [[] for _ in range(self.num_bootstrap)]         # realizations of region geometric area list, (same)
        self.arffene_lst_rlz = [[] for _ in range(self.num_bootstrap)]      # realizations of arf first energy list, (same)
        self.fene_lst_rlz = [[] for _ in range(self.num_bootstrap)]         # realizations of pi first energy list, (same)
        self.pi_totcts_lst_rlz = [[] for _ in range(self.num_bootstrap)]    # realizations of per-source total spectral counts (same)
        self.bkgpi_totcts_lst_rlz = [[] for _ in range(self.num_bootstrap)] # realizations of per-source total scaled bkg counts (same)
        self.bkgpi_sft_lst_rlz = [[] for _ in range(self.num_bootstrap)]    # for bkg uncertainty estimation, (num_bootstrap, Nsrc, Nchan)
        ##--- below are for accumulating
        self.pi_stk_rlz = [np.zeros_like(self.ENE_CE,dtype=np.float64) for _ in range(self.num_bootstrap)]      # realizations of stacked pi spectrum (num_bootstrap, Nchan)
        self.bkgpi_stk_rlz = [np.zeros_like(self.ENE_CE,dtype=np.float64) for _ in range(self.num_bootstrap)]   # realizations of stacked bkgpi spectrum (same)
        self.rspmat_stk_rlz = [np.zeros_like(self.PROB,dtype=np.float64) for _ in range(self.num_bootstrap)]    # realizations of stacked full response (num_bootstrap, Niene, Nene)
        # ** np.float64 is very important for LMN mode where rspmat value is very small **
        ##--- below will be overwritten later, so set to None
        self.pierr_stk_rlz = [None for _ in range(self.num_bootstrap)]
        self.bkgpierr_stk_rlz = [None for _ in range(self.num_bootstrap)]
        self.specresp_stk_rlz = [None for _ in range(self.num_bootstrap)]   # realizations of stacked ARF effective area curve, (num_bootstrap, Niene)
        self.prob_stk_rlz = [None for _ in range(self.num_bootstrap)]       # realizations of stacked RMF probability matrix, (num_bootstrap, Niene, Nene)
        self.rspnorm_rlz = [None for _ in range(self.num_bootstrap)]        # realizations of stacked response norm (num_bootstrap,)
        self.expo_stk_rlz = [None for _ in range(self.num_bootstrap)]       # realizations of stacked exposure (num_bootstrap,)
        self.rega_stk_rlz = [None for _ in range(self.num_bootstrap)]       # realizations of stacked region area (num_bootstrap,)

        #--- output name
        if self.bootstrap:
            self.o_pi_fname_rlz = [f"{prefix}{idx:0{len(str(self.num_bootstrap))}d}_pi.fits" for idx in range(self.num_bootstrap)]
            self.o_bkgpi_fname_rlz = [f"{prefix}{idx:0{len(str(self.num_bootstrap))}d}_bkgpi.fits" for idx in range(self.num_bootstrap)]
            self.o_arf_fname_rlz = [f"{prefix}{idx:0{len(str(self.num_bootstrap))}d}_arf.fits" for idx in range(self.num_bootstrap)]
            self.o_rmf_fname_rlz = [f"{prefix}{idx:0{len(str(self.num_bootstrap))}d}_rmf.fits" for idx in range(self.num_bootstrap)]
            self.o_fene_fname_rlz = [f"{prefix}{idx:0{len(str(self.num_bootstrap))}d}_fene.fits" for idx in range(self.num_bootstrap)]
        else:
            self.o_pi_fname_rlz = [f"{prefix}pi.fits"]
            self.o_bkgpi_fname_rlz = [f"{prefix}bkgpi.fits"]
            self.o_arf_fname_rlz = [f"{prefix}arf.fits"]
            self.o_rmf_fname_rlz = [f"{prefix}rmf.fits"]
            self.o_fene_fname_rlz = [f"{prefix}fene.fits"]
        
       

    def run(self):
        """
        Shift all PIs + bkgPIs + ARFs + RMFs to rest-frame and stack in one go.

        Returns
        -------
        pi_stk_rlz : numpy.ndarray
            Stacked PI spectra from all realizations, with shape (num_bootstrap, Nchan).
            Note that if `bootstrap`==False, the shape is simply (1, Nchan).

        pierr_stk_rlz : numpy.ndarray
            Stacked PI err spectra from all realizations, with shape (num_bootstrap, Nchan).
            Note that if `bootstrap`==False, the shape is simply (1, Nchan).

        bkgpi_stk_rlz : numpy.ndarray
            Stacked bkg PI spectra from all realizations, with shape (num_bootstrap, Nchan).
            Note that if `bootstrap`==False, the shape is simply (1, Nchan).

        bkgpierr_stk_rlz : numpy.ndarray
            Stacked bkg PI err spectra from all realizations, with shape (num_bootstrap, Nchan).
            Note that if `bootstrap`==False, the shape is simply (1, Nchan).

        specresp_stk_rlz : numpy.ndarray
            Stacked ARFs from all realizations, with shape (num_bootstrap, Niene).
            Note that if `bootstrap`==False, the shape is simply (1, Niene).

        prob_stk_rlz : numpy.ndarray
            Stacked RMF matrices from all realizations, with shape (num_bootstrap, Niene, Nene).
            Note that if `bootstrap`==False, the shape is simply (1, Niene, Nene).
        """
        self.main_logger.info("#######################################################")
        self.main_logger.info("################ Welcome to Xstack! ###################")
        self.main_logger.info("#######################################################")
        self.main_logger.info(f"Version: {VERSION}")
        self.main_logger.info(f"Last updated: {LASTUPDATE}")
        self.main_logger.info("******************* Input Summary *********************")
        self.main_logger.info(f"Number of sources: {len(self.pifile_lst)}")
        self.main_logger.info(f"Redshift range: {np.min(self.z_lst):.3f} -- {np.max(self.z_lst):.3f}")
        self.main_logger.info(f"NH range: {np.min(self.nh_lst)} -- {np.max(self.nh_lst)}")
        self.main_logger.info(f"NH file: {self.nh_file if self.nh_file is not None else 'None'}")
        self.main_logger.info(f"RSP weighting method: {self.rspwt_method}")
        self.main_logger.info(f"RSP projection gamma: {self.rspproj_gamma}")
        self.main_logger.info(f"Flux calculation range: {self.int_rng[0]} -- {self.int_rng[1]} keV (used only in `SHP` mode)")
        self.main_logger.info(f"ARF Truncation energy: {self.ene_trc} keV")
        self.main_logger.info(f"Source type: {'extended sources' if self.extended else 'point sources'}")
        self.main_logger.info(f"Number of CPUs used for shifting RMF: {self.nthreads}")
        self.main_logger.info(f"Number of background groups: {self.Nbkggrp}")
        if self.bootstrap:
            self.main_logger.info(f"Bootstrap: TRUE")
            self.main_logger.info(f"Number of realizations: {self.num_bootstrap}")
            self.main_logger.info(f"Fraction of sources to participate bootstrap: {self.bootstrap_portion}")
        else:
            self.main_logger.info(f"Bootstrap: FALSE")
        if self.do_cache:
            self.main_logger.info(f"Do_cache: You have chosen to save and load individual rest-frame files. These files are saved under individual source directories. Please ensure you have write permissions.")
        self.main_logger.info(f"Output directory: {self.outdir}")
        self.main_logger.info(f"Output PI spectrum (base)name: {self.o_pi_fname_rlz[0]}")
        self.main_logger.info(f"Output bkg PI spectrum (base)name: {self.o_bkgpi_fname_rlz[0]}")
        self.main_logger.info(f"Output ARF (base)name: {self.o_arf_fname_rlz[0]}")
        self.main_logger.info(f"Output RMF (base)name: {self.o_rmf_fname_rlz[0]}")
        self.main_logger.info(f"Output FENE (base)name: {self.o_fene_fname_rlz[0]}")
        self.main_logger.info("*******************************************************")
        
        #--- rest-frame shifting
        ##--- NOTE: we shift all sources regardless of bootstrap to save run time
        self.main_logger.info("")
        self.main_logger.info("******************* Shifting ... **********************")
        t0 = time.time()
        results = Parallel(
            n_jobs=self.nthreads,
            backend="loky",     ## use backend="loky" to avoid memory leakage
            # verbose=1,
            # return_as="generator",
            # pre_dispatch=self.nthreads,
        )(
            delayed(self._run_single_source)(i)
            for i in tqdm(range(self.Nsrc))
        )
        peak_rss_sft = get_ram_gb() # peak RAM usage for shifting
        self.main_logger.info(f"Peak RAM usage during shifting: {peak_rss_sft:.1f} GB")
        self.main_logger.info(f"Total time used for shifting: {time.time()-t0} s")
        
        #--- stacking
        self.main_logger.info("")
        self.main_logger.info("******************* Stacking ... **********************")
        peak_rss_stk = 0.0  # peak RAM usage for stacking
        t0_all = time.time()
        ##--- for each realization, we randomly draw sample from ** shifted pi & rsp **
        for k in range(self.num_bootstrap):
            t0 = time.time()
            self.main_logger.info("")
            self.main_logger.info(f"=== Realization {k:0{len(str(self.num_bootstrap))}d} ===")
            for i,(pi_sft,bkgpi_sft,bkgscal,rspmat_sft,rspwt,arffene,fene,expo,rega,msg) in enumerate(tqdm(results,total=self.Nsrc,desc="stacking")):
                # NOTE: bwt_src: how many times the i-th source appears in the k-th bootstrap realization (bootstrap weight)
                bwt_src = self.bwt_lst_rlz[k][i]
                ##--- stacking pi & rsp
                self.pi_stk_rlz[k] += pi_sft * bwt_src
                self.bkgpi_stk_rlz[k] += bkgpi_sft * bkgscal * bwt_src
                self.rspmat_stk_rlz[k] += rspmat_sft * rspwt * bwt_src
                ##--- saving meta-data for later renormalization
                self.bkgscal_lst_rlz[k].append(bkgscal)                         # (num_bootstrap, Nsrc)
                self.rspwt_lst_rlz[k].append(rspwt)                             # (same)
                self.expo_lst_rlz[k].append(expo)                               # (same)
                self.rega_lst_rlz[k].append(rega)                               # (same)
                self.arffene_lst_rlz[k].append(arffene)                         # (same)
                self.fene_lst_rlz[k].append(fene)                               # (same)
                self.pi_totcts_lst_rlz[k].append(np.sum(pi_sft))                # (same)
                self.bkgpi_totcts_lst_rlz[k].append(np.sum(bkgpi_sft*bkgscal))  # (same)
                self.bkgpi_sft_lst_rlz[k].append(bkgpi_sft)                     # (num_bootstrap, Nsrc, Nchan)
                ##--- print warning if exists
                if msg is not None:
                    self.main_logger.warning(msg)
                ##--- record ram usage
                rss_stk = get_ram_gb()
                peak_rss_stk = max(peak_rss_stk,rss_stk)
            self.bkgscal_lst_rlz[k] = np.array(self.bkgscal_lst_rlz[k])
            self.bkgpi_sft_lst_rlz[k] = np.array(self.bkgpi_sft_lst_rlz[k])
            self.rspwt_lst_rlz[k] = np.array(self.rspwt_lst_rlz[k])
            self.expo_lst_rlz[k] = np.array(self.expo_lst_rlz[k])
            self.rega_lst_rlz[k] = np.array(self.rega_lst_rlz[k])
            self.arffene_lst_rlz[k] = np.array(self.arffene_lst_rlz[k])
            self.fene_lst_rlz[k] = np.array(self.fene_lst_rlz[k])
            self.pi_totcts_lst_rlz[k] = np.array(self.pi_totcts_lst_rlz[k])
            self.bkgpi_totcts_lst_rlz[k] = np.array(self.bkgpi_totcts_lst_rlz[k])
            self.main_logger.info(f"Peak RAM usage during stacking (realization {k:0{len(str(self.num_bootstrap))}d}): {peak_rss_stk:.1f} GB")
            self.main_logger.info(f"Total time used for stacking (realization {k:0{len(str(self.num_bootstrap))}d}): {time.time()-t0} s")
        self.main_logger.info("----")
        self.main_logger.info(f"Total time used for all stacking: {time.time()-t0_all} s")

        #--- PI error calculation
        self.main_logger.info("")
        self.main_logger.info("************** PI error calculation ... ***************")
        t0 = time.time()
        for k in range(self.num_bootstrap):
            ##--- for src pi
            self.pi_stk_rlz[k],self.pierr_stk_rlz[k] = calc_pi_error(pi_stk=self.pi_stk_rlz[k]) # update pi_stk to integer, and calculate pierr_stk
            ##--- for bkg pi
            self.bkgpi_stk_rlz[k],self.bkgpierr_stk_rlz[k] = calc_bkgpi_error(
                bkgpi_lst=self.bkgpi_sft_lst_rlz[k],bkgscal_lst=self.bkgscal_lst_rlz[k],Nbkggrp=self.Nbkggrp,
            )
        self.main_logger.info(f"Total time used for PI error calculation: {time.time()-t0} s.")

        #--- response renormalization and RMF&ARF extraction
        self.main_logger.info("")
        self.main_logger.info("************** Extracting ARF & RMF ... ***************")
        t0 = time.time()
        for k in range(self.num_bootstrap):
            ##--- renormalization
            self.rspmat_stk_rlz[k],self.rspnorm_rlz[k],self.rspwt_lst_rlz[k],self.expo_stk_rlz[k],self.rega_stk_rlz[k] = rescale_rspmat(
                rspmat=self.rspmat_stk_rlz[k],rspwt_lst=self.rspwt_lst_rlz[k],
                expo_lst=self.expo_lst_rlz[k],rega_lst=self.rega_lst_rlz[k],
                rspwt_method=self.rspwt_method,extended=self.extended,
            )
            ##--- extract ARF & RMF from the stacked full response
            self.specresp_stk_rlz[k],self.prob_stk_rlz[k] = extract_arf_rmf_from_rspmat(self.rspmat_stk_rlz[k])
        self.main_logger.info(f"Total time used for ARF & RMF extraction: {time.time()-t0} s.")

        #--- finally, write fits files
        self.main_logger.info("")
        self.main_logger.info("****************** Saving files ... *******************")
        for k in range(self.num_bootstrap):
            write_pi(
                chan=self.CHANNEL,pi=self.pi_stk_rlz[k],pierr=self.pierr_stk_rlz[k],pi_fname=self.o_pi_fname_rlz[k],
                expo=self.expo_stk_rlz[k],rega=self.rega_stk_rlz[k],bkgpi_fname=self.o_bkgpi_fname_rlz[k],rmf_fname=self.o_rmf_fname_rlz[k],arf_fname=self.o_arf_fname_rlz[k],spec_type="STACKED",z=None,
            )
            write_bkgpi(
                chan=self.CHANNEL,bkgpi=self.bkgpi_stk_rlz[k],bkgpierr=self.bkgpierr_stk_rlz[k],bkgpi_fname=self.o_bkgpi_fname_rlz[k],
                expo=self.expo_stk_rlz[k],rega=self.rega_stk_rlz[k],spec_type="STACKED",z=None,
            )
            write_arf(
                arfene_lo=self.IENE_LO,arfene_hi=self.IENE_HI,specresp=self.specresp_stk_rlz[k],arf_fname=self.o_arf_fname_rlz[k],
                detchans=len(self.CHANNEL),expo=self.expo_stk_rlz[k],rega=self.rega_stk_rlz[k],rspwt_method=self.rspwt_method,rspnorm=self.rspnorm_rlz[k],
                srcid_lst=self.srcid_lst_rlz[k],rspwt_lst=self.rspwt_lst_rlz[k],pi_totcts_lst=self.pi_totcts_lst_rlz[k],bkgpi_totcts_lst=self.bkgpi_totcts_lst_rlz[k],flg=self.int_flg,spec_type="STACKED",z=None,
            )
            write_rmf(
                chan=self.CHANNEL,ene_lo=self.ENE_LO,ene_hi=self.ENE_HI,iene_lo=self.IENE_LO,iene_hi=self.IENE_HI,prob=self.prob_stk_rlz[k],
                rmf_fname=self.o_rmf_fname_rlz[k],expo=self.expo_stk_rlz[k],rega=self.rega_stk_rlz[k],rspwt_method=self.rspwt_method,
                srcid_lst=self.srcid_lst_rlz[k],rspwt_lst=self.rspwt_lst_rlz[k],arf_fname=self.o_arf_fname_rlz[k],spec_type="STACKED",z=None,
            )
            write_fene(
                srcid_lst=self.srcid_lst_rlz[k],arffene_lst=self.arffene_lst_rlz[k],fene_lst=self.fene_lst_rlz[k],
                fene_fname=self.o_fene_fname_rlz[k],
            )
            
        #--- generate summary log
        self.main_logger.info("")
        self.main_logger.info(f"#######################################################")
        self.main_logger.info(f"########## Stacking {self.Nsrc} spectra completed! ###########")
        self.main_logger.info(f"#######################################################")
        if self.bootstrap:
            self.main_logger.info(f"Stacked PI spectrum saved to: {self.o_pi_fname_rlz[0]} --- {self.o_pi_fname_rlz[-1]}")
            self.main_logger.info(f"Stacked BKGPI spectrum saved to: {self.o_bkgpi_fname_rlz[0]} --- {self.o_bkgpi_fname_rlz[-1]}")
            self.main_logger.info(f"Stacked ARF saved to: {self.o_arf_fname_rlz[0]} --- {self.o_arf_fname_rlz[-1]}")
            self.main_logger.info(f"Stacked RMF saved to: {self.o_rmf_fname_rlz[0]} --- {self.o_rmf_fname_rlz[-1]}")
            self.main_logger.info(f"Stacked FENE saved to: {self.o_fene_fname_rlz[0]} --- {self.o_fene_fname_rlz[-1]}")
        else:
            self.main_logger.info(f"Stacked PI spectrum saved to: {self.o_pi_fname_rlz[0]}")
            self.main_logger.info(f"Stacked BKGPI spectrum saved to: {self.o_bkgpi_fname_rlz[0]}")
            self.main_logger.info(f"Stacked ARF saved to: {self.o_arf_fname_rlz[0]}")
            self.main_logger.info(f"Stacked RMF saved to: {self.o_rmf_fname_rlz[0]}")
            self.main_logger.info(f"Stacked FENE saved to: {self.o_fene_fname_rlz[0]}")
        self.main_logger.info("")
        self.main_logger.info(f"# NOTE: the output stacked spectra have {{BACK,AREA,CORR}}SCAL=1, even though the inputs have different ratios. This is because these information have already gone into the background spectrum by scaling it.")
        self.main_logger.info("")
        self.main_logger.info("****** Response weighting factor for each source ******")
        self.main_logger.info(f"Your sources are {'extended sources' if self.extended else 'point sources'}.")
        if self.rspwt_method == "SHP":
            self.main_logger.info("`SHP` mode: assuming all sources have similar spectral shape, and weights calculated as COUNTS/ARF (normalized). This gives the most robust estimate of average spectral shape, but the y-axis of stacked spectrum would not carry physical meaning.")
        elif self.rspwt_method == "FLX":
            self.main_logger.info(f"`FLX` mode: assuming all sources have similar spectral shape + flux [{'erg/cm^2/s/deg^2' if self.extended else 'erg/cm^2/s'}], and weights calculated as {'EXPOSURE*REGAREA' if self.extended else 'EXPOSURE'}. Spectral shape may be biased if fluxes vary significantly among the sample. The y-axis of stacked spectrum gives the average flux.")
        elif self.rspwt_method == "LMN":
            self.main_logger.info(f"`LMN` mode: assuming all sources have similar spectral shape + luminosity [{'erg/s/deg^2' if self.extended else 'erg/s'}], and weights calculated as {'EXPOSURE*REGAREA/DISTANCE**2' if self.extended else 'EXPOSURE/DISTANCE**2'}. Spectral shape may be biased if luminosities vary significantly among the sample. The y-axis of stacked spectrum gives the average luminosity, divided by 1e60.")
        self.main_logger.info(f"Below is your response weighting factor list under `{self.rspwt_method}` mode (showing 1st realization only):")
        self.main_logger.info(self.rspwt_lst_rlz[0])
        if self.bootstrap:
            self.main_logger.info(f"Full list can be seen in the `WEIGHT` extension of the output ARF file: {self.o_arf_fname_rlz[0]} --- {self.o_arf_fname_rlz[-1]}")
        else:
            self.main_logger.info(f"Full list can be seen in the `WEIGHT` extension of the output ARF file: {self.o_arf_fname_rlz[0]}")
        self.main_logger.info("*******************************************************")

        print(f"Finished. Please check {self.logger_fname} for detailed log.")
        
        return self.pi_stk_rlz,self.pierr_stk_rlz,self.bkgpi_stk_rlz,self.bkgpierr_stk_rlz,self.specresp_stk_rlz,self.prob_stk_rlz
    

    def _run_single_source(self,i):
        """
        Shift single spectrum & response to rest-frame.
        This is an internal function for `run`.

        Paramters
        ---------
        i : int
            Source index.

        Returns
        -------
        pi_sft : numpy.ndarray
            Rest-frame PI spectrum.

        bkgpi_sft : numpy.ndarray
            Rest-frame bkg PI spectrum.

        bkgscal : float
            Background-to-source scaling ratio.

        rspmat_sft : numpy.ndarray
            Rest-frame full response matrix.

        rspwt : float
            Response scaling weight. Note, a further renormalization 
            will be needed after stacking.

        arffene : float
            First contributing energy from ARF.

        fene : float
            First contributing energy from rest-frame PI (i.e., which 
            energy starts to contribute at least 1 photon).

        expo : float
            Exposure.

        rega : float
            Region area.

        msg: str
            Error message.
        """
        pifile = self.pifile_lst[i]
        if self.bkgpifile_lst is not None:
            bkgpifile = self.bkgpifile_lst[i]
        else:
            bkgpifile = None
        arffile = self.arffile_lst[i]
        rmffile = self.rmffile_lst[i]
        z = self.z_lst[i]
        nh = self.nh_lst[i]
        expo = get_expo(pifile)
        rega = get_rega(pifile)

        #--- rest-frame filename (for do_cache)
        pifile_rf    = f"{pifile}.rf"
        bkgpifile_rf = f"{bkgpifile}.rf" if bkgpifile else None
        rspfile_rf   = f"{pifile}.rf.rsp"

        #--- shifting pi to rest-frame
        pi_chan,pi_sft,z_pi_cached,flg_pi = self._load_or_shift_pi(
            infile=pifile,
            outfile=pifile_rf,
            z=z,
            do_cache=self.do_cache,
            shift_pi_kwargs=dict(
                ene_lo=self.ENE_LO,ene_hi=self.ENE_HI,ene_ce=self.ENE_CE,ene_wd=self.ENE_WD,
                rmf_fname=self.sample_rmf,ene_trc=self.ene_trc,
            ),
            write_pi_kwargs=dict(
                expo=expo,rega=rega,
                bkgpi_fname=bkgpifile_rf,rmf_fname=rspfile_rf,arf_fname=None,
                spec_type="RESTFRAM",
            ),
        )

        #--- shifting bkgpi to rest-frame
        if self.bkgpifile_lst is not None and bkgpifile is not None:
            bkgpi_chan,bkgpi_sft,z_bkg_cached,flg_bkg = self._load_or_shift_pi(
                infile=bkgpifile,
                outfile=bkgpifile_rf,
                z=z,
                do_cache=self.do_cache,
                shift_pi_kwargs=dict(
                    ene_lo=self.ENE_LO,ene_hi=self.ENE_HI,ene_ce=self.ENE_CE,ene_wd=self.ENE_WD,
                    rmf_fname=self.sample_rmf,ene_trc=self.ene_trc,
                ),
                write_pi_kwargs=dict(
                    expo=expo,rega=rega,
                    bkgpi_fname=None,rmf_fname=rspfile_rf,arf_fname=None,
                    spec_type="RESTFRAM",
                ),
            )
            bkgscal = get_bkgscal(pifile,bkgpifile)
        else:
            bkgpi_sft = np.zeros_like(pi_sft,dtype=np.float32)
            bkgscal = 1.0
            z_bkg_cached,flg_bkg = None,False

        #--- shifting full response (arf*rmf) to rest-frame
        rspmat_sft,z_rsp_cached,flg_rsp = self._load_or_shift_rsp(
            outfile=rspfile_rf,
            z=z,
            do_cache=self.do_cache,
            shift_rsp_kwargs=dict(
                arf_fname=arffile,rmf_fname=rmffile,z=z,
                nh_file=self.nh_file,nh=nh,ene_trc=self.ene_trc,
            ),
            write_rsp_kwargs=dict(
                chan=pi_chan,
                ene_lo=self.ENE_LO,ene_hi=self.ENE_HI,
                iene_lo=self.IENE_LO,iene_hi=self.IENE_HI,
                expo=expo,rega=rega,
                rspwt_method=None,srcid_lst=None,rspwt_lst=None,arf_fname=None,
                spec_type="RESTFRAM",
            ),
        )

        #--- computing response weighting factor
        rsp1d_sft = project_rspmat(
            rspmat=rspmat_sft,ene_lo=self.ENE_LO,ene_hi=self.ENE_HI,arfene_lo=self.IENE_LO,arfene_hi=self.IENE_HI,
            proj_axis="CHANNEL",gamma=self.rspproj_gamma,
        )
        rspwt = compute_rspwt(
            specresp=rsp1d_sft,pi=pi_sft,z=z,bkgpi=bkgpi_sft,bkgscal=bkgscal,expo=expo,ene_wd=self.ENE_WD,flg=self.int_flg,
            rspwt_method=self.rspwt_method,extended=self.extended,rega=rega,
        )
        
        #--- looking for effective first energy for later visualization
        arf_sft = project_rspmat(
            rspmat=rspmat_sft,ene_lo=self.ENE_LO,ene_hi=self.ENE_HI,arfene_lo=self.IENE_LO,arfene_hi=self.IENE_HI,
            proj_axis="MODEL",
        )
        arf_nonzero_mask = (arf_sft!=0)
        arffene = self.IENE_CE[arf_nonzero_mask][0]
        pi_nonzero_mask = (pi_sft!=0)
        fene = self.ENE_CE[pi_nonzero_mask][0] if pi_nonzero_mask.any() else -1

        #--- generate warning message (if any)
        msg = None
        if self.do_cache and (flg_pi or flg_bkg or flg_rsp):
            ##--- compute fractional mismatch using available cached z values
            diffs = []
            if z_pi_cached is not None:
                diffs.append(abs(z_pi_cached - z))
            if z_bkg_cached is not None:
                diffs.append(abs(z_bkg_cached - z))
            if z_rsp_cached is not None:
                diffs.append(abs(z_rsp_cached - z))
            if diffs:
                max_err = max(diffs) / (z if z != 0 else 1.0)
                msg = f"{pifile_rf}: conflicting redshift ({max_err*100:.3f}%)"
            else:
                msg = f"{pifile_rf}: conflicting redshift (cache mismatch)"

        return pi_sft,bkgpi_sft,bkgscal,rspmat_sft,rspwt,arffene,fene,expo,rega,msg


    def _load_or_shift_pi(
        self,
        infile,outfile,z,do_cache,
        shift_pi_kwargs,write_pi_kwargs,
    ):
        """
        Internal function for `_run_single_source()`. Either load rest-frame PI from
        existing `.rf` file (when `do_cache` is activated), or re-do the rest-frame 
        shifting.

        Parameters
        ----------
        infile : str
            Input PI file name.

        outfile : str
            Rest-frame PI file name.

        z : float
            Redshift.

        do_cache: bool
            Whether or not to store and read the intermediate rest-frame files 
            for each individual source. If true, source-wise rest-frame files 
            will be saved under the same path as `pifile`. This will save a lot 
            of running time, at the cost of additional disk space (and burden on
            I/O).

        shift_pi_kwargs : dict
            Parameter dictionary for `shift_pi`.

        write_pi_kwargs : dict
            Parameter dictionary for `write_pi`.

        Returns
        -------
        chan : numpy.ndarray
            Rest-frame channel.

        pi_sft : numpy.ndarray
            Photon counts in each rest-frame channel.

        z_cached : float
            Redshift.

        mismatch_flag : bool
            True if there is a mismatch in redshift between the rest-frame file and input value. 
        """
        if do_cache and os.path.exists(outfile):
            chan,pi_sft,z_cached = read_pi(pi_fname=outfile)
            flg = not np.isclose(z_cached,z,rtol=1e-4,atol=0.0)
            return chan,pi_sft,z_cached,flg

        chan,pi_sft,_,_ = shift_pi(pi_fname=infile,z=z,**shift_pi_kwargs)
        if do_cache:
            write_pi(chan=chan,pi=pi_sft,pierr=None,pi_fname=outfile,z=z,**write_pi_kwargs)

        return chan,pi_sft,None,False


    def _load_or_shift_rsp(
        self,
        outfile,z,do_cache,
        shift_rsp_kwargs,write_rsp_kwargs,
    ):
        """
        Internal function for `_run_single_source()`. Either load rest-frame RSP from
        existing `.rf.rsp` file (when `do_cache` is activated), or re-do the rest-frame 
        shifting.

        Parameters
        ----------
        outfile : str
            Rest-frame rsp file name.

        z : float
            Redshift.

        do_cache: bool
            Whether or not to store and read the intermediate rest-frame files 
            for each individual source. If true, source-wise rest-frame files 
            will be saved under the same path as `pifile`. This will save a lot 
            of running time, at the cost of additional disk space (and burden on
            I/O). 

        shift_rsp_kwargs : dict
            Parameter dictionary for `shift_rsp`.

        write_rsp_kwargs : dict
            Parameter dictionary for `write_rsp`.

        Returns
        -------
        rspmat_sft : numpy.ndarray
		    Rest-frame 2D RSP matrix.
            
        z_cached : float
            Redshift.
            
        mismatch_flag : bool
            True if there is a mismatch in redshift between the rest-frame file and input value. 
        """
        if do_cache and os.path.exists(outfile):
            rspmat_sft, z_cached = read_rsp(rsp_fname=outfile)
            flg = not np.isclose(z_cached,z,rtol=1e-4,atol=0.0)
            return rspmat_sft,z_cached,flg

        rspmat_sft = shift_rsp(**shift_rsp_kwargs)
        if do_cache:
            write_rmf(rmf_fname=outfile,prob=rspmat_sft,z=z,**write_rsp_kwargs)

        return rspmat_sft,None,False
