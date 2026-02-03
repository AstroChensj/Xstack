#!/usr/bin/env python3
"""
==========================================
Module for shifting and stacking responses
==========================================
:Authors:   Shi-Jiang Chen (MPE, USTC)
			Johannes Buchner (MPE)
			Teng Liu (USTC)
:Email:     JohnnyCsj666@gmail.com


"""
import numpy as np
from astropy.io import fits
from numba import jit
from astropy.cosmology import Planck18
import astropy.units as u
import os
from Xstack.utils.logger import utc_now_iso
from Xstack.config import VERSION,LASTUPDATE,WEB


def read_rsp(rsp_fname):
	"""
	Read RMF/RSP file.

    Parameters
    ----------
    rsp_fname : str
        RMF or RSP file name.

		
    Returns
    -------
    prob : numpy.ndarray
		RMF 2D probability matrix, or RSP 2D matrix.

    z : float
        Redshift if exists.
	"""
	try:
		with fits.open(rsp_fname) as hdu:
			mat = hdu["MATRIX"].data
			ebo = hdu["EBOUNDS"].data
			head = hdu["MATRIX"].header
	except Exception:
		raise Exception(f"{rsp_fname} corrupted!")
	f_chan_0 = get_tlmin_from_header(rsp_fname)
	prob = get_prob(mat,ebo,f_chan_0)
	z = head.get("REDSHIFT",-999.0)

	return prob,z


def shift_rsp(
		arf_fname,rmf_fname,z,nh_file=None,nh=1e20,ene_trc=None,
		ene_lo=None,ene_hi=None,
):
	"""
	Rest-frame shifting the ARF&RMF. This is literally done by three steps: 
	1. Combine input ARF and RMF into a single RSP matrix (full response);
	2. Shift in the direction of output channel energy. That is to say, 
	   shift and broaden the probability profile for each input energy 
	   (i.e. when the detector receive a photon with some input energy, 
	   the probability that a signal at some output channel energy will 
	   be observed; so this is a function of output channel energy) by 
	   (1+z); 
	3. Shift in the direction of input energy by (1+z), with height 
	   (effective area) unchanged.
	
	Parameters
	----------
	arf_fname : str
		The ARF file name.

	rmf_fname : str
		The RMF file name.

	z : float
		Redshift.

	nh_file : str, optional
		Galactic absorption profile (absorption factor vs. energy). If 
		specified, galactic absorption correction will be applied on the 
		ARF before shifting.
		- Should be in txt format. 
		- Should also contain the following columns in the first 
		  extension: `nhene_ce`, `nhene_wd`, `factor`.
		- `factor` should indicate the absorption factor when nh=1e20.
		- An easy way to obtain the `nh_file`: iplot `tbabs*powerlaw` 
		  with `Nh`=1e20 and `PhoIndex`=0.0, `Norm`=1 in Xspec.

	nh : float, optional
		The galactic absorption nh of the source (e.g. 3e20). Defaults 
		to 1e20.

	ene_trc : float, optional
		Truncate energy below which manually set ARF and PI counts to 
		zero. For eROSITA, `ene_trc` is typically 0.2 keV. Defaults to 
		None.

		
	Returns
	-------
	rspmat_sft : numpy.ndarray
		Shifted 2D RSP matrix.
	"""
	#--- read ARF and RMF file
	with fits.open(arf_fname) as hdu:
		arf = hdu["SPECRESP"].data    # SPECRESP extension
	arfene_lo = arf["ENERG_LO"].astype(np.float32)  # because @jit method do not accept >f4
	arfene_hi = arf["ENERG_HI"].astype(np.float32)
	arfene_ce = (arfene_lo + arfene_hi) / 2
	arfene_wd = arfene_hi - arfene_lo
	specresp = arf["SPECRESP"]

	with fits.open(rmf_fname) as hdu:
		mat = hdu["MATRIX"].data
		ebo = hdu["EBOUNDS"].data
	ene_lo = ebo["E_MIN"].astype(np.float32)
	ene_hi = ebo["E_MAX"].astype(np.float32)
	iene_lo = mat["ENERG_LO"].astype(np.float32)
	iene_hi = mat["ENERG_HI"].astype(np.float32)
	# get f_chan_0 using TLMIN* keyword according to OGIP standards
	f_chan_0 = get_tlmin_from_header(rmf_fname)

	#--- sanity check: if the energy bins match
	assert np.all(arfene_lo==iene_lo), "arfene_lo (from arf_fname) and iene_lo (from rmf_fname) do not match!"
	assert np.all(arfene_hi==iene_hi), "arfene_hi (from arf_fname) and iene_hi (from rmf_fname) do not match!"

	#--- GalNH correction on ARF (optional)
	if nh_file is not None:
		with open(nh_file,"r") as file:
			lines = file.readlines()
		nhene_ce = []
		nhene_wd = []
		factor = []
		for line in lines:
			nhene_ce.append(float(line.split(" ")[0]))
			nhene_wd.append(float(line.split(" ")[1]))
			factor.append(float(line.split(" ")[2]))
		nhene_ce = np.array(nhene_ce)
		nhene_wd = np.array(nhene_wd)
		nhene_lo = nhene_ce - nhene_wd
		nhene_hi = nhene_ce + nhene_wd
		factor = np.array(factor)
		specresp = correct_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh)

	#--- truncate below ene_trc (optional)
	if ene_trc is not None:
		idx_trc = np.argmin(abs(arfene_ce-ene_trc))
		specresp[:idx_trc] = 0

	#--- combine ARF and RMF into a single RSP matrix
	prob = get_prob(mat,ebo,f_chan_0)       # the RMF 2D matrix, shape=(iene_ce, ene_ce)
	rspmat = prob*specresp[:,np.newaxis]    # the RSP matrix (RMF*ARF)

	#--- finally, shift the RSP matrix (currently we use only Non-parametric method, which is the most accurate one)
	rspmat_sft = shift_matrix(rspmat,arfene_lo,arfene_hi,ene_lo,ene_hi,z)

	del mat,ebo,prob    # to clear memory

	return rspmat_sft


@jit
def shift_matrix(prob,iene_lo,iene_hi,ene_lo,ene_hi,z):
	"""
	Numba code for Non-parametric RSP/RMF shifting.

	Parameters
	----------
	prob : numpy.ndarray
		The RMF 2D probability matrix, or the RSP 2D matrix.

	iene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin.

	iene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin.

	ene_lo : numpy.ndarray
		Lower edge of output channel energy bin.

	ene_hi : numpy.ndarray
		Upper edge of output channel energy bin.

	z : float
		Redshift.

		
	Returns
	-------
	prob_sft : numpy.ndarray
		The rest-frame shifted RSP/RMF 2D matrix. 
	"""
	iene_ce = (iene_lo + iene_hi) / 2
	iene_wd = iene_hi - iene_lo
	iene_id = np.arange(len(iene_ce))

	ene_ce = (ene_lo + ene_hi) / 2
	ene_wd = ene_hi - ene_lo
	ene_id = np.arange(len(ene_ce))
	
	# de-redshift probability matrix
	# step 1: horizontal shift, output channel energy *(1+z), dispersion automatically *(1+z)
	prob_sft_horizontal = np.zeros(prob.shape)  # the probability matrix after step 1: horizontal shift
	iene_ubound = np.max(iene_lo)
	iene_lbound = np.min(iene_hi)
	for i in range(len(iene_ce)):

		iene_lo_map = iene_lo[i] * (1+z)
		iene_hi_map = iene_hi[i] * (1+z)
		
		if iene_lo_map > iene_ubound:
			break
		if iene_hi_map < iene_lbound:
			continue

		prob_1d = np.zeros(len(ene_ce))
		ene_ubound = np.max(ene_lo)
		ene_lbound = np.min(ene_hi)
		for j in range(len(ene_ce)):
			ene_lo_map = ene_lo[j] * (1+z)
			ene_hi_map = ene_hi[j] * (1+z)
			
			if ene_lo_map > ene_ubound:
				break
			if ene_hi_map < ene_lbound:
				continue
			
			mask = (ene_lo_map < ene_hi) & (ene_hi_map > ene_lo)
			ene_id_mask = ene_id[mask]
			ene_wd_mask = ene_wd[mask]
			ene_lo_mask = ene_lo[mask]
			ene_hi_mask = ene_hi[mask]
			
			ene_wd_mask[0] = ene_hi_mask[0] - ene_lo_map
			ene_wd_mask[-1] = ene_hi_map - ene_lo_mask[-1]
			
			prob_mask = ene_wd_mask / np.sum(ene_wd_mask)
			
			prob_1d[ene_id_mask] += prob[i][j] * prob_mask
		
		if np.sum(prob_1d) > 0: # to deal with the high energy tail; we want to make sure that the sum along horizontal axis equals to arf specresp in the energy
			prob_1d *= np.sum(prob[i])/np.sum(prob_1d)
		prob_sft_horizontal[i] = prob_1d
			
	# step 2: vertical shift, input model energy *(1+z), height unchanged
	prob_sft_vertical = np.zeros(prob.shape)

	iene_sft_lo = iene_lo * (1+z)
	iene_sft_hi = iene_hi * (1+z)
	iene_sft_ce = iene_ce * (1+z)
	iene_sft_wd = iene_wd * (1+z)

	for i in range(prob_sft_vertical.shape[0]):
		mask = (iene_lo[i] <= iene_sft_hi) & (iene_hi[i] >= iene_sft_lo)
		if np.all(mask==False):
			continue
		iene_mask_lo = iene_sft_lo[mask].copy()
		iene_mask_hi = iene_sft_hi[mask].copy()
		iene_mask_ce = iene_sft_ce[mask].copy()
		iene_mask_wd = iene_sft_wd[mask].copy()
		prob_sft_horizontal_mask = prob_sft_horizontal[mask].copy()
		
		# for the first and last channel in the basket, we need to recalculate their widths
		iene_mask_wd[0] = iene_mask_hi[0] - iene_lo[i]
		iene_mask_wd[-1] = iene_hi[i] - iene_mask_lo[-1]
		
		prob_mask = iene_mask_wd / iene_mask_wd.sum()
		prob_sft_vertical[i] = np.sum(prob_sft_horizontal_mask*prob_mask[:,np.newaxis],axis=0)

	return prob_sft_vertical


def compute_rspwt(
		specresp,pi,z,bkgpi,bkgscal,expo,ene_wd,flg,rspwt_method,
		extended=False,rega=1,
	):
	"""
	Get the weighting factor for a single RSP.

	Parameters
	----------
	specresp : numpy.ndarray
		RSP specresp projected on channel energy axis (cm^2 vs. channel 
		energy). This is **not** simply the ARF curve.

	pi : numpy.ndarray
		PI spectrum.

	z : float
		Redshift.

	bkgpi : numpy.ndarray
		Background PI spectrum.

	bkgscal : float
		Background scaling-ratio.

	expo : float
		Exposure.

	ene_wd : numpy.ndarray
		Output channel energy bin width.

	flg : numpy.ndarray
		Output channel energy flag.

	method : str
		Method for calculating ARFSCAL. Available methods are:
		- `SHP`: assuming all sources have same spectral shape
		- `FLX`: assuming all sources have same spectral shape and flux
			+ For point sources (`extended`==False), flux is in units of 
			  [erg/cm^2/s].
			+ For extended sources (`extended`==True), flux is in units 
			  of [erg/cm^2/s/deg^2].
		- `LMN`: assuming all sources have same luminosity
			+ For point sources (`extended`==False), luminosity is in 
			  units of [erg/s].
			+ For extended sources (`extended`==True), luminosity is in 
			  units of [erg/s/deg^2].

	extended : bool, optional
		Whether or not the source is extended. Defaults to False, i.e., 
		a point source.

	rega : int or float, optional
		`REGAREA` list. Used when `extended`==True.

		
	Returns
	-------
	rspwt : numpy.ndarray
		The RSP weight for each source.

	rspnorm : float
		The RSP weight normalization. This is only useful for `SHP` mode.

	expo_stacked : float
		The final EXPOSURE to be written in the header of stacked PI
		and RSP.

	rega_stacked : float
		The final REGAREA to be written in the header of stacked PI
		and RSP.

	Notes
	-----
	The ideal choice of `method` should be `SHP`, which starts from the 
	minimum assumption and thus gives the most unbiased results on 
	spectral shape. A caveat of `SHP` is that the individual spectrum 
	should have sufficient photon counts (>~10), and the resulting 
	stacked spectrum does not carry a physical flux unit.

	The second option of `method`, in case the individual photon counts
	is too low, should be `FLX`. In addition to the minimum assumption 
	used by `SHP`, it assumes that all sources have similar flux (
	[erg/cm^2/s] for point source or [erg/cm^2/s/deg^2] for extended).
	This should be reasonable for a flux-limited survey, where most 
	sources lie around the detection flux limit, and should thus have 
	similar flux.

	`LMN` is similar to `FLX`, except it assumes all sources to be 
	summed have similar luminosity. # Luminosity can be calcualted as
	e.g., flux 0.5 2 --> luminosity in 0.5-2 keV band / 1e60 

	"""

	if rspwt_method == "SHP":   # SHAPE
		# This is the minimum assumption for spectral stacking
		# that all spectra look similar in shape
		# thus should be most widely applicable
		# A trade-off is that the stacked spectrum does not carry
		# a physical flux unit; only spectral shape info is preserved
		net_pi = pi - bkgpi*bkgscal
		net_pi = net_pi[flg]
		sum_net_pi = np.sum(net_pi)

		resp_ene = specresp * ene_wd
		resp_ene = resp_ene[flg]
		sum_resp_ene = np.sum(resp_ene)

		rspwt = sum_net_pi / sum_resp_ene

	elif rspwt_method == "FLX":   # FLUX
		# For extended sources, flux in units of [erg/cm^2/s/deg^2]
		if extended:
			# We take the solid-angle-weighted averaged exposure 
			# as the stacked EXPOSURE, and 1 deg^2 as the stacked
			# REGAREA, following X. Zhang+2024
			# NOTE: additional (1+z) for the same reason as PS
			rspwt = expo * rega * (1+z)
		# For point sources, flux in units of [erg/cm^2/s]
		else:
			# We take the summed exposure as the stacked EXPOSURE
			# NOTE: we multiply (1+z), so that the "stacked rest-frame flux"
			# is simply "stacked rest-frame luminosity" / (4*pi*d_L^2)
			# where d_L is the average luminosity distance for the sample
			rspwt = expo * (1+z)

	elif rspwt_method == "LMN":   # LUMINOSITY
		# luminosity distances in units of [Mpc]
		dist = Planck18.luminosity_distance(z).to(u.cm).value
		if extended:
			# the averaged exposure following X. Zhang+2024
			rspwt = expo * rega / (4*np.pi*dist**2/(1+z))
		else:
			rspwt = expo / (4*np.pi*dist**2/(1+z))

	else:
		raise Exception("Available method for ARF scaling ratio calculation: `FLX`, `LMN`, or `SHP` !")
	
	return rspwt


def rescale_rspmat(rspmat,rspwt_lst,expo_lst,rega_lst,rspwt_method,extended=False):
	"""
	Rescale full response matrix (RSP) for different methods.

	Parameters
	----------
	rspmat : numpy.ndarray
		Stacked RSP 2D probability matrix.

	rspwt_lst : numpy.ndarray
		Response weighting factor for each source (to be rescaled).

	expo_lst : numpy.ndarray
		Exposure for each source.

	rega_lst : numpy.ndarray
		Region area parameter for each source. 
		TODO: applicable only to eROSITA ... update for other inst?

	rspwt_method : str
		Response weighting method.

	extended : bool, optional
		Extended or not. Defaults to False

		
	Returns
	-------
	rspmat : numpy.ndarray
		Rescaled RSP matrix.
	
	rspnorm : float
		To prevent overflow of very large number in the case of `LMN` 
		mode, the rescaled RSP matrix has been multiplied by a very small 
		number. Multiply your `rspmat` by `rspnorm` to bring it back to 
		the appropriate number. 

	rspwt_lst : numpy.ndarray
		List of response weighting factors.

	expo_stk : float
		Stacked exposure.

	rega_stk : float
		Stacked region area.
	"""
	if rspwt_method == "SHP":
		# renormalize so that sum of rspwt_lst is 1
		rspnorm = 1 / np.sum(rspwt_lst)
		rspmat *= rspnorm
		rspwt_lst *= rspnorm	# the updated rspwt_lst, for check only
		# EXPOSURE and REGAREA is meaningless in SHP mode
		expo_stk = np.sum(expo_lst)
		rega_stk = 1.0

	elif rspwt_method == "FLX":
		if extended:
			# we take the solid-angle-weighted averaged exposure 
			# as the stacked EXPOSURE, and 1 deg^2 as the stacked
			# REGAREA, following X. Zhang+2024
			expo_stk = np.sum(expo_lst * rega_lst) / np.sum(rega_lst)
			# REGAREA renormalized to 1
			rega_stk = 1.0
			rscal_factor = 1 / (expo_stk * rega_stk)
			# No normalization is performed, and thus chosen arbitrarily
			rspnorm = 1.0
		else:
			# we take the summed exposure as the stacked EXPOSURE
			expo_stk = np.sum(expo_lst)
			# REGAREA not involved at all, and thus chosen arbitrarily
			rega_stk = 1.0
			# rescale factor
			rscal_factor = 1 / expo_stk
			# additional renormalization factor
			rspnorm = 1.0
		rspmat *= rscal_factor * rspnorm
		rspwt_lst *= rscal_factor * rspnorm

	elif rspwt_method == "LMN":
		if extended:
			# the averaged exposure following X. Zhang+2024
			expo_stk = np.sum(expo_lst * rega_lst) / np.sum(rega_lst)
			# REGAREA renormalized to 1
			rega_stk = 1.0
			# rescale factor
			rscal_factor = 1 / (expo_stk * rega_stk)
			# set rspnorm to a large very large number as rspmat is typically very small
			rspnorm = 1e60
		else:
			expo_stk = np.sum(expo_lst)
			# REGAREA not involved at all, and thus chosen arbitrarily
			rega_stk = 1.0
			# rescale factor
			rscal_factor = 1 / expo_stk
			# set rspnorm to a large very large number as rspmat is typically very small
			rspnorm = 1e60
		rspmat *= rscal_factor * rspnorm
		rspwt_lst *= rscal_factor * rspnorm

	else:
		raise Exception("Available method for ARF scaling ratio calculation: `FLX`, `LMN`, or `SHP` !")
			
	return rspmat,rspnorm,rspwt_lst,expo_stk,rega_stk


def correct_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh):
	"""
	Multiply the ARF specresp with the galactic absorption profile. 
	The template galactic absorption profile should be at nh=1e20.
	The source nh value is specified by `nh`.

	Parameters
	----------
	specresp : numpy.ndarray
		The ARF specresp to be corrected.

	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin.

	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin. Defaults to 
		None.

	factor : numpy.ndarray
		Template galactic absorption profile at nh=1e20.

	nhene_lo : numpy.ndarray
		Lower edge of nh model energy bin.

	nhene_hi : numpy.ndarray
		Upper edge of nh model energy bin.

	nh : float
		Galactic nh of the source.

		
	Returns
	-------
	specresp_cor : numpy.ndarray
		The corrected ARF specresp.
	"""
	nhene_ce = (nhene_lo + nhene_hi) / 2
	nhene_wd = nhene_hi - nhene_lo
	nh_scal = nh / 1e20
	factor_scal = factor ** nh_scal
	specresp_cor = specresp.copy()
	# for each arf energy bin, find the nearest nh energy bins, and assign the correction factor
	# if more than one nh bins can be found, do interpolation
	for i in range(len(specresp_cor)):
		mask = (nhene_hi >= arfene_lo[i]) & (nhene_lo <= arfene_hi[i])
		if np.all(mask==False):
			continue
		nhene_mask_lo = nhene_lo[mask].copy()
		nhene_mask_hi = nhene_hi[mask].copy()
		nhene_mask_ce = nhene_ce[mask].copy()
		nhene_mask_wd = nhene_wd[mask].copy()
		factor_scal_mask = factor_scal[mask].copy()

		# for the first and last channel in the basket, we need to recalculate their widths
		nhene_mask_wd[0] = nhene_mask_hi[0] - arfene_lo[i]
		nhene_mask_wd[-1] = arfene_hi[i] - nhene_mask_lo[-1]
		
		prob_mask = nhene_mask_wd / nhene_mask_wd.sum()
		specresp_cor[i] *= (factor_scal_mask * prob_mask).sum()

	return specresp_cor


def project_rspmat(rspmat,ene_lo,ene_hi,arfene_lo,arfene_hi,proj_axis="CHANNEL",gamma=2.):
	"""
	Project the 2D RSP matrix onto CHANNEL/MODEL energy axis, to get the 
	effective specresp (cm^2 vs. energy)
	
	Parameters
	----------
	rspmat : numpy.ndarray
		The 2D RSP matrix.

	ene_lo : numpy.ndarray
		Lower edge of output channel energy bin.

	ene_hi : numpy.ndarray
		Upper edge of output channel energy bin.

	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin.

	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin.

	proj_axis : str, optional
		The projection axis. Available options are:
		- `CHANNEL`: project on output channel energy axis. Note that to 
		   do this projection, we would nevertheless need to assume a 
		   spectral slope, or photo index (specified in `gamma`). This is 
		   to match the convention of unfolded spectrum (in e.g., XSPEC), 
		   where the effective area anchored on channel energy axis is in 
		   fact (folded model)/(model).
		- `MODEL`: project on input model energy axis
		Defaults to `CHANNEL`.

	gamma : float, optional
		The spectral slope. Defaults to 2.0. This is only used when 
		`proj_axis` is `CHANNEL`. For AGN sources, a powerlaw with 
		photon index of 2.0 is a good approximation.

		
	Returns
	-------
	rsp1d : numpy.ndarray
		The 1D effective area profile.
	"""
	# sanity check
	assert ene_lo.shape == ene_hi.shape, ""
	assert arfene_lo.shape == arfene_hi.shape, ""
	assert rspmat.shape[0] == len(arfene_lo), ""
	assert rspmat.shape[1] == len(ene_lo), ""

	arfene_ce = (arfene_lo + arfene_hi) / 2
	arfene_wd = arfene_hi - arfene_lo
	ene_ce = (ene_lo + ene_hi) / 2
	ene_wd = ene_hi - ene_lo

	if proj_axis == "CHANNEL":
		# To project the RSP matrix onto the output channel energy axis, we would nevertheless need to assume a spectral slope
		# for AGN sources, a powerlaw with photon index of 2.0 is a good approximation
		F_model = 1*arfene_ce**(-gamma) # the model spectrum (from our prior knowledge) as a function of model energy
		F_channel = 1*ene_ce**(-gamma)  # the same model spectrum, but as a function of output channel energy
		
		F_folded = np.sum(rspmat*arfene_wd[:,np.newaxis]*F_model[:,np.newaxis],axis=0)/ene_wd   # the folded model
		rsp1d = F_folded/F_channel  # effective area as a function of output channel energy = (folded model)/(model)

	elif proj_axis == "MODEL":
		rsp1d = np.sum(rspmat,axis=1)

	else:
		raise Exception("Invalid `proj_axis` parameter (available: `CHANNEL` or `MODEL`)!")

	return rsp1d



def get_prob(mat,ebo,f_chan_0=None):
	"""
	Parse the RMF file (input the `MATRIX` and `EBOUNDS` extension) into 
	a 2D probability matrix. 

	Parameters
	----------
	mat : astropy.io.fits.FITS_rec
		The `MATRIX` extension of a standard OGIP RMF file. Must include 
		the following columns:
		- `ENERG_LO`
		- `ENERG_HI`
		- `N_GRP`
		- `F_CHAN`
		- `N_CHAN`
		- `MATRIX`

	ebo : astropy.io.fits.FITS_rec
		The `EBOUNDS` extension of a standard OGIP RMF file. Must include 
		the following columns:
		- `E_MIN` 
		- `E_MAX`

	f_chan_0 : int, optional
		First channel index. Defaults to None. 
		If not specified, will be determined from rmf file.

		
	Returns
	-------
	prob : numpy.ndarray
		The RMF 2D probability matrix. Index [i,j], where:
		- i represents arfene (iene or inpu model energy)
		- j represents ene (output channel energy)
	"""
	ene_lo = ebo["E_MIN"].astype(np.float32)
	ene_hi = ebo["E_MAX"].astype(np.float32)
	ene_ce = (ene_lo + ene_hi) / 2
	ene_wd = ene_hi - ene_lo
	iene_lo = mat["ENERG_LO"].astype(np.float32)
	iene_hi = mat["ENERG_HI"].astype(np.float32)
	iene_ce = (iene_lo + iene_hi) / 2
	iene_wd = iene_hi - iene_lo
	grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
	prob = np.zeros(grid[0].shape) # probability per channel
	
	n_grp = mat["N_GRP"]
	f_chan = mat["F_CHAN"]
	n_chan = mat["N_CHAN"]
	matrix = np.array(mat["MATRIX"])
	
	# sanity check on f_chan_0
	if f_chan_0 not in [0,1]:
		f_chan_0 = int(np.min([np.min(f_chan[_]) if len(f_chan[_])>0 else 0 for _ in range(len(f_chan))])) # the zero point of channel index

	for i in range(len(iene_ce)):
		if isinstance(f_chan[i],(int,np.int16,np.int32)):	# sjchen@20251106: deal with XMM MOS format
			prob[i][f_chan[i]:f_chan[i]+n_chan[i]] += matrix[i]
		else:
			f_matrix = 0   # starting index of matrix[i]
			for grp_j in range(n_grp[i]):
				f_chan_j = f_chan[i][grp_j] - f_chan_0  # starting index of channel
				n_chan_j = n_chan[i][grp_j]             # number of channel
				e_chan_j = f_chan_j + n_chan_j          # ending index of in channel
				e_matrix = f_matrix + n_chan_j          # ending index of matrix[i]
				
				prob[i][f_chan_j:e_chan_j] += matrix[i][f_matrix:e_matrix]
				f_matrix += n_chan_j

	return prob


def get_prob1d(n_grp,f_chan,n_chan,matrix1d,Nene,f_chan_0=None):
	"""
	Get the 1d probability distribution for output channel energy at a 
	specific input model energy.

	Parameters
	----------
	n_grp : int
		`N_GRP` array of your specific input model energy, from `MATRIX` 
		extension.

	f_chan : int
		`F_CHAN` array of your specific input model energy, from `MATRIX` 
		extension.

	n_chan : int
		`N_CHAN` array of your specific input model energy, from `MATRIX` 
		extension.

	matrix1d : numpy.ndarray
		`MATRIX` array of your specific input model energy, from `MATRIX` 
		extension.

	Nene : int
		Length of output channel energy.

	f_chan_0 : int, optional
		The index number of the first output channel energy (0 or 1). 
		Defaults to 0.

		
	Returns
	-------
	prob1d : numpy.ndarray
		The 1d probability distribution for output channel energy at a 
		specific input model energy.
	"""
	f_matrix = 0   # starting index of matrix1d
	# sanity check on f_chan_0
	if f_chan_0 not in [0,1]:
		f_chan_0 = int(np.min([np.min(f_chan[_]) if len(f_chan[_])>0 else 0 for _ in range(len(f_chan))]))
	prob1d = np.zeros(Nene)
	if isinstance(f_chan,(int,np.int16,np.int32)):	# sjchen@20251106: deal with XMM MOS format
		prob1d[f_chan:f_chan+n_chan] += matrix1d
	else:
		for j in range(n_grp):
			f_chan_j = f_chan[j] - f_chan_0         # starting index of channel
			n_chan_j = n_chan[j]                    # number of channel
			e_chan_j = f_chan_j + n_chan_j          # ending index of in channel
			e_matrix = f_matrix + n_chan_j          # ending index of matrix[i]
			prob1d[f_chan_j:e_chan_j] += matrix1d[f_matrix:e_matrix]
			f_matrix += n_chan_j

	return prob1d


def extract_arf_rmf_from_rspmat(rspmat):
	"""
	Extract ARF and RMF from the full response RSP.

	Parameters
	----------
	rspmat : numpy.ndarray
		Full response 2D matrix.

	
	Returns
	-------
	specresp : numpy.ndarray
		ARF effective area as a function of input energies (`iene`).

	prob : numpy.ndarray
		2D probability RMF matrix as a function of input (`iene`) and 
		output (`ene`) energies.
	"""
	#--- ARF
	specresp = np.sum(rspmat,axis=1)

	#--- RMF
	with np.errstate(invalid="ignore"): # wrap up division warning
		prob = rspmat / specresp[:,np.newaxis]
	prob[np.isclose(prob,0,rtol=1e-06, atol=1e-06, equal_nan=False)] = 0 # remove elements with probability below the 1e-6 threshold
	prob[np.isnan(prob)] = 0 # remove NaN
	prob[prob<0] = 0 # remove negative elements
	with np.errstate(invalid="ignore"): # wrap up division warning
		prob /= np.sum(prob,axis=1)[:,np.newaxis] # renormalize
	prob[np.isnan(prob)] = 0 # remove NaN (produced when 0/0)
	# for the first few input energies, the probability may be empty
	# assign the first channel with 1 (an arbitrary choice)
	for i in range(len(prob)):
		if np.max(prob[i]) == 0.:
			prob[i][0] = 1

	return specresp,prob


def write_arf(
		arfene_lo,arfene_hi,specresp,arf_fname="stacked_arf.fits",
		detchans=1000,expo=10,rega=1,rspwt_method="SHP",rspnorm=1,
		srcid_lst=None,rspwt_lst=None,pi_totcts_lst=None,bkgpi_totcts_lst=None,flg=None,
		spec_type="STACKED",z=None,
):
	"""
	Write ARF file according to OGIP standards.
	Assume all spectral files (PI, ARF, RMF) under the same path for xspec convenience.

	Parameters
	----------
	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin, aka `iene_lo`.

	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin, aka `iene_hi`.

	specresp : numpy.ndarray
		Effective area defined within `arfene_lo` and `arfene_hi`.

	arf_fname : str, optional
		Output ARF name. Defaults to "stacked_arf.fits".

	detchans : int, optional
		Number of detected channels. This should be the length of PI spectral 
		channels, or equivalently the length of `ene`. Defaults to 1000.

	expo : int or float, optional
		Exposure in units of s. Defaults to 10.

	rega : int or float, optional
		Region area in units of deg^2. Defaults to 1.

	rspwt_method : str, optional
		Response weighting method. Defaults to "SHP".

	rspnorm : int or float, optional
		To prevent overflow of very large number in the case of `LMN` 
		mode, the rescaled RSP matrix has been multiplied by a very small 
		number. Multiply your `rspmat` by `rspnorm` to bring it back to 
		the appropriate number. Defaults to 1.

	srcid_lst : numpy.ndarray, optional
		Source id list. Defaults to None.

	rspwt_lst : numpy.ndarray, optional
		Response weighting factor list. Defaults to None.

	pi_totcts_lst : numpy.ndarray, optional
		PI spectrum total counts. Defaults to None.

	bkgpi_totcts_lst : numpy.ndarray, optional
		BKG PI spectrum total counts. Defaults to None.

	flg : numpy.ndarray, optional
		Which channels used in `SHP` mode in calculating response weighting
		factors. Defaults to None.

	spec_type : str, optional
		"STACKED" if this is the stacked ARF. "RESTFRAM" if this is single 
		source rest-frame ARF. Defaults to "STACKED".

	z : float, optional
		Source redshift if this is single source rest-frame ARF.

	Returns
	-------
	None
	"""
	hdu_lst = fits.HDUList()

	#--- extension 0: primary hdu
	primary_hdu = fits.PrimaryHDU()
	hdu_lst.append(primary_hdu)
	
	#--- extension 1: SPECRESP
	cols = [
		fits.Column(name="ENERG_LO",format="D",array=arfene_lo),
		fits.Column(name="ENERG_HI",format="D",array=arfene_hi),
		fits.Column(name="SPECRESP",format="D",array=specresp),
	]
	hdu_specresp = fits.BinTableHDU.from_columns(cols, name="SPECRESP")
	# ARF header following OGIP standards 
	# https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis"
	hdu_specresp.header["TELESCOP"] = spec_type
	hdu_specresp.header["INSTRUME"] = spec_type
	if z is not None:
		hdu_specresp.header["REDSHIFT"] = z
	hdu_specresp.header["CHANTYPE"] = "PI"
	hdu_specresp.header["DETCHANS"] = detchans
	hdu_specresp.header["HDUCLASS"] = "OGIP"
	hdu_specresp.header["HDUCLAS1"] = "RESPONSE"
	hdu_specresp.header["HDUCLAS2"] = "SPECRESP"
	hdu_specresp.header["HDUVERS"] = "1.1.0"
	hdu_specresp.header["EXPOSURE"] = (expo, "stacked exposure time [s]")
	hdu_specresp.header["REGAREA"] = (rega, "stacked region area [deg^2]")
	hdu_specresp.header["WTMETH"] = (rspwt_method, "response weighting method [SHP/FLX/LMN]")
	hdu_specresp.header["CREATOR"] = "XSTACK"
	hdu_specresp.header["HISTORY"] = f"{utc_now_iso()}: stacked source ARF created by Xstack v{VERSION} [{LASTUPDATE}] [{WEB}]"
	hdu_lst.append(hdu_specresp)

	#--- extension 2: WEIGHT
	cols = []
	if srcid_lst is not None:
		cols.append(fits.Column(name="SRCID",format="J",array=srcid_lst))
	if rspwt_lst is not None:
		cols.append(fits.Column(name="RSPWT",format="D",array=rspwt_lst))
	if pi_totcts_lst is not None:
		cols.append(fits.Column(name="PHOCOUN",format="J",array=pi_totcts_lst))
	if bkgpi_totcts_lst is not None:
		cols.append(fits.Column(name="BPHOCOUN",format="D",array=bkgpi_totcts_lst))
	if len(cols) > 0:
		hdu_weight = fits.BinTableHDU.from_columns(cols, name="WEIGHT")
		hdu_weight.header["RSPNORM"] = (rspnorm, "response normalizing factor")
		hdu_lst.append(hdu_weight)

	#--- extension 3: FLAG
	if flg is not None:
		cols = [
			fits.Column(name="CHANNEL",format="J",array=np.arange(1,len(flg)+1)),
			fits.Column(name="FLAG",format="J",array=flg.astype("int"))
		]
		hdu_flag = fits.BinTableHDU.from_columns(cols,name="FLAG")
		hdu_flag.header["FLAG"] = "whether the bin is used for RSPWT estimation"
		hdu_lst.append(hdu_flag)
	
	hdu_lst.writeto(f"{arf_fname}", overwrite=True)

	return


def write_rmf(
		chan,ene_lo,ene_hi,iene_lo,iene_hi,prob,rmf_fname="./stacked_rmf.fits",
		expo=10,rega=1,rspwt_method="SHP",
		srcid_lst=None,rspwt_lst=None,arf_fname="./stacked_arf.fits",
		spec_type="STACKED",z=None,
):
	"""
	Write RMF file according to OGIP standards.
	Assume all spectral files (PI, ARF, RMF) under the same path for xspec convenience.

	Parameters
	----------
	chan : numpy.ndarray
		PI Channel. Must be the same length as `ene`.

	ene_lo : numpy.ndarray
		Lower edge of output channel energy (PI energy) bin.
	
	ene_hi : numpy.ndarray
		Upper edge of output channel energy (PI energy) bin.

	prob : numpy.ndarray
		2D RMF matrix.

	rmf_fname : str, optional
		Output RMF name. Defaults to "stacked_rmf.fits".

	expo : int or float, optional
		Exposure in units of s. Defaults to 10.

	rega : int or float, optional
		Region area in units of deg^2. Defaults to 1.

	rspwt_method : str, optional
		Response weighting method. Defaults to "SHP".

	srcid_lst : numpy.ndarray, optional
		Source id list. Defaults to None.

	rspwt_lst : numpy.ndarray, optional
		Response weighting factor list. Defaults to None.

	arf_fname : str, optional
		Associated ARF name. Defaults to "stacked_arf.fits".

	spec_type : str, optional
		"STACKED" if this is the stacked ARF. "RESTFRAM" if this is single 
		source rest-frame ARF. Defaults to "STACKED".

	z : float, optional
		Source redshift if this is single source rest-frame ARF.

	Returns
	-------
	None
	"""
	hdu_lst = fits.HDUList()
		
	#--- extension 0: primary hdu
	primary_hdu = fits.PrimaryHDU()
	hdu_lst.append(primary_hdu)
	
	#--- extension 1: MATRIX
	n_grp = []
	f_chan = []
	n_chan = []
	mat = []
	for i in range(len(iene_lo)):
		n_grp.append(1)
		f_chan.append(np.array([1]))
		prob_i = prob[i]
		# Find the index of the first non-zero element from the end
		last_nonzero_idx = len(prob_i) - np.argmax(prob_i[::-1] != 0) - 1
		n_chan.append(np.array([last_nonzero_idx+1]))
		mat.append(prob_i[:last_nonzero_idx+1])
	n_grp = np.array(n_grp)
		
	cols = [
		fits.Column(name="ENERG_LO",format="D",array=iene_lo),
		fits.Column(name="ENERG_HI",format="D",array=iene_hi),
		fits.Column(name="N_GRP",format="J", array=n_grp),
		fits.Column(name="F_CHAN",format="PJ()",array=f_chan),
		fits.Column(name="N_CHAN",format="PJ()",array=n_chan),
		fits.Column(name="MATRIX",format="PD()",array=mat),
	]
	hdu_matrix = fits.BinTableHDU.from_columns(cols, name="MATRIX")
	# RMF header following OGIP standards 
	# https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis"
	hdu_matrix.header["TELESCOP"] = spec_type
	hdu_matrix.header["INSTRUME"] = spec_type
	hdu_matrix.header["CHANTYPE"] = "PI"
	hdu_matrix.header["DETCHANS"] = prob.shape[1]
	hdu_matrix.header["HDUCLASS"] = "OGIP"
	hdu_matrix.header["HDUCLAS1"] = "RESPONSE"
	hdu_matrix.header["HDUCLAS2"] = "RSP_MATRIX"
	hdu_matrix.header["HDUVERS"] = "1.3.0"
	hdu_matrix.header["TLMIN4"] = 1 # the first channel in the response
	hdu_matrix.header["EXPOSURE"] = (expo, "stacked exposure time [s]")
	hdu_matrix.header["REGAREA"] = (rega, "stacked region area [deg^2]")
	if rspwt_method is not None:
		hdu_matrix.header["WTMETH"] = (rspwt_method, "response weighting method [SHP/FLX/LMN]")
	if z is not None:
		hdu_matrix.header["REDSHIFT"] = z
	if arf_fname is not None:
		hdu_matrix.header["ANCRFILE"] = (os.path.basename(arf_fname), "associated ancillary response file")
	hdu_matrix.header["CREATOR"] = "XSTACK"
	hdu_matrix.header["HISTORY"] = f"{utc_now_iso()}: stacked source RMF created by Xstack v{VERSION} [{LASTUPDATE}] [{WEB}]"
	hdu_lst.append(hdu_matrix)
	
	#--- extension 2: EBOUNDS
	cols = [
		fits.Column(name="CHANNEL",format="J",array=chan),
		fits.Column(name="E_MIN",format="D",array=ene_lo),
		fits.Column(name="E_MAX",format="D",array=ene_hi),
	]
	hdu_ebounds = fits.BinTableHDU.from_columns(cols, name="EBOUNDS")
	# RMF header following OGIP standards
	# https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis"
	hdu_ebounds.header["TELESCOP"] = spec_type
	hdu_ebounds.header["INSTRUME"] = spec_type
	hdu_ebounds.header["CHANTYPE"] = "PI"
	hdu_ebounds.header["DETCHANS"] = prob.shape[1]
	hdu_ebounds.header["HDUCLASS"] = "OGIP"
	hdu_ebounds.header["HDUCLAS1"] = "RESPONSE"
	hdu_ebounds.header["HDUCLAS2"] = "EBOUNDS"
	hdu_ebounds.header["HDUVERS"] = "1.2.0"
	hdu_lst.append(hdu_ebounds)
	
	#--- extension 3: WEIGHT
	cols = []
	if srcid_lst is not None:
		cols.append(fits.Column(name="SRCID",format="J",array=srcid_lst))
	if rspwt_lst is not None:
		cols.append(fits.Column(name="RSPWT",format="D",array=rspwt_lst))
	if len(cols) > 0:
		hdu_weight = fits.BinTableHDU.from_columns(cols, name="WEIGHT")
		hdu_lst.append(hdu_weight)
	
	hdu_lst.writeto(f"{rmf_fname}", overwrite=True)

	return


def get_tlmin_from_header(rmf_fname):
	"""
	Get first channel index from keyword TLMIN*, according to OGIP 
	standards.

	Parameters
	----------
	rmf_fname : str
		The RMF file name.

		
	Returns
	-------
	f_chan_0 : int
		First channel index. Will be unity if not found (OGIP default).
	"""
	mat_hdr = fits.getheader(rmf_fname,extname="MATRIX")
	f_chan_0 = [mat_hdr[k] for k in mat_hdr if k.startswith("TLMIN")]
	if len(f_chan_0) > 0:
		f_chan_0 = f_chan_0[0]
	else:
		f_chan_0 = 1

	return f_chan_0



#--- below for visualization purposes

def rebin_arf(arfene_lo,arfene_hi,specresp,ene_lo,ene_hi,coun,grpflg,prob=None):
	"""
	Anchor the ARF specresp (input model energy) on the output channel 
	energy grid.

	Parameters
	----------
	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin.

	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin.

	specresp : numpy.ndarray
		Effective area defined within `arfene_lo` and `arfene_hi`.

	ene_lo : numpy.ndarray
		Lower edge of output channel energy bin.

	ene_hi : numpy.ndarray
		Upper edge of output channel energy bin.

	coun : numpy.ndarray
		Net photon counts in each channel energy bin.

	grpflg : numpy.ndarray
		Channel energy grouping flag, should be passed from `rebin_pi`.

	prob : numpy.ndarray, optional
		The RMF 2D probability matrix. If given, the ARF used for 
		rebinning will be RMF-weighted. Defaults to None.

		
	Returns
	-------
	grpene_lo : numpy.ndarray
		Lower edge of grouped output channel energy bin.

	grpene_hi : numpy.ndarray
		Upper edge of grouped output channel energy bin.

	grpspecresp : numpy.ndarray
		Grouped effective area as a function of grouped output channel 
		energy.
	"""
	ene_ce = (ene_lo + ene_hi) / 2
	specresp_ali = align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp,prob)

	grpene_lo = []
	grpene_hi = []
	grpspecresp = []

	tmpene_lo = []
	tmpene_hi = []
	tmpspecresp = []
	tmpwt = []    # weight

	for i in range(len(ene_ce)):
		if grpflg[i] == 1:    # start of group
			# collect data
			# if ene_tmp_lst is empty (usually the case for the first energy bin), just skip this step
			if len(tmpene_lo)!=0:
				grpene_lo.append(tmpene_lo[0])
				grpene_hi.append(tmpene_hi[-1])
				tmpspecresp = np.array(tmpspecresp)
				tmpwt = np.array(tmpwt)
				grpspecresp.append((tmpspecresp * tmpwt / tmpwt.sum()).sum())
			tmpene_lo = [ene_lo[i]]
			tmpene_hi = [ene_hi[i]]
			tmpspecresp = [specresp_ali[i]]
			tmpwt = [coun[i]/specresp_ali[i] if coun[i]>0 else 0]   # caution! may be refined later
		elif grpflg[i] == -1:    # continuing of group
			tmpene_lo.append(ene_lo[i])
			tmpene_hi.append(ene_hi[i])
			tmpspecresp.append(specresp_ali[i])
			tmpwt.append(coun[i]/specresp_ali[i] if coun[i]>0 else 0)   # caution! may be refined later
		else: 
			raise Exception("`grpflg` not in standard format (`1` for start of group, `-1` for continuing of group)")
		
	# for the last energy bin
	grpene_lo.append(tmpene_lo[0])
	grpene_hi.append(tmpene_hi[-1])
	tmpspecresp = np.array(tmpspecresp)
	tmpwt = np.array(tmpwt)
	grpspecresp.append((tmpspecresp * tmpwt / tmpwt.sum()).sum())
	
	grpene_lo = np.array(grpene_lo)
	grpene_hi = np.array(grpene_hi)
	grpspecresp = np.array(grpspecresp)
		
	return grpene_lo,grpene_hi,grpspecresp



def align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp,prob=None):
	"""
	The ARF energy bin and RMF energy bin (also the PI channel energy 
	bin) does not always match. Align the ARF to get the effective area 
	at each RMF energy bin.

	Parameters
	----------
	ene_lo : numpy.ndarray
		Lower edge of output channel energy bin.

	ene_hi : numpy.ndarray
		Upper edge of output channel energy bin.

	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin.

	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin.

	specresp : numpy.ndarray
		The ARF specresp (cm^2 vs. arf energy).

	prob : numpy.ndarray, optional
		RMF 2D matrix (prob.shape=(len(`arfene_lo`),len(`ene_lo`))).

		
	Returns
	-------
	specresp_ali : numpy.ndarray
		The aligned ARF specresp.
	"""
	assert ene_lo.shape == ene_hi.shape, ""
	
	if prob is None:
		arfene_wd = arfene_hi - arfene_lo
		specresp_ali = np.zeros(len(ene_lo))    # aligned specresp
		for i in range(len(specresp_ali)):
			mask = (ene_lo[i] <= arfene_hi) & (ene_hi[i] >= arfene_lo)
			if np.all(mask==False):
				continue
			arfene_mask_lo = arfene_lo[mask].copy()
			arfene_mask_hi = arfene_hi[mask].copy()
			arfene_mask_wd = arfene_wd[mask].copy()
			specresp_mask = specresp[mask].copy()
			
			# for the first and last masked channel, we need to recalculate their widths
			arfene_mask_wd[0] = arfene_mask_hi[0] - ene_lo[i]
			arfene_mask_wd[-1] = ene_hi[i] - arfene_mask_lo[-1]
			
			prob_mask = arfene_mask_wd / arfene_mask_wd.sum()
			specresp_ali[i] = (specresp_mask * prob_mask).sum()

	else:
		arfene_ce = (arfene_lo + arfene_hi) / 2
		arfene_wd = arfene_hi - arfene_lo
		ene_ce = (ene_lo + ene_hi) / 2
		ene_wd = ene_hi - ene_lo
		assert prob.shape[0] == len(arfene_ce), ""
		assert prob.shape[1] == len(ene_ce), ""

		specresp_arfenewd = specresp * arfene_wd
		specresp_arfenewd_ali = np.sum(specresp_arfenewd[:,np.newaxis]*prob,axis=0)
		specresp_ali = specresp_arfenewd_ali / ene_wd
		
	return specresp_ali



#===================================================
################ Concatenating RMFs ################
#===================================================
def concat_rmf(rmf_fname1,rmf_fname2,Es,Ee,Ngrid,out_fname):
	"""
	Concatenate two RMFs into a single large RMF.
	
	Parameters
	----------
	rmf_fname1 : str
		Name of rmf with lower energy.

	rmf_fname2 : str
		Name of rmf with higher energy.

	Es : float
		Starting energy of the output rmf. Cannot be larger than minimum 
		energy of `rmf_fname1`.

	Ee : float
		Ending energy of the output rmf. Cannot be smaller than maximum 
		energy of `rmf_fname2`.

	Ngrid : int
		Number of grids between `Es` and `rmf_fname1` (also between 
		`rmf_fname1` and `rmf_fname2`, between `rmf_fname2` and `Ee`).

	out_fname : str
		Output rmf name.

		
	Returns
	-------
	prob : numpy.ndarray
		The output 2D RMF matrix.
	"""
	with fits.open(rmf_fname1) as hdu:
		mat1 = hdu["MATRIX"].data
		ebo1 = hdu["EBOUNDS"].data
		expo = hdu["MATRIX"].header["EXPOSURE"]
	arfene1_lo = mat1["ENERG_LO"]
	arfene1_hi = mat1["ENERG_HI"]
	ene1_lo = ebo1["E_MIN"]
	ene1_hi = ebo1["E_MAX"]
	n_grp1 = mat1["N_GRP"]
	f_chan1 = mat1["F_CHAN"]
	n_chan1 = mat1["N_CHAN"]
	matrix1 = np.array(mat1["MATRIX"])
	f_chan1_0 = get_tlmin_from_header(rmf_fname1)

	with fits.open(rmf_fname2) as hdu:
		mat2 = hdu["MATRIX"].data
		ebo2 = hdu["EBOUNDS"].data
	arfene2_lo = mat2["ENERG_LO"]
	arfene2_hi = mat2["ENERG_HI"]
	ene2_lo = ebo2["E_MIN"]
	ene2_hi = ebo2["E_MAX"]
	n_grp2 = mat2["N_GRP"]
	f_chan2 = mat2["F_CHAN"]
	n_chan2 = mat2["N_CHAN"]
	matrix2 = np.array(mat2["MATRIX"])
	f_chan2_0 = get_tlmin_from_header(rmf_fname2)

	assert np.max(arfene1_hi) <= np.min(arfene2_lo), "Highest model energy of `rmf_fname1` (detected: %f) should be no greater than lowest model energy (detected: %f) of `rmf_fname2` !"%(np.max(arfene1_hi),np.min(arfene2_lo))
	assert np.max(arfene1_hi) <= np.min(arfene2_lo), "Highest model energy of `rmf_fname1` (detected: %f) should be no greater than lowest model energy (detected: %f) of `rmf_fname2` !"%(np.max(arfene1_hi),np.min(arfene2_lo))
	assert np.max(ene1_hi) <= np.min(ene2_lo), "Highest channel energy of `rmf_fname1` (detected: %f) should be no greater than lowest channel energy (detected: %f) of `rmf_fname2` !"%(np.max(ene1_hi),np.min(ene2_lo))

	arfenes1 = np.logspace(np.log10(Es),np.log10(np.min(arfene1_lo)),Ngrid) # model energy grid from Es to 1st min model energy of rmf_fname1
	arfene12 = np.logspace(np.log10(np.max(arfene1_hi)),np.log10(np.min(arfene2_lo)),Ngrid) # model energy grid from last max model energy of rmf_fname1 to 1st min model energy of rmf_fname2
	arfene2e = np.logspace(np.log10(np.max(arfene2_hi)),np.log10(Ee),Ngrid) # model energy grid from last max model energy of rmf_fname2 to Ee
	arfene_lo = np.concatenate((arfenes1[:-1],arfene1_lo,arfene12[:-1],arfene2_lo,arfene2e[:-1]))   # model lower energy of the new arfene grid 
	arfene_hi = np.concatenate((arfenes1[1:],arfene1_hi,arfene12[1:],arfene2_hi,arfene2e[1:]))      # model upper energy of the new arfene grid 
	arfene_ce = (arfene_lo + arfene_hi) / 2
	arfene_wd = arfene_hi - arfene_lo
	arfene_id = np.arange(len(arfene_ce))
	didx_arfene1 = len(arfenes1) - 1    # 1st idx of rmf_fname1 in the new arfene grid
	didx_arfene2 = len(arfenes1) - 1 + len(arfene1_lo) + len(arfene12) - 1  # 1st idx of rmf_fname2 in the new arfene grid

	enes1 = np.logspace(np.log10(Es),np.log10(np.min(ene1_lo)),Ngrid)
	ene12 = np.logspace(np.log10(np.max(ene1_hi)),np.log10(np.min(ene2_lo)),Ngrid)
	ene2e = np.logspace(np.log10(np.max(ene2_hi)),np.log10(Ee),Ngrid)
	ene_lo = np.concatenate((enes1[:-1],ene1_lo,ene12[:-1],ene2_lo,ene2e[:-1]))
	ene_hi = np.concatenate((enes1[1:],ene1_hi,ene12[1:],ene2_hi,ene2e[1:]))
	ene_ce = (ene_lo + ene_hi) / 2
	ene_wd = ene_hi - ene_lo
	ene_id = np.arange(len(ene_ce))
	didx_ene1 = len(enes1) - 1    # 1st idx of rmf_fname1 in the new ene grid
	didx_ene2 = len(enes1) - 1 + len(ene1_lo) + len(ene12) - 1  # 1st idx of rmf_fname2 in the new ene grid


	grid = np.meshgrid(ene_ce,arfene_ce)    # ( (len(arfene_ce),len(ene_ce)), (len(arfene_ce),len(ene_ce)) )
	prob = np.zeros(grid[0].shape)          # probability per channel

	for i in range(len(arfene_ce)):
		if i < didx_arfene1:
			mask = (arfene_ce[i] <= ene_hi) & (arfene_ce[i] > ene_lo)
			prob[i][ene_id[mask][0]] = 1
		elif (i >= didx_arfene1) and (i < didx_arfene1 + len(arfene1_lo)):
			arfene1_idx = i - didx_arfene1
			prob[i][didx_ene1:didx_ene1+len(ene1_lo)] = get_prob1d(n_grp1[arfene1_idx],f_chan1[arfene1_idx],n_chan1[arfene1_idx],matrix1[arfene1_idx],len(ene1_lo),f_chan1_0)
		elif (i >= didx_arfene1 + len(arfene1_lo)) and (i < didx_arfene2):
			mask = (arfene_ce[i] <= ene_hi) & (arfene_ce[i] > ene_lo)
			prob[i][ene_id[mask][0]] = 1
		elif (i >= didx_arfene2) and (i < didx_arfene2 + len(arfene2_lo)):
			arfene2_idx = i - didx_arfene2
			prob[i][didx_ene2:didx_ene2+len(ene2_lo)] = get_prob1d(n_grp2[arfene2_idx],f_chan2[arfene2_idx],n_chan2[arfene2_idx],matrix2[arfene2_idx],len(ene2_lo),f_chan2_0)
		else:
			mask = (arfene_ce[i] <= ene_hi) & (arfene_ce[i] > ene_lo)
			prob[i][ene_id[mask][0]] = 1

	# in case you have any nan values
	prob[np.isclose(prob,0,rtol=1e-06, atol=1e-06, equal_nan=False)] = 0 # remove elements with probability below the 1e-6 threshold
	prob[np.isnan(prob)] = 0 # remove NaN
	prob[prob<0] = 0 # remove negative elements
	prob /= np.sum(prob,axis=1)[:,np.newaxis] # renormalize
	prob[np.isnan(prob)] = 0 # remove NaN (produced when 0/0)
	# for the first few input energies, the probability may be empty
	# assign the first channel with 1 (an arbitrary choice)
	for i in range(len(prob)):
		if np.max(prob[i]) == 0.:
			prob[i][0] = 1

	# Create fits file
	hdu_lst = fits.HDUList()
			
	# extension 0: primary hdu
	primary_hdu = fits.PrimaryHDU()
	hdu_lst.append(primary_hdu)

	# extension 1: MATRIX
	n_grp = []
	f_chan = []
	n_chan = []
	matrix = []
	for i in range(len(arfene_lo)):
		n_grp.append(1)
		f_chan.append(np.array([1]))
		prob_i = prob[i]
		# Find the index of the first non-zero element from the end
		last_nonzero_idx = len(prob_i) - np.argmax(prob_i[::-1] != 0) - 1
		n_chan.append(np.array([last_nonzero_idx+1]))
		matrix.append(prob_i[:last_nonzero_idx+1])
	n_grp = np.array(n_grp)
		
	cols = [fits.Column(name="ENERG_LO", format="D", array=arfene_lo),
			fits.Column(name="ENERG_HI", format="D", array=arfene_hi),
			fits.Column(name="N_GRP", format="J", array=n_grp),
			fits.Column(name="F_CHAN", format="PJ()", array=f_chan),
			fits.Column(name="N_CHAN", format="PJ()", array=n_chan),
			fits.Column(name="MATRIX", format="PD()", array=matrix)]
	hdu_matrix = fits.BinTableHDU.from_columns(cols, name="MATRIX")
	# RMF header following OGIP standards
	# https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis"
	hdu_matrix.header["TELESCOP"] = "CONCAT"
	hdu_matrix.header["INSTRUME"] = "CONCAT"
	hdu_matrix.header["CHANTYPE"] = "PI"
	hdu_matrix.header["DETCHANS"] = prob.shape[1]
	hdu_matrix.header["HDUCLASS"] = "OGIP"
	hdu_matrix.header["HDUCLAS1"] = "RESPONSE"
	hdu_matrix.header["HDUCLAS2"] = "RSP_MATRIX"
	hdu_matrix.header["HDUVERS"] = "1.3.0"
	hdu_matrix.header["TLMIN4"] = 1 # the first channel in the response
	hdu_matrix.header["EXPOSURE"] = expo
	hdu_matrix.header["ANCRFILE"] = "NONE"
	hdu_matrix.header["CREATOR"] = "XSTACK"
	hdu_lst.append(hdu_matrix)

	# extension 2: EBOUNDS
	chan = np.arange(1,len(ene_lo)+1)
	cols = [fits.Column(name="CHANNEL", format="J", array=chan),
			fits.Column(name="E_MIN", format="D", array=ene_lo),
			fits.Column(name="E_MAX", format="D", array=ene_hi)]
	hdu_ebounds = fits.BinTableHDU.from_columns(cols, name="EBOUNDS")
	# RMF header following OGIP standards
	# https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis"
	hdu_ebounds.header["TELESCOP"] = "CONCAT"
	hdu_ebounds.header["INSTRUME"] = "CONCAT"
	hdu_ebounds.header["CHANTYPE"] = "PI"
	hdu_ebounds.header["DETCHANS"] = prob.shape[1]
	hdu_ebounds.header["HDUCLASS"] = "OGIP"
	hdu_ebounds.header["HDUCLAS1"] = "RESPONSE"
	hdu_ebounds.header["HDUCLAS2"] = "EBOUNDS"
	hdu_ebounds.header["HDUVERS"] = "1.2.0"
	hdu_lst.append(hdu_ebounds)

	hdu_lst.writeto(f"{out_fname}", overwrite=True)

	return prob


#===================================================
################ Concatenating ARFs ################
#===================================================
def concat_arf(arf_fname1,arf_fname2,Es,Ee,Ngrid,out_fname):
	"""
	Concatenate two ARFs into a single large ARF.
	
	Parameters
	----------
	arf_fname1 : str
		Name of arf with lower energy.

	arf_fname2 : str
		Name of arf with higher energy.

	Es : float
		Starting energy of the output arf. Cannot be larger than minimum 
		energy of `arf_fname1`.

	Ee : float
		Ending energy of the output arf. Cannot be smaller than maximum 
		energy of `arf_fname2`.

	Ngrid : int
		Number of grids between `Es` and `arf_fname1` (also between 
		`arf_fname1` and `arf_fname2`, between `arf_fname2` and `Ee`).

	out_fname : str
		Output ARF name.

		
	Returns
	-------
	specresp : numpy.ndarray
		The output ARF specresp.
	"""
	with fits.open(arf_fname1) as hdu:
		arf1 = hdu["SPECRESP"].data
		expo = hdu["SPECRESP"].header["EXPOSURE"]
	arfene1_lo = arf1["ENERG_LO"]
	arfene1_hi = arf1["ENERG_HI"]
	arfene1_ce = (arfene1_lo + arfene1_hi) / 2
	arfene1_wd = arfene1_hi - arfene1_lo
	specresp1 = arf1["SPECRESP"]

	with fits.open(arf_fname2) as hdu:
		arf2 = hdu["SPECRESP"].data
	arfene2_lo = arf2["ENERG_LO"]
	arfene2_hi = arf2["ENERG_HI"]
	arfene2_ce = (arfene2_lo + arfene2_hi) / 2
	arfene2_wd = arfene2_hi - arfene2_lo
	specresp2 = arf2["SPECRESP"]

	arfenes1 = np.logspace(np.log10(Es),np.log10(np.min(arfene1_lo)),Ngrid) # model energy grid from Es to 1st min model energy of arf_fname1
	arfene12 = np.logspace(np.log10(np.max(arfene1_hi)),np.log10(np.min(arfene2_lo)),Ngrid) # model energy grid from last max model energy of rmf_fname1 to 1st min model energy of arf_fname2
	arfene2e = np.logspace(np.log10(np.max(arfene2_hi)),np.log10(Ee),Ngrid) # model energy grid from last max model energy of arf_fname2 to Ee
	arfene_lo = np.concatenate((arfenes1[:-1],arfene1_lo,arfene12[:-1],arfene2_lo,arfene2e[:-1]))   # model lower energy of the new arfene grid 
	arfene_hi = np.concatenate((arfenes1[1:],arfene1_hi,arfene12[1:],arfene2_hi,arfene2e[1:]))      # model upper energy of the new arfene grid 
	arfene_ce = (arfene_lo + arfene_hi) / 2
	arfene_wd = arfene_hi - arfene_lo
	arfene_id = np.arange(len(arfene_ce))

	specresps1 = np.ones(Ngrid-1) * specresp1[0]
	specresp12 = np.logspace(np.log10(max(specresp1[-1],1)),np.log10(max(specresp2[0],1)),Ngrid-1)
	specresp2e = np.ones(Ngrid-1) * specresp2[-1]
	specresp = np.concatenate((specresps1,specresp1,specresp12,specresp2,specresp2e))

	# make fits
	hdu_lst = fits.HDUList()

	primary_hdu = fits.PrimaryHDU()
	hdu_lst.append(primary_hdu)

	cols = [fits.Column(name="ENERG_LO", format="D", array=arfene_lo),
			fits.Column(name="ENERG_HI", format="D", array=arfene_hi),
			fits.Column(name="SPECRESP", format="D", array=specresp)]
	hdu_specresp = fits.BinTableHDU.from_columns(cols, name="SPECRESP")
	hdu_specresp.header["TELESCOP"] = "CONCAT"
	hdu_specresp.header["INSTRUME"] = "CONCAT"
	hdu_specresp.header["CHANTYPE"] = "PI"
	hdu_specresp.header["DETCHANS"] = len(specresp)
	hdu_specresp.header["HDUCLASS"] = "OGIP"
	hdu_specresp.header["HDUCLAS1"] = "RESPONSE"
	hdu_specresp.header["HDUCLAS2"] = "SPECRESP"
	hdu_specresp.header["HDUVERS"] = "1.1.0"
	hdu_specresp.header["EXPOSURE"] = expo
	hdu_specresp.header["CREATOR"] = "XSTACK"
	hdu_lst.append(hdu_specresp)

	hdu_lst.writeto(f"{out_fname}", overwrite=True)

	return specresp