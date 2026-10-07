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
from Xstack.utils.logger import utc_now_iso,add_run_cmd_history
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


def read_rsp_from_arf_rmf(arf_fname,rmf_fname):
	"""
	Read observer-frame full response matrix (RSP = ARF * RMF).

	Parameters
	----------
	arf_fname : str
		ARF file name.
	rmf_fname : str
		RMF file name.

	Returns
	-------
	rspmat : numpy.ndarray
		Observer-frame full response matrix.
	"""
	with fits.open(arf_fname) as hdu:
		arf = hdu["SPECRESP"].data
	specresp = arf["SPECRESP"]
	arfene_lo = arf["ENERG_LO"]
	arfene_hi = arf["ENERG_HI"]

	with fits.open(rmf_fname) as hdu:
		mat = hdu["MATRIX"].data
		ebo = hdu["EBOUNDS"].data
	iene_lo = mat["ENERG_LO"]
	iene_hi = mat["ENERG_HI"]

	assert np.allclose(arfene_lo,iene_lo), f"ARF/RMF input energy grids do not match: {arf_fname} vs {rmf_fname}"
	assert np.allclose(arfene_hi,iene_hi), f"ARF/RMF input energy grids do not match: {arf_fname} vs {rmf_fname}"

	f_chan_0 = get_tlmin_from_header(rmf_fname=rmf_fname)
	prob = get_prob(mat=mat,ebo=ebo,f_chan_0=f_chan_0)
	return prob * specresp[:,np.newaxis]


def shift_rsp(
		arf_fname,rmf_fname,z,nh_file=None,nh=1e20,ene_trc=None,
		ene_lo=None,ene_hi=None,
):
	"""
	Rest-frame shifting the ARF&RMF. This is literally done by three steps: 

	1. Combine GalNH-corrected ARF and RMF into a single RSP matrix 
	   (full response);
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
		  extension: ``nhene_ce``, ``nhene_wd``, ``factor``.
		- ``factor`` should indicate the absorption factor when ``nh=1e20``.
		- An easy way to obtain the ``nh_file``: iplot ``tbabs*powerlaw`` 
		  with ``Nh=1e20`` and ``PhoIndex=0.0``, ``Norm=1`` in ``XSPEC``.

	nh : float, optional
		The galactic absorption nh of the source (e.g. 3e20). Defaults 
		to ``1e20``.
	ene_trc : float, optional
		Truncate energy below which manually set ARF and PI counts to 
		zero. For eROSITA, ``ene_trc`` is typically 0.2 keV. Defaults to 
		``None``.

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
def shift_matrix_reference(prob,iene_lo,iene_hi,ene_lo,ene_hi,z):
	"""
	Older, slower, but easier-to-understand reference implementation of
	:func:`shift_matrix`.

	This function retains the original nested-loop algorithm for scientific
	verification, regression testing, and performance comparisons. Production
	Xstack processing uses :func:`shift_matrix`.

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


@jit
def _build_overlap_map(
		query_lo,query_hi,grid_lo,grid_hi,query_scale=1.0,grid_scale=1.0,
		include_touching=False,float32_widths=False,
):
	"""
	Find how each query interval overlaps a target energy grid.

	Parameters
	----------
	query_lo : numpy.ndarray
		Lower edges of intervals that need to be mapped.
	query_hi : numpy.ndarray
		Upper edges of intervals that need to be mapped.
	grid_lo : numpy.ndarray
		Lower edges of the destination grid.
	grid_hi : numpy.ndarray
		Upper edges of the destination grid.
	query_scale : float, optional
		Scale applied to each query boundary.
	grid_scale : float, optional
		Scale applied to each destination-grid boundary.
	include_touching : bool, optional
		Include bins whose boundaries only touch. This reproduces the inclusive
		comparison used by the reference implementation's vertical shift.
	float32_widths : bool, optional
		Calculate overlap widths in float32. The horizontal part of the
		reference implementation does this because its channel widths inherit
		the energy-grid dtype.

	Returns
	-------
	offsets : numpy.ndarray
		Offsets into ``indices`` and ``weights`` for each query interval.
	indices : numpy.ndarray
		Destination-bin indices, stored in increasing accumulation order.
	weights : numpy.ndarray
		Normalized overlap weights corresponding to ``indices``.
	"""
	scaled_grid_lo = grid_lo * grid_scale
	scaled_grid_hi = grid_hi * grid_scale
	grid_ubound = np.max(scaled_grid_lo)
	grid_lbound = np.min(scaled_grid_hi)

	starts = np.empty(len(query_lo),dtype=np.int64)
	stops = np.empty(len(query_lo),dtype=np.int64)
	offsets = np.zeros(len(query_lo)+1,dtype=np.int64)
	for i in range(len(query_lo)):
		lo = query_lo[i] * query_scale
		hi = query_hi[i] * query_scale
		if not include_touching and (lo > grid_ubound or hi < grid_lbound):
			first = 0
			last = 0
		elif include_touching:
			first = np.searchsorted(scaled_grid_hi,lo,side="left")
			last = np.searchsorted(scaled_grid_lo,hi,side="right")
		else:
			first = np.searchsorted(scaled_grid_hi,lo,side="right")
			last = np.searchsorted(scaled_grid_lo,hi,side="left")
		starts[i] = first
		stops[i] = last
		offsets[i+1] = offsets[i] + max(0,last-first)

	indices = np.empty(offsets[-1],dtype=np.int64)
	weights = np.empty(offsets[-1],dtype=np.float64)
	for i in range(len(query_lo)):
		first = starts[i]
		last = stops[i]
		count = last-first
		if count <= 0:
			continue

		lo = query_lo[i] * query_scale
		hi = query_hi[i] * query_scale
		start = offsets[i]
		if float32_widths:
			# Preserve the reference code's float32 channel-width arithmetic.
			widths32 = np.empty(count,dtype=np.float32)
			for j in range(count):
				index = first+j
				indices[start+j] = index
				widths32[j] = grid_hi[index]-grid_lo[index]
			widths32[0] = scaled_grid_hi[first]-lo
			widths32[-1] = hi-scaled_grid_lo[last-1]
			normalized32 = widths32/np.sum(widths32)
			for j in range(count):
				weights[start+j] = normalized32[j]
		else:
			widths64 = np.empty(count,dtype=np.float64)
			for j in range(count):
				index = first+j
				indices[start+j] = index
				widths64[j] = (grid_hi[index]-grid_lo[index])*grid_scale
			widths64[0] = scaled_grid_hi[first]-lo
			widths64[-1] = hi-scaled_grid_lo[last-1]
			normalized64 = widths64/np.sum(widths64)
			for j in range(count):
				weights[start+j] = normalized64[j]

	return offsets,indices,weights


@jit
def shift_matrix(prob,iene_lo,iene_hi,ene_lo,ene_hi,z):
	"""
	Shift an RMF probability matrix or full response matrix to the rest frame.

	The transformation shifts the output channel-energy direction followed by
	the input model-energy direction. Bin-overlap mappings are calculated once
	and then applied to the complete response matrix.

	Parameters
	----------
	prob : numpy.ndarray
		Input RMF probability matrix or full ARF*RMF response matrix. Shape
		must be ``(len(iene_lo), len(ene_lo))``.
	iene_lo : numpy.ndarray
		Lower edges of the input model-energy bins.
	iene_hi : numpy.ndarray
		Upper edges of the input model-energy bins.
	ene_lo : numpy.ndarray
		Lower edges of the output channel-energy bins.
	ene_hi : numpy.ndarray
		Upper edges of the output channel-energy bins.
	z : float
		Source redshift. Energy boundaries are multiplied by ``1 + z``.

	Returns
	-------
	shifted : numpy.ndarray
		Rest-frame shifted response matrix with the same shape as ``prob``.
	"""
	expected_shape = (len(iene_lo),len(ene_lo))
	if prob.shape != expected_shape:
		raise ValueError(
			f"Response shape {prob.shape} does not match energy grids "
			f"{expected_shape}"
		)

	# Keep an explicit identity-scale path, but still apply the legacy overlap
	# normalization below: on real response grids it is not always bitwise
	# equivalent to returning ``prob.copy()``.
	if z == 0:
		scale = 1.0
	else:
		scale = 1.0 + z

	# ------------------------------------------------------------
	# Step 1: shift the output-channel-energy direction.
	#
	# channel_map[j] describes where observed channel j lands after
	# multiplying its energy boundaries by (1 + z).
	# ------------------------------------------------------------
	channel_offsets,channel_indices,channel_weights = _build_overlap_map(
		ene_lo,ene_hi,ene_lo,ene_hi,scale,1.0,False,True,
	)

	horizontal = np.zeros(prob.shape,dtype=np.float64)
	model_ubound = np.max(iene_lo)
	model_lbound = np.min(iene_hi)
	for model_row in range(len(iene_lo)):
		model_lo = iene_lo[model_row]*scale
		model_hi = iene_hi[model_row]*scale
		if model_lo > model_ubound:
			break
		if model_hi < model_lbound:
			continue

		prob_1d = np.zeros(len(ene_lo),dtype=np.float64)
		for old_channel in range(len(ene_lo)):
			start = channel_offsets[old_channel]
			stop = channel_offsets[old_channel+1]
			for position in range(start,stop):
				new_channel = channel_indices[position]
				prob_1d[new_channel] += (
					prob[model_row][old_channel]*channel_weights[position]
				)

		# to deal with the high energy tail; we want to make sure that the
		# sum along horizontal axis equals to arf specresp in the energy
		if np.sum(prob_1d) > 0:
			prob_1d *= np.sum(prob[model_row])/np.sum(prob_1d)
		horizontal[model_row] = prob_1d

	# ------------------------------------------------------------
	# Step 2: shift the input-model-energy direction.
	#
	# model_map[i] describes which redshifted input model-energy rows
	# contribute to rest-frame model-energy row i.
	# ------------------------------------------------------------
	model_offsets,model_indices,model_weights = _build_overlap_map(
		iene_lo,iene_hi,iene_lo,iene_hi,1.0,scale,True,False,
	)

	shifted = np.zeros(prob.shape,dtype=np.float64)
	for new_row in range(len(iene_lo)):
		start = model_offsets[new_row]
		stop = model_offsets[new_row+1]
		if start == stop:
			continue
		old_rows = model_indices[start:stop]
		weights = model_weights[start:stop]
		shifted[new_row] = np.sum(
			horizontal[old_rows].copy()*weights[:,np.newaxis],axis=0,
		)

	return shifted


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

		- ``SHP``: assuming all sources have same spectral shape
		- ``FLX``: assuming all sources have same spectral shape and flux

		  - For point sources (``extended==False``), flux is in units of 
		    erg/cm^2/s.
		  - For extended sources (``extended==True``), flux is in units 
		    of erg/cm^2/s/deg^2.

		- ``LMN``: assuming all sources have same luminosity

		  - For point sources (``extended==False``), luminosity is in 
		    units of erg/s.
		  - For extended sources (``extended==True``), luminosity is in 
		    units of erg/s/deg^2.

	extended : bool, optional
		Whether or not the source is extended. Defaults to ``False``, i.e., 
		a point source.
	rega : int or float, optional
		``REGAREA`` list. Used when ``extended==True``.

	Returns
	-------
	rspwt : numpy.ndarray
		The RSP weight for each source.
	rspnorm : float
		The RSP weight normalization. This is only useful for ``SHP`` mode.
	expo_stacked : float
		The final ``EXPOSURE`` to be written in the header of stacked PI
		and RSP.
	rega_stacked : float
		The final ``REGAREA`` to be written in the header of stacked PI
		and RSP.

	Notes
	-----
	The ideal choice of ``method`` should be ``SHP``, which starts from the 
	minimum assumption and thus gives the most unbiased results on 
	spectral shape. A caveat of ``SHP`` is that the individual spectrum 
	should have sufficient photon counts (>~10), and the resulting 
	stacked spectrum does not carry a physical flux unit.

	The second option of ``method``, in case the individual photon counts
	is too low, should be ``FLX``. In addition to the minimum assumption 
	used by ``SHP``, it assumes that all sources have similar flux (
	erg/cm^2/s for point source or erg/cm^2/s/deg^2 for extended).
	This should be reasonable for a flux-limited survey, where most 
	sources lie around the detection flux limit, and should thus have 
	similar flux.

	``LMN`` is similar to ``FLX``, except it assumes all sources to be 
	summed have similar luminosity. Note, luminosity can be calcualted in 
	``XSPEC`` as e.g., flux 0.5 2 --> luminosity in 0.5-2 keV band / 1e60 
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


def get_folded_model_rate(
		rspmat,ene_lo,ene_hi,iene_lo,iene_hi,flg,gamma=2.0,
	):
	"""
	Fold a reference power law through a full response matrix.

	The absolute normalization of the reference power law is arbitrary and
	cancels when rates from two response matrices are divided.

	Parameters
	----------
	rspmat : numpy.ndarray
		Full response matrix. Axis 0 is input model energy and axis 1 is
		output-channel energy.
	ene_lo, ene_hi : numpy.ndarray
		Lower and upper edges of output-channel energy bins.
	iene_lo, iene_hi : numpy.ndarray
		Lower and upper edges of input model-energy bins.
	flg : numpy.ndarray
		Boolean selection of output channels used for normalization.
	gamma : float, optional
		Photon index of the reference power law. Defaults to ``2.0``.

	Returns
	-------
	rate : float
		Reference-model count rate in the selected output-channel band.
	"""
	rspmat = np.asarray(rspmat,dtype=np.float64)
	# FITS response grids are commonly float32 in input files, while Xstack
	# writes the stacked grids as float64. Always do the center/width
	# arithmetic in float64 so an in-memory response and the same response
	# read back from the output FITS file fold identically.
	ene_lo = np.asarray(ene_lo,dtype=np.float64)
	ene_hi = np.asarray(ene_hi,dtype=np.float64)
	iene_lo = np.asarray(iene_lo,dtype=np.float64)
	iene_hi = np.asarray(iene_hi,dtype=np.float64)
	flg = np.asarray(flg,dtype=bool)

	if ene_lo.shape != ene_hi.shape:
		raise ValueError("ene_lo and ene_hi must have identical shapes.")
	if iene_lo.shape != iene_hi.shape:
		raise ValueError("iene_lo and iene_hi must have identical shapes.")
	if rspmat.shape != (len(iene_lo),len(ene_lo)):
		raise ValueError(
			f"rspmat has shape {rspmat.shape}; expected "
			f"{(len(iene_lo),len(ene_lo))}."
		)
	if flg.shape != ene_lo.shape:
		raise ValueError("flg must have one element per output channel.")
	if not np.any(flg):
		raise ValueError("No output channels were selected for normalization.")

	iene_ce = (iene_lo + iene_hi) / 2
	iene_wd = iene_hi - iene_lo
	model = iene_ce**(-gamma)
	folded = np.sum(
		rspmat * model[:,np.newaxis] * iene_wd[:,np.newaxis],axis=0,
	)
	rate = np.sum(folded[flg])

	if not np.isfinite(rate) or rate <= 0:
		raise ValueError(
			"The response produces a non-positive or non-finite reference-model count rate."
		)

	return rate


def rescale_rspmat(
		rspmat,rspwt_lst,expo_lst,rega_lst,rspwt_method,extended=False,
		shp_normalization="LEGACY",norm_rspmat=None,
		ene_lo=None,ene_hi=None,iene_lo=None,iene_hi=None,flg=None,gamma=2.0,
	):
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
		Extended or not. Defaults to ``False``.
	shp_normalization : str, optional
		Absolute normalization for an SHP-weighted response. ``LEGACY``
		preserves the historical behavior; ``FLX`` and ``LMN`` preserve the
		SHP response shape while adopting the corresponding physical scale.
		Ignored when ``rspwt_method`` is not ``SHP``.
	norm_rspmat : numpy.ndarray, optional
		Raw auxiliary FLX- or LMN-weighted response matrix. Required for
		``FLX``- or ``LMN``-normalized SHP.
	ene_lo, ene_hi : numpy.ndarray, optional
		Output-channel energy-bin edges used for physical SHP normalization.
	iene_lo, iene_hi : numpy.ndarray, optional
		Input model-energy-bin edges used for physical SHP normalization.
	flg : numpy.ndarray, optional
		Output-channel selection used for physical SHP normalization.
	gamma : float, optional
		Reference power-law photon index. Defaults to ``2.0``.

	Returns
	-------
	rspmat : numpy.ndarray
		Rescaled RSP matrix.
	rspnorm : float
		To prevent overflow of very large number in the case of ``LMN`` 
		mode, the rescaled RSP matrix has been multiplied by a very small 
		number. Multiply your ``rspmat`` by ``rspnorm`` to bring it back to 
		the appropriate number.
	rspwt_lst : numpy.ndarray
		List of response weighting factors.
	expo_stk : float
		Stacked exposure.
	rega_stk : float
		Stacked region area.
	shp_renorm : float
		Additional FLX/LMN normalization applied to an SHP response. This is
		``1.0`` for legacy SHP and ordinary FLX/LMN operation.
	"""
	rspwt_method = str(rspwt_method).upper()
	shp_normalization = str(shp_normalization).upper()
	rspwt_lst = np.asarray(rspwt_lst,dtype=np.float64)
	expo_lst = np.asarray(expo_lst,dtype=np.float64)
	rega_lst = np.asarray(rega_lst,dtype=np.float64)
	shp_renorm = 1.0

	# FLX and LMN share the same exposure/area rescaling. LMN introduces
	# the additional factor of 1e60 below.
	if extended:
		physical_expo_stk = np.sum(expo_lst * rega_lst) / np.sum(rega_lst)
		physical_rega_stk = 1.0
		physical_scale = 1 / (physical_expo_stk * physical_rega_stk)
	else:
		physical_expo_stk = np.sum(expo_lst)
		physical_rega_stk = 1.0
		physical_scale = 1 / physical_expo_stk

	if rspwt_method == "SHP":
		# First apply the existing SHP shape normalization.
		shp_scale = 1 / np.sum(rspwt_lst)
		rspmat *= shp_scale

		if shp_normalization == "LEGACY":
			final_scale = shp_scale
			rspnorm = final_scale
			expo_stk = np.sum(expo_lst)
			rega_stk = 1.0

		elif shp_normalization in ("FLX","LMN"):
			if norm_rspmat is None:
				raise ValueError(
					"norm_rspmat is required for FLX- or LMN-normalized SHP."
				)
			if any(value is None for value in (ene_lo,ene_hi,iene_lo,iene_hi,flg)):
				raise ValueError(
					"Energy grids and channel selection are required for "
					"FLX- or LMN-normalized SHP."
				)

			norm_scale = physical_scale
			if shp_normalization == "LMN":
				# Keep Xstack's existing convention: the fitted cflux value is
				# the rest-frame luminosity divided by 1e60.
				norm_scale *= 1e60
			norm_rspmat *= norm_scale

			# Anchor the matrices as they will actually appear in the output
			# ARF+RMF. extract_arf_rmf_from_rspmat applies Xstack's RMF
			# probability threshold and row renormalization, which can otherwise
			# introduce a small mismatch between the in-memory full response and
			# the response reconstructed from the written FITS files.
			shp_specresp,shp_prob = extract_arf_rmf_from_rspmat(rspmat)
			norm_specresp,norm_prob = extract_arf_rmf_from_rspmat(norm_rspmat)
			shp_rate = get_folded_model_rate(
				shp_prob * shp_specresp[:,np.newaxis],
				ene_lo,ene_hi,iene_lo,iene_hi,flg,gamma=gamma,
			)
			norm_rate = get_folded_model_rate(
				norm_prob * norm_specresp[:,np.newaxis],
				ene_lo,ene_hi,iene_lo,iene_hi,flg,gamma=gamma,
			)
			shp_renorm = norm_rate / shp_rate
			if not np.isfinite(shp_renorm) or shp_renorm <= 0:
				raise ValueError("Invalid SHP normalization factor.")

			rspmat *= shp_renorm
			final_scale = shp_scale * shp_renorm
			rspnorm = final_scale
			expo_stk = physical_expo_stk
			rega_stk = physical_rega_stk

		else:
			raise ValueError(
				"shp_normalization must be `LEGACY`, `FLX`, or `LMN`."
			)

	elif rspwt_method == "FLX":
		final_scale = physical_scale
		rspnorm = 1.0
		expo_stk = physical_expo_stk
		rega_stk = physical_rega_stk
		rspmat *= final_scale

	elif rspwt_method == "LMN":
		final_scale = physical_scale * 1e60
		rspnorm = 1e60
		expo_stk = physical_expo_stk
		rega_stk = physical_rega_stk
		rspmat *= final_scale

	else:
		raise ValueError(
			"Available methods for response scaling are `SHP`, `FLX`, and `LMN`."
		)

	rspwt_lst *= final_scale
	return rspmat,rspnorm,rspwt_lst,expo_stk,rega_stk,shp_renorm


def correct_arf(specresp,arfene_lo,arfene_hi,factor,nhene_lo,nhene_hi,nh):
	"""
	Multiply the ARF specresp with the galactic absorption profile. 
	The template galactic absorption profile should be at nh=1e20.
	The source nh value is specified by ``nh``.

	Parameters
	----------
	specresp : numpy.ndarray
		The ARF specresp to be corrected.
	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin.
	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin.
	factor : numpy.ndarray
		Template galactic absorption profile at ``nh=1e20``.
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

		- ``CHANNEL``: project on output channel energy axis. Note that to 
		  do this projection, we would nevertheless need to assume a 
		  spectral slope, or photo index (specified in ``gamma``). This is 
		  to match the convention of unfolded spectrum (in e.g., ``XSPEC``), 
		  where the effective area anchored on channel energy axis is in 
		  fact (folded model)/(model).
		- ``MODEL``: project on input model energy axis
		
		Defaults to ``CHANNEL``.
	gamma : float, optional
		The spectral slope. Defaults to ``2.0``. This is only used when 
		``proj_axis`` is ``CHANNEL``. For AGN sources, a powerlaw with 
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



def get_prob(mat,ebo,f_chan_0=0):
	"""
	Parse the RMF file (input the ``MATRIX`` and ``EBOUNDS`` extension) into 
	a 2D probability matrix. 

	Parameters
	----------
	mat : astropy.io.fits.FITS_rec
		The ``MATRIX`` extension of a standard OGIP RMF file. Must include 
		the following columns:

		- ``ENERG_LO``
		- ``ENERG_HI``
		- ``N_GRP``
		- ``F_CHAN``
		- ``N_CHAN``
		- ``MATRIX``

	ebo : astropy.io.fits.FITS_rec
		The ``EBOUNDS`` extension of a standard OGIP RMF file. Must include 
		the following columns:

		- ``E_MIN`` 
		- ``E_MAX``

	f_chan_0 : int, optional
		First channel index. Defaults to ``None``. 
		If not specified, will be determined from rmf file.
	
	Returns
	-------
	prob : numpy.ndarray
		The RMF 2D probability matrix. Index ``[i,j]``, where:

		- ``i`` represents arfene (iene or inpu model energy)
		- ``j`` represents ene (output channel energy)
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


def get_prob1d(n_grp,f_chan,n_chan,matrix1d,Nene,f_chan_0=0):
	"""
	Get the 1d probability distribution for output channel energy at a 
	specific input model energy.

	Parameters
	----------
	n_grp : int
		``N_GRP`` array of your specific input model energy, from ``MATRIX`` 
		extension.
	f_chan : int
		``F_CHAN`` array of your specific input model energy, from ``MATRIX`` 
		extension.
	n_chan : int
		``N_CHAN`` array of your specific input model energy, from ``MATRIX`` 
		extension.
	matrix1d : numpy.ndarray
		``MATRIX`` array of your specific input model energy, from ``MATRIX`` 
		extension.
	Nene : int
		Length of output channel energy.
	f_chan_0 : int, optional
		The index number of the first output channel energy (``0`` or ``1``). 
		Defaults to ``0``.
		
	Returns
	-------
	prob1d : numpy.ndarray
		The 1d probability distribution for output channel energy at a 
		specific input model energy.
	"""
	f_matrix = 0   # starting index of matrix1d
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
		ARF effective area as a function of input energies (``iene``).
	prob : numpy.ndarray
		2D probability RMF matrix as a function of input (``iene``) and 
		output (``ene``) energies.
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
		shp_normalization="LEGACY",shp_renorm=1.0,
		norm_elo=None,norm_ehi=None,norm_gamma=None,
		srcid_lst=None,rspwt_lst=None,pi_totcts_lst=None,bkgpi_totcts_lst=None,flg=None,
		spec_type="STACKED",z=None,run_cmd=None,
):
	"""
	Write ARF file according to OGIP standards.
	Assume all spectral files (PI, ARF, RMF) under the same path for ``XSPEC`` convenience.
	
	Parameters
	----------
	arfene_lo : numpy.ndarray
		Lower edge of input model energy (ARF energy) bin, aka ``iene_lo``.
	arfene_hi : numpy.ndarray
		Upper edge of input model energy (ARF energy) bin, aka ``iene_hi``.
	specresp : numpy.ndarray
		Effective area defined within ``arfene_lo`` and ``arfene_hi``.
	arf_fname : str, optional
		Output ARF name. Defaults to ``stacked_arf.fits``.
	detchans : int, optional
		Number of detected channels. This should be the length of PI spectral 
		channels, or equivalently the length of ``ene``. Defaults to ``1000``.
	expo : int or float, optional
		Exposure in units of s. Defaults to ``10``.
	rega : int or float, optional
		Region area in units of :math:`\mathrm{deg}^2`. Defaults to ``1``.
	rspwt_method : str, optional
		Response weighting method. Defaults to ``SHP``.
	rspnorm : int or float, optional
		To prevent overflow of very large number in the case of ``LMN`` 
		mode, the rescaled RSP matrix has been multiplied by a very small 
		number. Multiply your ``rspmat`` by ``rspnorm`` to bring it back to 
		the appropriate number. Defaults to ``1``.
	shp_normalization : str, optional
		Absolute SHP normalization: ``LEGACY``, ``FLX``, or ``LMN``.
	shp_renorm : float, optional
		Additional FLX/LMN normalization applied to the SHP response.
	norm_elo, norm_ehi : float, optional
		Lower and upper rest-frame normalization-band edges in keV.
	norm_gamma : float, optional
		Reference power-law photon index used for physical SHP normalization.
	srcid_lst : numpy.ndarray, optional
		Source id list. Defaults to ``None``.
	rspwt_lst : numpy.ndarray, optional
		Response weighting factor list. Defaults to ``None``.
	pi_totcts_lst : numpy.ndarray, optional
		PI spectrum total counts. Defaults to ``None``.
	bkgpi_totcts_lst : numpy.ndarray, optional
		BKG PI spectrum total counts. Defaults to ``None``.
	flg : numpy.ndarray, optional
		Which channels used in ``SHP`` mode in calculating response weighting
		factors. Defaults to ``None``.
	spec_type : str, optional
		``STACKED`` if this is the stacked ARF. ``RESTFRAM`` if this is single 
		source rest-frame ARF. Defaults to ``STACKED``.
	z : float, optional
		Source redshift if this is single source rest-frame ARF.
	run_cmd : str, optional
		Full command string recorded in ``HISTORY`` for provenance.

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
	if rspwt_method == "SHP":
		hdu_specresp.header["SHPNORM"] = (shp_normalization, "SHP absolute normalization")
		hdu_specresp.header["SHPRENOR"] = (shp_renorm, "additional SHP normalization factor")
		if norm_elo is not None:
			hdu_specresp.header["NORMELO"] = (norm_elo, "normalization-band lower edge [keV]")
		if norm_ehi is not None:
			hdu_specresp.header["NORMEHI"] = (norm_ehi, "normalization-band upper edge [keV]")
		if norm_gamma is not None:
			hdu_specresp.header["NORMGAM"] = (norm_gamma, "normalization reference photon index")
		if shp_normalization == "LMN":
			hdu_specresp.header["LMNSCALE"] = (1e60, "multiply cflux by this for erg/s")
	hdu_specresp.header["CREATOR"] = "XSTACK"
	hdu_specresp.header["HISTORY"] = f"{utc_now_iso()}: stacked source ARF created by Xstack v{VERSION} [{LASTUPDATE}] [{WEB}]"
	add_run_cmd_history(hdu_specresp.header,run_cmd)
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
		spec_type="STACKED",z=None,run_cmd=None,
):
	"""
	Write RMF file according to OGIP standards.
	Assume all spectral files (PI, ARF, RMF) under the same path for ``XSPEC`` convenience.

	Parameters
	----------
	chan : numpy.ndarray
		PI Channel. Must be the same length as ``ene``.
	ene_lo : numpy.ndarray
		Lower edge of output channel energy (PI energy) bin.
	ene_hi : numpy.ndarray
		Upper edge of output channel energy (PI energy) bin.
	prob : numpy.ndarray
		2D RMF matrix.
	rmf_fname : str, optional
		Output RMF name. Defaults to ``stacked_rmf.fits``.
	expo : int or float, optional
		Exposure in units of :math:`\mathrm{s}`. Defaults to ``10``.
	rega : int or float, optional
		Region area in units of deg^2. Defaults to ``1``.
	rspwt_method : str, optional
		Response weighting method. Defaults to ``SHP``.
	srcid_lst : numpy.ndarray, optional
		Source id list. Defaults to ``None``.
	rspwt_lst : numpy.ndarray, optional
		Response weighting factor list. Defaults to ``None``.
	arf_fname : str, optional
		Associated ARF name. Defaults to ``stacked_arf.fits``.
	spec_type : str, optional
		``STACKED`` if this is the stacked ARF. ``RESTFRAM`` if this is single 
		source rest-frame ARF. Defaults to ``STACKED``.
	z : float, optional
		Source redshift if this is single source rest-frame ARF.
	run_cmd : str, optional
		Full command string recorded in ``HISTORY`` for provenance.
		
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
	add_run_cmd_history(hdu_matrix.header,run_cmd)
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
	Get first channel index from keyword ``TLMIN*``, according to OGIP 
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
		f_chan_0 = int(f_chan_0[0])
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
		Effective area defined within ``arfene_lo`` and ``arfene_hi``.
	ene_lo : numpy.ndarray
		Lower edge of output channel energy bin.
	ene_hi : numpy.ndarray
		Upper edge of output channel energy bin.
	coun : numpy.ndarray
		Net photon counts in each channel energy bin.
	grpflg : numpy.ndarray
		Channel energy grouping flag, should be passed from ``rebin_pi``.
	prob : numpy.ndarray, optional
		The RMF 2D probability matrix. If given, the ARF used for 
		rebinning will be RMF-weighted. Defaults to ``None``.

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
		RMF 2D matrix (prob.shape=(len(``arfene_lo``),len(``ene_lo``))).

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
		energy of ``rmf_fname1``.
	Ee : float
		Ending energy of the output rmf. Cannot be smaller than maximum 
		energy of ``rmf_fname2``.
	Ngrid : int
		Number of grids between ``Es`` and ``rmf_fname1`` (also between 
		``rmf_fname1`` and ``rmf_fname2``, between ``rmf_fname2`` and ``Ee``).
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
		energy of ``arf_fname1``.
	Ee : float
		Ending energy of the output arf. Cannot be smaller than maximum 
		energy of ``arf_fname2``.
	Ngrid : int
		Number of grids between ``Es`` and ``arf_fname1`` (also between 
		``arf_fname1`` and ``arf_fname2``, between ``arf_fname2`` and ``Ee``).
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
