#!/usr/bin/env python3
"""
===========================================
Module for shifting and stacking PI spectra
===========================================
:Authors:   Shi-Jiang Chen (MPE, USTC)
            Johannes Buchner (MPE)
            Teng Liu (USTC)
:Email:     JohnnyCsj666@gmail.com


"""
import numpy as np
from astropy.io import fits
import os


def shift_pi(pi_file,rmf_file,z,ene_trc=None):
    """
    Shift a single PI to rest-frame.
    
    Parameters
    ----------
    pi_file : str
        Observed-frame pi file to be shifted, in standard OGIP format.
    rmf_file : str
        RMF file defining channel-energy conversion, in standard OGIP 
        format.
    z : float
        Redshift.
    ene_trc : float, optional
        Truncate energy [keV] below which manually set ARF and PI counts 
        to zero. For eROSITA, `ene_trc` is typically 0.2 keV.
    
    Returns
    -------
    rest_chan : list
        Rest-frame channel.
    rest_coun : list
        Photon counts in each rest-frame channel.
    pi_chan : list
        Observed-frame channel.
    pi_coun : list
        Photon counts in each observed-frame channel.
    """
    with fits.open(rmf_file) as hdu:
        mat = hdu["MATRIX"].data
        ebo = hdu["EBOUNDS"].data
    
    chan = ebo["CHANNEL"]
    ene_lo = ebo["E_MIN"]
    ene_hi = ebo["E_MAX"]
    ene_ce = (ene_lo + ene_hi)/2
    ene_wd = ene_hi - ene_lo
    ene_id = np.arange(len(ene_ce))
    
    with fits.open(pi_file) as hdu:
        pi = hdu["SPECTRUM"].data
    pi_chan = pi["CHANNEL"] # pi_chan starts from 0
    pi_coun = pi["COUNTS"] # the obs-frame photon counts
    assert (pi_chan == chan).all()
    chan_id = np.arange(len(pi_chan))

    # truncate below ene_trc
    if ene_trc is not None:
        idx_trc = np.argmin(abs(ene_ce-ene_trc))
        pi_coun[:idx_trc] = 0
    
    rest_chan = pi_chan.copy()
    # rest_coun = np.zeros(len(rest_chan),dtype=int) # the src-frame photon counts
    rest_coun = np.zeros(len(rest_chan),dtype=int) # the src-frame photon counts
    
    ene_ubound = ene_lo.max()
    ene_lbound = ene_hi.min() # set lower and upper bound of energy to avoid overflow issues
    
    for i in range(len(pi_chan)):
        #print(i)
        ene_lo_map = ene_lo[i] * (1+z)
        ene_hi_map = ene_hi[i] * (1+z)
        if ene_lo_map > ene_ubound:
            continue
        if ene_hi_map < ene_lbound:
            continue
        
        mask = (ene_hi > ene_lo_map) & (ene_lo < ene_hi_map)
        ene_id_mask = ene_id[mask]
        ene_wd_mask = ene_wd[mask]
        ene_lo_mask = ene_lo[mask]
        ene_hi_mask = ene_hi[mask]
        
        chan_id_mask = chan_id[mask]
        
        # For the first and last channel in the basket, we need to recalculate their widths
        # This is because they are defined by chan_lo_map[i] and chan_hi_map[i], respectively
        ene_wd_mask[0] = ene_hi_mask[0] - ene_lo_map
        ene_wd_mask[-1] = ene_hi_map - ene_lo_mask[-1]
        
        # Each channel in the basket would get number of photons proportional to its width
        prob_mask = ene_wd_mask / ene_wd_mask.sum() # the probability of entering each channel in the basket
        # # NOTE: we do not force each channel to have integer number of photon counts at this step
        # # Instead, we round them to integers after stacking all sources (`add_pi`), where each channel accumulates sufficient photon counts
        # # This would be helpful when each individual source has only few counts in total
        # phoct_mask = (pi_coun[i] * prob_mask).astype(float)   # this is float

        phoct_mask = (pi_coun[i] * prob_mask).astype(int)
        # if there are more than 1 channel in the basket, we want to make sure that the sum of photons in all bins are equal to pi_coun[i]
        if len(phoct_mask) > 1:
            phoct_mask[0] = pi_coun[i] - phoct_mask[1:].sum()
        
        # finally, assign the photons
        for idx in range(len(chan_id_mask)):
            try:
                rest_coun[chan_id_mask[idx]] += phoct_mask[idx]
            except IndexError:
                continue

    return (rest_chan, rest_coun, pi_chan, pi_coun)


def add_pi(pi_lst,scal_lst=None,fits_name=None,expo=10,bkg_file=None,rmf_file=None,arf_file=None):
    """
    The weighted sum of many PI files. The weights are specified by 
    `scal_lst`. 
    
    If source PIs are to be summed, the weights should all be unity.
    Otherwise if background PIs are to be summed, the weights should 
    be (as a return from function `get_bkgscal`):

    ..

        `AREASCAL`_src / `AREASCAL`_src * `BACKSCAL`_src / 
        `BACKSCAL`_bkg * `EXPOSURE`_src / `EXPOSURE`_bkg
    
    
    The uncertainty in each channel is calculated with Gaussian error 
    propogation.
    
    Caution should be taken when adding background spectra with varing 
    scaling ratios:

    1. If all backgrounds have same scaling ratio, we can simply add them 
       together. 
    2. However if the scaling ratio for each spectrum varies, we need to 
       scale them first before summing together. Since the scaling ratio 
       for background spectrum is often a number much smaller than 1, 
       each scaled background spectrum may have float number of photon 
       counts < 1 in some channel i. In this case, the uncertainty in 
       channel i cannot be calculated with Poisson statistics 
       (i.e. :math: `\sqrt{N}` ). 
    3. To conclude, `add_bkgpi` should be used when considering error of 
       stacked background spectra with varied scaling ratio. `add_bkgpi` 
       first group background spectra with similar scaling ratios, then 
       calculate error for each group with Poisson statistics (each 
       channel has enough photon counts now), and finally calculate the 
       error for the total summed background spectra (each group of 
       spectra is in high-counts regime, so Gaussian error propagation 
       works).
    4. Nevertheless for the photon counts (rather than the error), it is 
       still recommended to use `add_pi`.
    
    Parameters
    ----------
    pi_lst : list or numpy.ndarray
        PI file list.
    scal_lst : list or numpy.ndarray, optional
        Weight (scaling ratio) list. Defaults to None (unity).
    fits_name : str, optional
        If specified, create a fits file with name `fits_name`. 
        Defaults to None.
    expo : float, optional
        Total exposure time (seconds) to be written in the header of 
        `fits_name`. Defaults to 10.
    bkg_file : str, optional
        Stacked background PI fits name to be written in the header of 
        `fits_name`. Defaults to None.
    rmf_file : str, optional
        Stacked RMF fits name to be written in the header of `fits_name`. 
        Defaults to None.
    bkg_file : str, optional
        Stacked ARF fits name to be written in the header of `fits_name`. 
        Defaults to None.
    
    Returns
    -------
    sum_pi : numpy.ndarray
        Stacked PI array.
    sum_pierr : numpy.ndarray
        Stacked PI error array.

    Notes
    -----
    The Net counts for some source:

    .. math::

        \mathrm{net\ counts} = \mathrm{source\ counts} - 
        \mathrm{background\ counts} \times \mathrm{scaling\ factor}
        
    where: 
    
    - `scaling factor` = `AREASCAL`_src / `AREASCAL`_src * `BACKSCAL`_src 
      / `BACKSCAL`_bkg * `EXPOSURE`_src / `EXPOSURE`_bkg,
      and should be a return from function `get_bkgscal`.
        
    The error propagation tells us that:

    .. math::
    
        Err(net)^2 = Err(source)^2 + Err(background)^2 \times 
        \mathrm{scaling\ factor}^2
        
    Assuming Poisson distribution,

    .. math::
    
        Err(source)^2 = \mathrm{source\ counts} \\
        Err(background)^2 = \mathrm{background\ counts}
        
    So the error for source spectrum and **scaled** background spectrum 
    to be used in background subtracting should be:

    .. math::
        
        Err(source,scaled) = \sqrt{\mathrm{source\ counts}} \\
        Err(background,scaled) = \sqrt{\mathrm{background\ counts}} 
        \times \mathrm{scaling\ factor}
        
    Note for source spectrum, scaling factor is by convention 1;
    so the two equations have same mathematical form.
    
    Also note that the additional factor for the error is **scaling 
    factor**, not **:math: `\sqrt{\mathrm{scaling\ factor}}`**. 
    This should be reasonable:
    
    1. :math: `\sqrt{\mathrm{background\ counts}} / \sqrt{\mathrm{scaling
      \ factor}}` tells us the error of background spectrum, 
      when it is estimated from a region as small as the source region. 
    
    2. However, we often estimate the background from a region much 
      larger than the source region (determined by scaling 
      factor): so we should have better understanding of the background 
      spectrum, and therefore smaller uncertainties, as expected from 
      :math: `\sqrt{\mathrm{background\ counts}} / \mathrm{scaling\ factor}`.
    
    """
    pi_lst = np.array(pi_lst)
    if scal_lst is None:
        scal_lst = np.ones(pi_lst.shape[0])
    scal_lst = np.array(scal_lst)
    assert pi_lst.shape[0] == scal_lst.shape[0], "pi number and ratio number do not match!"
    
    # For spectral counts
    pi_scal_lst = pi_lst * scal_lst[:,np.newaxis]
    sum_pi = np.sum(pi_scal_lst, axis=0)
    # # We round photon counts in each channel to nearest integer, to approximate Poisson
    # # which is necessary to calculate uncertainties
    # sum_pi = np.round(np.sum(pi_scal_lst, axis=0)).astype(int)
    
    # For spectral counts uncertainties
    # Gaussian error propagation: each channel of `pi_scal_lst` has to have enough photon counts!
    # But this is generally not the case for bkg spectra (scal_lst << 1), so function `add_bkgpi` should be used instead!
    pierr_lst = np.sqrt(pi_lst) # Poisson statistics
    pierr_scal_lst = pierr_lst * scal_lst[:,np.newaxis] # see explanations above
    sum_pierr = np.sqrt(np.sum(pierr_scal_lst**2, axis=0))
    
    # Write fits file (optional)
    if fits_name is not None:
        hdulist = fits.HDUList()
    
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        channels = np.arange(1,len(sum_pi)+1)
        cols = [fits.Column(name="CHANNEL", format="I", array=channels),
                fits.Column(name="COUNTS", format="J", array=sum_pi),
                fits.Column(name="STAT_ERR", format="D", array=sum_pierr)]
        hdu_spectrum = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
        hdulist.append(hdu_spectrum)

        # PI header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, OGIP/92-007: "The OGIP Spectral File Format")
        hdu_spectrum.header["TELESCOP"] = "STACKED"
        hdu_spectrum.header["INSTRUME"] = "STACKED"
        hdu_spectrum.header["EXPOSURE"] = expo
        if bkg_file is not None:
            hdu_spectrum.header["BACKFILE"] = bkg_file
        hdu_spectrum.header["BACKSCAL"] = 1.0
        hdu_spectrum.header["CORRSCAL"] = 1.0
        if rmf_file is not None:
            hdu_spectrum.header["RESPFILE"] = rmf_file
        if arf_file is not None:
            hdu_spectrum.header["ANCRFILE"] = arf_file
        hdu_spectrum.header["AREASCAL"] = 1.0
        hdu_spectrum.header["HDUCLASS"] = "OGIP"
        hdu_spectrum.header["HDUCLAS1"] = "SPECTRUM"
        hdu_spectrum.header["HDUVERS"] = "1.2.1"
        hdu_spectrum.header["POISSERR"] = False # statistical errors specified in `STAT_ERR` instead
        hdu_spectrum.header["CHANTYPE"] = "PI"
        hdu_spectrum.header["DETCHANS"] = len(channels)
        hdu_spectrum.header["CREATOR"] = "XSTACK"
        hdu_spectrum.header["HDUCLAS2"] = "TOTAL"
        hdu_spectrum.header["HDUCLAS3"] = "COUNT"
        
        hdulist.writeto(f"{fits_name}", overwrite=True)
        
    return sum_pi,sum_pierr


def add_bkgpi(bkgpi_lst,bkgscal_lst,Ngrp=10,fits_name=None,expo=10):
    """
    The weighted sum of background PI files. The weights are specified by 
    `bkgscal_lst`.
    
    Group sources into bins of similar bkg-to-source scaling ratio 
    (considering both `BACKSCAL` and `EXPOSURE`) For each group, sum the 
    background counts, and compute the uncertainty with Poisson 
    statistics. Then sum the groups, scaling with the averaged scaling 
    ratio, and use Gaussian error propagation.
    
    Parameters
    ----------
    bkgpi_lst : list or numpy.ndarray
        Background PI file list.
    bkgscal_lst : list or numpy.ndrray
        Scaling ratio list.
    Ngrp : int, optional
        Number of groups with similar background-to-source scaling ratio. 
        Defaults to 10.
    fits_name : str, optional
        If specified, create a fits file with name `fits_name`. Defaults 
        to None.
    expo : float, optional
        Total exposure time (seconds) to be written in the header of 
        `fits_name`. Defaults to 10.
    
    Returns
    -------
    bkgpi : numpy.ndarray
        Stacked background PI array.
    bkgpi_err : numpy.ndarray
        Stacked background PI error array.
    """
    bkgpi_lst = np.array(bkgpi_lst)
    bkgscal_lst = np.array(bkgscal_lst)
    assert bkgpi_lst.shape[0] == bkgscal_lst.shape[0], "number of bkgpis and number of scaling ratios do not match!"
    
    # Stacked bkg spectral counts calculated as stacked src spectral counts
    bkgpi, bkgpi_err_UNUSED = add_pi(bkgpi_lst,bkgscal_lst)
    
    # Stacked bkg spectral counts uncertainties estimation: grouping method
    bkggrpflg_lst, bkgscal_ave_lst = make_bkggrpflg(bkgscal_lst,Ngrp=Ngrp) # group bkg spectra with similar scaling ratios
    bkgpi_grp_lst = []
    for i in range(Ngrp):
        bkgpi_tmp = bkgpi_lst[bkggrpflg_lst==i]
        bkgpi_grp_lst.append(add_pi(bkgpi_tmp)[0])
    bkgpi_grp_lst = np.array(bkgpi_grp_lst)
    # then sum the groups (scaling with average scaling ratio, and use Gaussian error propagation)
    bkgpi_UNUSED, bkgpi_err = add_pi(bkgpi_grp_lst,bkgscal_ave_lst)
    
    # write fits file (optional)
    if fits_name is not None:
        hdulist = fits.HDUList()
    
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        channels = np.arange(1,len(bkgpi)+1)
        cols = [fits.Column(name="CHANNEL", format="I", array=channels),
                fits.Column(name="COUNTS", format="D", array=bkgpi), # BKG counts: float
                fits.Column(name="STAT_ERR", format="D", array=bkgpi_err)]
        hdu_spectrum = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
        hdulist.append(hdu_spectrum)

        # PI header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, OGIP/92-007: "The OGIP Spectral File Format")
        hdu_spectrum.header["TELESCOP"] = "STACKED"
        hdu_spectrum.header["INSTRUME"] = "STACKED"
        hdu_spectrum.header["EXPOSURE"] = expo
        hdu_spectrum.header["BACKFILE"] = "None"
        hdu_spectrum.header["BACKSCAL"] = 1.0
        hdu_spectrum.header["CORRSCAL"] = 1.0
        hdu_spectrum.header["RESPFILE"] = "None"
        hdu_spectrum.header["ANCRFILE"] = "None"
        hdu_spectrum.header["AREASCAL"] = 1.0
        hdu_spectrum.header["HDUCLASS"] = "OGIP"
        hdu_spectrum.header["HDUCLAS1"] = "SPECTRUM"
        hdu_spectrum.header["HDUVERS"] = "1.2.1"
        hdu_spectrum.header["POISSERR"] = False # statistical errors specified in `STAT_ERR` instead
        hdu_spectrum.header["CHANTYPE"] = "PI"
        hdu_spectrum.header["DETCHANS"] = len(channels)
        hdu_spectrum.header["CREATOR"] = "XSTACK"
        hdu_spectrum.header["HDUCLAS2"] = "BKG"
        hdu_spectrum.header["HDUCLAS3"] = "COUNT"

        hdulist.writeto(f"{fits_name}", overwrite=True)
    
    return bkgpi,bkgpi_err


def get_bkgscal(src_file,bkg_file=None):
    """
    Get background-to-source scaling ratio, which is calculated as:

    .. math::
        :label: eq:bkgscal

        `scaling factor` = `AREASCAL`_src / `AREASCAL`_src * 
        `BACKSCAL`_src / `BACKSCAL`_bkg * `EXPOSURE`_src 
        / `EXPOSURE`_bkg
    
        
    Parameters
    ----------
    src_file : str
        Source PI spectrum name.
    bkg_file : str, optional
        Background PI spectrum name. If not specified, will look for it 
        from the header of src_file.
    
    Returns
    -------
    bkgscal : float
        Background-to-source scaling ratio.

    Notes
    -----
    Equation :eq:`bkgscal` applies to both point sources and extended 
    sources.

    For eROSITA, `EXPOSURE` is the total exposure time during which at 
    least one pixel of the extraction aperture is in the FoV. Since the
    FoV is scanning over the region during the exposure, `EXPOSURE` is 
    not the real averaged exposure time per pixel in the region. The 
    real averaged exposure time per pixel, after correcting for such 
    "region-covering incompleteness" issue, is actually:

    .. math::

        T_\mathrm{ave} \equiv `BACKSCL`/`REGAREA` * `EXPOSURE`

    where `REGAREA`/`BACKSCAL` is the region-covering-incompleteness-
    correcting factor.
    """
    with fits.open(src_file) as hdu:
        head = hdu["SPECTRUM"].header
    src_expo = head["EXPOSURE"]
    src_areascal = head["AREASCAL"]
    src_backscal = head["BACKSCAL"]

    if bkg_file is None:
        bkg_file = head["BACKFILE"]
    assert os.path.exists(bkg_file), "Background file does not exist!"
    with fits.open(bkg_file) as hdu:
        head = hdu["SPECTRUM"].header
    bkg_expo = head["EXPOSURE"]
    bkg_areascal = head["AREASCAL"]
    bkg_backscal = head["BACKSCAL"]
    bkgscal = src_areascal / bkg_areascal * src_backscal / bkg_backscal * src_expo / bkg_expo
    
    return bkgscal


def get_expo(src_file):
    """
    Get source exposure time.

    Parameters
    ----------
    src_file : str
        Source PI spectrum name.

    Returns
    -------
    src_expo : float
        Source exposure time.
    """
    src_expo = fits.getval(src_file,keyword="EXPOSURE",extname="SPECTRUM")
    if src_expo == 0:
        print(f"Please check {src_file}: why EXPOSURE == 0?")
    return src_expo


def get_rega(src_file):
    """
    Get source geometric area, from the non-standard keyword `REGAREA`.
    For non-eROSITA instrument, return 1.

    Parameters
    ----------
    src_file : str
        Source PI spectrum name.
    
    Returns
    -------
    src_rega : float
        Source region area [deg^2].
    """
    try:
        src_rega = fits.getval(src_file,keyword="REGAREA",extname="SPECTRUM")
    except Exception as e:
        src_rega = 1
    return src_rega


def make_bkggrpflg(bkgscal_lst,Ngrp=4):
    """
    Group the background spectra into `Ngrp` groups, according to the 
    bkg-to-source scaling ratios. 

    Return an array `bkggrpflg_lst` that tells you which group each 
    background PI spectrum should be assigned to.
    
    Parameters
    ----------
    bkgscal_lst : list or numpy.ndarray
        The list of bkg-to-source scaling ratio (considering both 
        `BACKSCAL` and `EXPOSURE`) for each background PI spectrum.
    Ngrp : int, optional
        The number of groups to be created. Defaults to 4.
    
    Returns
    -------
    bkggrpflg_lst : numpy.ndarray
        An array that indicates which group each background PI spectrum 
        should be assigned to (length = len(bkgscal_lst)).
    bkgscal_ave_lst : numpy.ndrray
        The average bkg-to-source scaling ratio of each group 
        (length = `Ngrp`).
    """
    idx_lst = np.argsort(bkgscal_lst)
    idx_lo = np.array([int(len(idx_lst) / Ngrp * i) for i in range(Ngrp)])
    idx_hi = np.array([int(len(idx_lst) / Ngrp * (i+1)) - 1 for i in range(Ngrp)])
    
    bkggrpflg_lst = np.zeros(len(idx_lst),dtype="int")
    for i in range(len(idx_lst)):
        idx = idx_lst[i]
        mask = (idx <= idx_hi) & (idx >= idx_lo)
        bkggrpflg = np.arange(Ngrp)[mask][0] # the group id of idx
        bkggrpflg_lst[i] = bkggrpflg
    
    bkgscal_ave_lst = np.ones(Ngrp)
    for i in range(Ngrp):
        bkgscal_ave_lst[i] = np.average(bkgscal_lst[bkggrpflg_lst==i])
    
    return bkggrpflg_lst, bkgscal_ave_lst