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
import shutil
from Xstack.utils.logger import utc_now_iso,add_run_cmd_history
from Xstack.config import VERSION,LASTUPDATE,WEB


def read_pi(
        pi_fname,
):
    """
    Read PI spectrum file.

    Parameters
    ----------
    pi_fname : str
        Observed-frame pi file to be shifted, in standard OGIP format.

    Returns
    -------
    pi_chan : list
        PI channel.
    pi_coun : list
        Photon counts in each channel.
    z : float
        Redshift if exists.
    """
    with fits.open(pi_fname) as hdu:
        pi = hdu["SPECTRUM"].data
        head = hdu["SPECTRUM"].header
    #--- read channel and counts
    pi_chan = pi["CHANNEL"]     # pi_chan starts from 0/1, depending on TLMIN1
    pi_coun = pi["COUNTS"]      # the obs-frame photon counts
    #--- read redshift if exists
    z = head.get("REDSHIFT",-999.0)

    return pi_chan,pi_coun,z


def shift_pi(
        pi_fname,z,
        ene_lo=None,ene_hi=None,ene_ce=None,ene_wd=None,rmf_fname=None,
        ene_trc=None,
):
    """
    Shift a single PI to rest-frame.

    Parameters
    ----------
    pi_fname : str
        Observed-frame pi file to be shifted, in standard OGIP format.
    z : float
        Redshift.
    ene_lo : numpy.ndarray, optional
        Lower edge of channel energy bin (keV).
    ene_hi : numpy.ndarray, optional
        Upper edge of channel energy bin (keV).
    ene_ce : numpy.ndarray, optional
        (``ene_lo`` + ``ene_hi``) / 2
    ene_wd : numpy.ndarray, optional
        (``ene_hi`` - ``ene_lo``)
    rmf_fname : str, optional
        RMF file defining channel-energy conversion, in standard OGIP 
        format. This is optional, unless ``ene_lo`` ``ene_hi`` are not
        specified.
    ene_trc : float, optional
        Truncate energy (keV) below which manually set ARF and PI counts 
        to zero. For eROSITA, ``ene_trc`` is typically 0.2 keV.
      
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
    #--- read rmf file for energy edges if they are not provided
    if (ene_lo is None) or (ene_hi is None):
        with fits.open(rmf_fname) as hdu:
            ebo = hdu["EBOUNDS"].data
        ene_lo = ebo["E_MIN"]
        ene_hi = ebo["E_MAX"]
        
    if (ene_ce is None) or (ene_wd is None):
        ene_ce = (ene_lo + ene_hi)/2
        ene_wd = ene_hi - ene_lo
    
    ene_id = np.arange(len(ene_ce))
    ene_ubound = ene_lo.max()
    ene_lbound = ene_hi.min()   # set lower and upper bound of energy to avoid overflow issues
    
    #--- read pi file
    # with fits.open(pi_fname) as hdu:
    #     pi = hdu["SPECTRUM"].data
    # pi_chan = pi["CHANNEL"]     # pi_chan starts from 0/1, depending on TLMIN1
    # pi_coun = pi["COUNTS"]      # the obs-frame photon counts
    pi_chan,pi_coun,_ = read_pi(pi_fname)
    chan_id = np.arange(len(pi_chan))

    #--- truncate below ene_trc
    if ene_trc is not None:
        idx_trc = np.argmin(abs(ene_ce-ene_trc))
        pi_coun[:idx_trc] = 0
    
    #--- rest-frame shifting pi counts
    rest_chan = pi_chan.copy()
    rest_coun = np.zeros_like(rest_chan,dtype=np.float32) # the src-frame photon counts
    
    for i in range(len(pi_chan)):
        ene_lo_map = ene_lo[i] * (1+z)
        ene_hi_map = ene_hi[i] * (1+z)

        #--- skip all rest-frame channels which are outside energy bounds
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
        
        #--- for the first and last channel in the basket, we need to recalculate their widths
        #--- this is because they are defined by chan_lo_map[i] and chan_hi_map[i], respectively
        ene_wd_mask[0] = ene_hi_mask[0] - ene_lo_map
        ene_wd_mask[-1] = ene_hi_map - ene_lo_mask[-1]
        
        #--- Each channel in the basket would get number of photons proportional to its width
        prob_mask = ene_wd_mask / ene_wd_mask.sum() # the probability of entering each channel in the basket
        # NOTE: we do not force each channel to have integer number of photon counts at this step
        # Instead, we round them to integers after stacking all sources (`add_pi`), where each channel accumulates sufficient photon counts
        # This would be helpful when each individual source has only few counts in total
        phoct_mask = np.asarray(pi_coun[i]*prob_mask,dtype=np.float32)  # this is float
        
        # Finally, assign the photons
        for idx in range(len(chan_id_mask)):
            try:
                rest_coun[chan_id_mask[idx]] += phoct_mask[idx]
            except IndexError:
                continue

    return rest_chan,rest_coun,pi_chan,pi_coun


def calc_pi_error(pi_stk,):
    """
    Round PI counts to integer (so that Poisson applies), and calculate
    PI uncertainty using Poisson formula.

    Parameters
    ----------
    pi_stk : numpy.ndarray
        Stacked PI array.
  
    Returns
    -------
    pi_stk : numpy.ndarray
        Stacked PI array. Rounded to nearest integer. 
        E.g., 0.4 -> 0, 0.6 -> 1
    pierr_stk : numpy.ndarray
        Stacked PI error array.
    """
    #--- For spectral counts
    # We round photon counts in each channel to nearest integer, to approximate Poisson
    # which is necessary to calculate uncertainties
    pi_stk = np.round(pi_stk).astype(int)
    
    #--- For spectral counts uncertainties
    pierr_stk = np.sqrt(pi_stk)

    return pi_stk,pierr_stk


def calc_bkgpi_error(bkgpi_lst,bkgscal_lst,Nbkggrp=10):
    """
    Calculate stacked bkg PI counts and uncertainties.

    Parameters
    ----------
    bkgpi_lst : numpy.ndarray or list
        List of bkg PI spectra.
    bkgscal_lst : numpy.ndarray or list
        List of bkg PI scaling factors.
    Nbkggrp : int, optional
       Number of background groups with similar ``bkgscal`` to be created. 
       Defaults to ``10``.

    Returns
    -------
    pi_stk : numpy.ndarray
        Stacked PI array.
    pierr_stk : numpy.ndarray
        Stacked PI error array.
    """
    bkgpi_lst = np.array(bkgpi_lst)
    bkgscal_lst = np.array(bkgscal_lst)
    assert bkgpi_lst.shape[0] == bkgscal_lst.shape[0], "number of bkgpis and number of scaling ratios do not match!"
    
    # Stacked bkg spectral counts calculated in the same way as stacked src spectral counts
    bkgpi_stk = np.sum(bkgpi_lst*bkgscal_lst[:,np.newaxis],axis=0)
    
    # Stacked bkg spectral counts uncertainties estimation: grouping method
    bkggrpflg_lst, bkgscal_ave_lst = make_bkggrpflg(bkgscal_lst,Nbkggrp=Nbkggrp) # group bkg spectra with similar scaling ratios
    bkgpi_grp_lst = []
    for i in range(Nbkggrp):
        bkgpi_tmp = bkgpi_lst[bkggrpflg_lst==i]
        bkgpi_grp_lst.append(np.round(np.sum(bkgpi_tmp,axis=0)).astype(int))    # simply add without any scaling ratio
    bkgpi_grp_lst = np.array(bkgpi_grp_lst)
    # then sum the groups (scaling with average scaling ratio, and use Gaussian error propagation)
    bkgpierr_grp_lst = np.sqrt(bkgpi_grp_lst)   # Poisson statistics
    bkgpierr_stk = np.sqrt(
        np.sum(
            (bkgpierr_grp_lst*bkgscal_ave_lst[:,np.newaxis])**2,
            axis=0,
        )
    )   # Gaussian error propagation

    return bkgpi_stk,bkgpierr_stk


def get_bkgscal(src_fname,bkg_fname=None):
    """
    Get background-to-source scaling ratio, which is calculated as:

    .. math::
       :label: eq:bkgscal
  
       \mathrm{Scaling\ Factor}
       = \\frac{\mathrm{AREASCAL}_{\mathrm{src}}}{\mathrm{AREASCAL}_{\mathrm{bkg}}}
       \\times \\frac{\mathrm{BACKSCAL}_{\mathrm{src}}}{\mathrm{BACKSCAL}_{\mathrm{bkg}}}
       \\times \\frac{\mathrm{EXPOSURE}_{\mathrm{src}}}{\mathrm{EXPOSURE}_{\mathrm{bkg}}}
    
    Parameters
    ----------
    src_fname : str
        Source PI spectrum name.
    bkg_fname : str, optional
        Background PI spectrum name. If not specified, will look for it 
        from the header of ``src_fname``.
    
    Returns
    -------
    bkgscal : float
        Background-to-source scaling ratio.

    Notes
    -----
    Equation :eq:`bkgscal` applies to both point sources and extended 
    sources.

    For eROSITA, ``EXPOSURE`` is the total exposure time during which at 
    least one pixel of the extraction aperture is in the FoV. Since the
    FoV is scanning over the region during the exposure, ``EXPOSURE`` is 
    not the real averaged exposure time per pixel in the region. The 
    real averaged exposure time per pixel, after correcting for such 
    "region-covering incompleteness" issue, is actually:

    .. math::

        T_\mathrm{ave} \equiv \\frac{\mathrm{BACKSCL}}{\mathrm{REGAREA}} \\times \mathrm{EXPOSURE}

    where ``REGAREA``/``BACKSCAL`` is the region-covering-incompleteness-
    correcting factor.

    Note that this is different from Eq. 10 of X. Zhang+2024: the bkg 
    spectra should not only be scaled by ``REGAREA``, but additionally by 
    the averaged exposure per pixel, which effectively results in an 
    exactly same scaling formula as for the point sources (``BACKSCAL`` 
    * ``EXPOSURE`` * ``AREASCAL``).
    """
    with fits.open(src_fname) as hdu:
        head = hdu["SPECTRUM"].header
    src_expo = head["EXPOSURE"]
    src_areascal = head["AREASCAL"]
    src_backscal = head["BACKSCAL"]

    if bkg_fname is None:
        bkg_fname = head["BACKFILE"]
    assert os.path.exists(bkg_fname), "Background file does not exist!"
    with fits.open(bkg_fname) as hdu:
        head = hdu["SPECTRUM"].header
    bkg_expo = head["EXPOSURE"]
    bkg_areascal = head["AREASCAL"]
    bkg_backscal = head["BACKSCAL"]
    bkgscal = src_areascal / bkg_areascal * src_backscal / bkg_backscal * src_expo / bkg_expo
    
    return bkgscal


def get_expo(src_fname):
    """
    Get source exposure time.

    Parameters
    ----------
    src_fname : str
        Source PI spectrum name.

    Returns
    -------
    src_expo : float
        Source exposure time.
    """
    src_expo = fits.getval(src_fname,keyword="EXPOSURE",extname="SPECTRUM")
    if src_expo == 0:
        print(f"Please check {src_fname}: why EXPOSURE == 0?")
    return src_expo


def get_rega(src_fname):
    """
    Get source geometric area, from the non-standard keyword ``REGAREA``.
    For non-eROSITA instrument, return 1.

    Parameters
    ----------
    src_fname : str
        Source PI spectrum name.

    Returns
    -------
    src_rega : float
        Source region area (:math:`\mathrm{deg}^2`).
    """
    try:
        src_rega = fits.getval(src_fname,keyword="REGAREA",extname="SPECTRUM")
    except Exception:
        src_rega = 1
    return src_rega


def get_areascal_backscal_corrscal(src_fname):
    """
    Read scaling keywords from ``src_fname`` header.

    Parameters
    ----------
    src_fname : str
        PI spectrum file name.

    Returns
    -------
    areascal : float
        Header keyword ``AREASCAL``. -999 if not found.
    backscal : float
        Header keyword ``BACKSCAL``. -999 if not found.
    corrscal : float
        Header keyword ``CORRSCAL``. -999 if not found.
    """
    with fits.open(src_fname) as hdu:
        hdr = hdu["SPECTRUM"].header
    areascal = float(hdr.get("AREASCAL",-999))
    backscal = float(hdr.get("BACKSCAL",-999))
    corrscal = float(hdr.get("CORRSCAL",-999))
    return areascal,backscal,corrscal


def make_bkggrpflg(bkgscal_lst,Nbkggrp=10):
    """
    Group the background spectra into ``Ngrp`` groups, according to the 
    bkg-to-source scaling ratios. 

    Return an array ``bkggrpflg_lst`` that tells you which group each 
    background PI spectrum should be assigned to.

    Parameters
    ----------
    bkgscal_lst : list or numpy.ndarray
        List of bkg-to-source scaling ratio (considering both 
        ``BACKSCAL`` and ``EXPOSURE``) for each background PI spectrum.
    Nbkggrp : int, optional
        Number of background groups with similar ``bkgscal`` to be created. 
        Defaults to ``10``.

    Returns
    -------
    bkggrpflg_lst : numpy.ndarray
        An array that indicates which group each background PI spectrum 
        should be assigned to (length = len(``bkgscal_lst``)).
    bkgscal_ave_lst : numpy.ndrray
        The average bkg-to-source scaling ratio of each group 
        (length = ``Ngrp``).
    """
    #--- array-lize
    bkgscal_lst = np.asarray(bkgscal_lst,dtype=float)
    N = bkgscal_lst.size
    if N == 0:
        return np.array([],dtype=int),np.array([],dtype=float)
    Nbkggrp_eff = min(int(Nbkggrp),N)  # cannot have more groups than points

    #--- sort indices by bkgscal
    order = np.argsort(bkgscal_lst)

    #--- assign group IDs in sorted order: 0..Nbkggrp_eff-1, roughly equal counts
    grp_sorted = (np.arange(N) * Nbkggrp_eff) // N  # length N, values 0..Nbkggrp_eff-1

    #--- map back to original order
    bkggrpflg_lst = np.empty(N,dtype=int)
    bkggrpflg_lst[order] = grp_sorted

    #--- group means
    bkgscal_ave_lst = np.full(Nbkggrp_eff, np.nan, dtype=float)
    for g in range(Nbkggrp_eff):
        bkgscal_ave_lst[g] = bkgscal_lst[bkggrpflg_lst == g].mean()
    
    return bkggrpflg_lst, bkgscal_ave_lst


def write_pi(
        chan,pi,pierr=None,pi_fname="stacked_pi.fits",
        expo=10,rega=1,bkgpi_fname=None,rmf_fname=None,arf_fname=None,
        spec_type="STACKED",z=None,
        areascal=1.0,backscal=1.0,corrscal=1.0,run_cmd=None,
):
    """
    Write PI spectrum file according to OGIP standards.
    Assume all spectral files (PI, ARF, RMF) under the same path for ``XSPEC`` convenience.

    Parameters
    ----------
    chan : numpy.ndarray
        Stacked src spectrum channel.
    pi : numpy.ndarray
        Stacked src spectrum counts.
    pierr : numpy.ndarray, optional
        Stacked src spectrum uncertainty. Defaults to ``None`` (use ``XSPEC`` 
        ``POISSERR`` by default).
    pi_fname : str, optional
        Output src spectrum name. Defaults to ``stacked_srcpi.fits``.
    expo : int or float, optional
        Stacked exposure. Defaults to ``10``.
    rega : int or float, optional
        Stacked region area. Defaults to ``1``.
    bkgpi_fname : str, optional
        Stacked bkg PI filename. Defaults to ``None``.
    rmf_fname : str, optional
        Stacked RMF filename. Defaults to ``None``.
    arf_fname : str, optional
        Stacked ARF filename. Defaults to ``None``.
    spec_type : str, optional
        ``STACKED`` for stacked spectrum, or ``RESTFRAM`` for individual 
        rest-frame spectrum. Counts are stored as integer in ``STACKED`` 
        mode, while float in ``RESTFRAM`` mode. 
    z : float, optional
        Redshift.
    areascal : float, optional
        Header keyword ``AREASCAL``. Defaults to ``1.0``.
    backscal : float, optional
        Header keyword ``BACKSCAL``. Defaults to ``1.0``.
    corrscal : float, optional
        Header keyword ``CORRSCAL``. Defaults to ``1.0``.
    run_cmd : str, optional
        Full command string recorded in ``HISTORY`` for provenance.

    Returns
    -------
    None
    """
    hdu_lst = fits.HDUList()
    
    primary_hdu = fits.PrimaryHDU()
    hdu_lst.append(primary_hdu)
    
    if pierr is not None:
        cols = [
            fits.Column(name="CHANNEL",format="I",array=chan),
            fits.Column(name="COUNTS",format="J" if spec_type=="STACKED" else "D",array=pi),
            fits.Column(name="STAT_ERR", format="D", array=pierr),
        ]
    else:
        cols = [
            fits.Column(name="CHANNEL",format="I",array=chan),
            fits.Column(name="COUNTS",format="J" if spec_type=="STACKED" else "D",array=pi),
        ]
    hdu_spectrum = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
    hdu_lst.append(hdu_spectrum)

    # PI header following OGIP standards
    # https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, OGIP/92-007: "The OGIP Spectral File Format"
    hdu_spectrum.header["TELESCOP"] = spec_type
    hdu_spectrum.header["INSTRUME"] = spec_type
    hdu_spectrum.header["EXPOSURE"] = (expo, f"stacked exposure time [s]")
    hdu_spectrum.header["REGAREA"] = (rega, f"stacked region area [deg^2]")
    if z is not None:
        hdu_spectrum.header["REDSHIFT"] = z
    if bkgpi_fname is not None:
        # we assume all files under the same path for xspec convenience
        hdu_spectrum.header["BACKFILE"] = (os.path.basename(bkgpi_fname), f"associated background PI spectrum")   
    if rmf_fname is not None:
        hdu_spectrum.header["RESPFILE"] = (os.path.basename(rmf_fname), f"associated response matrix file")
    if arf_fname is not None:
        hdu_spectrum.header["ANCRFILE"] = (os.path.basename(arf_fname), f"associated ancillary response file")
    hdu_spectrum.header["AREASCAL"] = areascal
    hdu_spectrum.header["BACKSCAL"] = backscal
    hdu_spectrum.header["CORRSCAL"] = corrscal
    hdu_spectrum.header["HDUCLASS"] = "OGIP"
    hdu_spectrum.header["HDUCLAS1"] = "SPECTRUM"
    hdu_spectrum.header["HDUVERS"] = "1.2.1"
    hdu_spectrum.header["POISSERR"] = False     # statistical errors specified in `STAT_ERR` instead
    hdu_spectrum.header["CHANTYPE"] = "PI"
    hdu_spectrum.header["DETCHANS"] = len(chan)
    hdu_spectrum.header["TLMIN1"] = np.min(chan)
    hdu_spectrum.header["CREATOR"] = "XSTACK"
    hdu_spectrum.header["HDUCLAS2"] = "TOTAL"
    hdu_spectrum.header["HDUCLAS3"] = "COUNT"
    hdu_spectrum.header["HISTORY"] = f"{utc_now_iso()}: stacked source PI spectrum created by Xstack v{VERSION} [{LASTUPDATE}] [{WEB}]"
    add_run_cmd_history(hdu_spectrum.header,run_cmd)
    
    hdu_lst.writeto(f"{pi_fname}",overwrite=True)

    return


def write_bkgpi(
        chan,bkgpi,bkgpierr,bkgpi_fname="stacked_bkgpi.fits",
        expo=10,rega=1,
        spec_type="STACKED",z=None,
        areascal=1.0,backscal=1.0,corrscal=1.0,run_cmd=None,
):
    """
    Write bkg PI spectrum file according to OGIP standards.
    Assume all spectral files (PI, ARF, RMF) under the same path for 
    ``XSPEC`` convenience.

    Parameters
    ----------
    chan : numpy.ndarray
        Stacked bkg spectrum channel.
    bkgpi : numpy.ndarray
        Stacked bkg spectrum counts.
    bkgpierr : numpy.ndarray
        Stacked bkg spectrum uncertainty.
    bkgpi_fname : str, optional
        Output bkg spectrum name. Defaults to ``stacked_bkgpi.fits``.
    expo : int or float, optional
        Stacked exposure. Defaults to ``10``.
    rega : int or float, optional
        Stacked region area. Defaults to ``1``.
    spec_type : str, optional
        ``STACKED`` for stacked spectrum, or ``RESTFRAM`` for individual rest-frame spectrum.
        Counts are stored as float regardless.
    z : float, optional
        Redshift.
    areascal : float, optional
        Header keyword ``AREASCAL``. Defaults to ``1.0``.
    backscal : float, optional
        Header keyword ``BACKSCAL``. Defaults to ``1.0``.
    corrscal : float, optional
        Header keyword ``CORRSCAL``. Defaults to ``1.0``.
    run_cmd : str, optional
        Full command string recorded in ``HISTORY`` for provenance.

    Returns
    -------
    None
    """
    hdu_lst = fits.HDUList()
    
    primary_hdu = fits.PrimaryHDU()
    hdu_lst.append(primary_hdu)
    
    counts_format = "J" if np.issubdtype(np.asarray(bkgpi).dtype,np.integer) else "D"
    cols = [
        fits.Column(name="CHANNEL",format="I",array=chan),
        fits.Column(name="COUNTS",format=counts_format,array=bkgpi),
        fits.Column(name="STAT_ERR",format="D",array=bkgpierr),
    ]
    hdu_spectrum = fits.BinTableHDU.from_columns(cols, name="SPECTRUM")
    hdu_lst.append(hdu_spectrum)

    # PI header following OGIP standards 
    # https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, OGIP/92-007: "The OGIP Spectral File Format"
    hdu_spectrum.header["TELESCOP"] = spec_type
    hdu_spectrum.header["INSTRUME"] = spec_type
    hdu_spectrum.header["EXPOSURE"] = (expo, f"stacked exposure time [s]")
    hdu_spectrum.header["REGAREA"] = (rega, f"stacked region area [deg^2]")
    if z is not None:
        hdu_spectrum.header["REDSHIFT"] = z
    hdu_spectrum.header["BACKFILE"] = "None"
    hdu_spectrum.header["RESPFILE"] = "None"
    hdu_spectrum.header["ANCRFILE"] = "None"
    hdu_spectrum.header["AREASCAL"] = areascal
    hdu_spectrum.header["BACKSCAL"] = backscal
    hdu_spectrum.header["CORRSCAL"] = corrscal
    hdu_spectrum.header["HDUCLASS"] = "OGIP"
    hdu_spectrum.header["HDUCLAS1"] = "SPECTRUM"
    hdu_spectrum.header["HDUVERS"] = "1.2.1"
    hdu_spectrum.header["POISSERR"] = False     # statistical errors specified in `STAT_ERR` instead
    hdu_spectrum.header["CHANTYPE"] = "PI"
    hdu_spectrum.header["DETCHANS"] = len(chan)
    hdu_spectrum.header["TLMIN1"] = np.min(chan)
    hdu_spectrum.header["CREATOR"] = "XSTACK"
    hdu_spectrum.header["HDUCLAS2"] = "BKG"
    hdu_spectrum.header["HDUCLAS3"] = "COUNT"
    hdu_spectrum.header["HISTORY"] = f"{utc_now_iso()}: stacked background PI spectrum created by Xstack v{VERSION} [{LASTUPDATE}] [{WEB}]"
    add_run_cmd_history(hdu_spectrum.header,run_cmd)

    hdu_lst.writeto(f"{bkgpi_fname}",overwrite=True)

    return



#--- below for visualization purposes
def make_grpflg(src_fname,grp_fname=None,method="EDGE",rmf_fname="",eelo=None,eehi=None,bkg_fname=None,min_net=0):
    """
    Add ``GROUPING`` column to the source PI file.
    
    Parameters
    ----------
    src_fname : str
        Input source PI file name.
    grp_fname : str, optional
        Output grouped PI file name. If not specified, will not create 
        output file.
    method : str, optional
        Grouping method. Available methods:

        - ``EDGE``: Group by fixed energy bin edges. Edges provided by 
          ``eelo`` and ``eehi``.
        - ``MIN_NET``: Group by minimum net counts ``(src-bkg*bkgscal)``. 
           Needs to specify the ``bkg_fname`` and ``min_net`` in each group.

    rmf_fname : str, optional
        (for ``EDGE`` method) RMF file name. If not specified, the code 
        will automatically search the header of ``src_fname``.
    eelo : numpy.ndarray, optional
        (for ``EDGE`` method) Lower edge of fixed energy bin.
    eehi : numpy.ndarray, optional
        (for ``EDGE`` method) Upper edge of fixed energy bin.
    bkg_fname : str, optional
        Background file name used in ``MIN_NET`` mode. Defaults to ``None``. 
        If not specified, will look for it in the header of ``src_fname``.
    min_net : float or int, optional
        Minimum net counts in each group in ``MIN_NET`` mode. Defaults to ``0``.
    
    Returns
    -------
    grpflg : numpy.ndarray
        ``GROUPING`` column written in ``grp_fname``.
    """
    if method == "EDGE":
        if (eelo is None) or (eehi is None):
            raise Exception("Please specify `eelo` and `eehi` in method `EDGE`!")
        # find channel energy in EBOUNDS extension of RMF file
        with fits.open(src_fname) as hdu:
            data = hdu["SPECTRUM"].data
            head = hdu["SPECTRUM"].header
            chan = data["CHANNEL"]
            try:
                src_rmf = head["RESPFILE"]
            except Exception:
                src_rmf = ""
        
        # the RMF file must either be specified as `rmf_fname`, or specified in the header of `src_fname`
        if os.path.exists(rmf_fname):
            pass
        elif os.path.exists(src_rmf):
            rmf_fname = src_rmf
        else:
            raise Exception(f"Either the RMF file is not specified as `rmf_fname`, or the one in {src_fname} does not exist!")
        
        with fits.open(rmf_fname) as hdu:
            ebo = hdu["EBOUNDS"].data
        ene_lo = ebo["E_MIN"]
        ene_hi = ebo["E_MAX"]
        ene_ce = (ene_lo + ene_hi) / 2
        
        assert len(chan)==len(ene_ce), "CHANNEL and RMF EBOUNDS ENERGY does not match!"
        assert np.all(eelo<eehi)==True, "`eelo` has to be smaller than `eehi`!"
        
        # make grouping flag
        grpflg = np.ones(len(ene_ce))
        eece = (eelo + eehi) / 2
        eeid = np.arange(len(eece))
        eeid_bk = [-1] # stores the energy id that has been used
        for i in range(len(ene_ce)):
            mask = (ene_ce[i]<=eehi) & (ene_ce[i]>eelo)
            if np.all(mask==False): # outside eelo~eehi
                grpflg[i] = 1
                continue
            if eeid[mask] > max(eeid_bk): # step to a new bin
                grpflg[i] = 1
            else:
                grpflg[i] = -1
            eeid_bk.append(eeid[mask])
        
        # create output file
        if grp_fname is not None:
            shutil.copy(src_fname,grp_fname)
            with fits.open(grp_fname,mode="update") as hdu:
                SPECTRUM = hdu[1]
                if "GROUPING" in SPECTRUM.columns.names:
                    SPECTRUM.columns.del_col("GROUPING")    # remove "GROUPING" column if it exists beforehand
                GROUPING = fits.Column(name="GROUPING", format="I", array=grpflg)
                SPECTRUM.data = fits.BinTableHDU.from_columns(SPECTRUM.columns + GROUPING).data
        
        return grpflg
    
    elif method == "MIN_NET":
        with fits.open(src_fname) as hdu:
            data = hdu["SPECTRUM"].data
            src_chan = data["CHANNEL"]
            src_coun = data["COUNTS"]
            head = hdu["SPECTRUM"].header
        if bkg_fname is None:
            bkg_fname = head["BACKFILE"]
        assert os.path.exists(bkg_fname), "Background file does not exist!"
        with fits.open(bkg_fname) as hdu:
            data = hdu["SPECTRUM"].data
            bkg_chan = data["CHANNEL"]
            bkg_coun = data["COUNTS"]
        assert len(src_chan) == len(bkg_chan), f"src channel ({len(src_chan)}) and bkg channel ({len(bkg_chan)}) do not match!"
        bkgscal = get_bkgscal(src_fname,bkg_fname)

        # make grouping flag
        grpflg = np.ones(len(src_chan))
        net_cts = 0
        for i in range(len(src_chan)):
            net_cts += src_coun[i] - bkg_coun[i]*bkgscal
            if net_cts > min_net:
                grpflg[i] = 1   # 1 for end of the group
                net_cts = 0
            else:
                grpflg[i] = -1  # -1 for continuing the group

        # create output file
        if grp_fname is not None:
            shutil.copy(src_fname,grp_fname)
            with fits.open(grp_fname,mode="update") as hdu:
                SPECTRUM = hdu[1]
                if "GROUPING" in SPECTRUM.columns.names:
                    SPECTRUM.columns.del_col("GROUPING")    # remove "GROUPING" column if it exists beforehand
                GROUPING = fits.Column(name="GROUPING", format="I", array=grpflg)
                SPECTRUM.data = fits.BinTableHDU.from_columns(SPECTRUM.columns + GROUPING).data

        return grpflg
    
    else:
        raise Exception("Available method for grppi: EDGE, MIN_NET!")


def rebin_pi(ene_lo,ene_hi,coun,coun_err,grpflg):
    """
    Rebin PI file according to ``grpflg``.

    Parameters
    ----------
    ene_lo : numpy.ndarray
        Lower edge of channel energy bin.
    ene_hi : numpy.ndarray
        Upper edge of channel energy bin.
    coun : numpy.ndarray
        Photon counts in each channel.
    coun_err : numpy.ndarray
        Photon counts error in each channel.
    grpflg : numpy.ndarray
        Grouping flag. Must have same length as ``ene_lo`` or ``ene_hi``.

    Returns
    -------
    grpene_lo : numpy.ndarray
        Lower edge of grouped energy bin.
    grpene_hi : numpy.ndarray
        Upper edge of grouped energy bin.
    grpcoun : numpy.ndarray
        Photon counts in each grouped energy bin.
    grpcoun_err : numpy.ndarray
        Photon counts error in each grouped energy bin.
    """
    ene_ce = (ene_lo + ene_hi) / 2
    assert len(grpflg) == len(ene_ce), "grpflag shape "+str(grpflg.shape)+" does not match ene shape "+str(ene_ce.shape)+" !"
    
    grpene_lo = []
    grpene_hi = []
    grpcoun = []
    grpcoun_err = []
    
    tmpene_lo = []
    tmpene_hi = []
    tmpcoun = []
    tmpcoun_err = []
    
    for i in range(len(ene_ce)):
        if grpflg[i] == 1:    # start of group
            # collect data
            # if ene_tmp_lst is empty (usually the case for the first energy bin), just skip this step
            if len(tmpene_lo)!=0:
                grpene_lo.append(tmpene_lo[0])
                grpene_hi.append(tmpene_hi[-1])
                grpcoun.append(np.sum(tmpcoun))
                grpcoun_err.append(np.sqrt(np.sum(np.array(tmpcoun_err)**2)))
            tmpene_lo = [ene_lo[i]]
            tmpene_hi = [ene_hi[i]]
            tmpcoun = [coun[i]]
            tmpcoun_err = [coun_err[i]]
        elif grpflg[i] == -1:    # continuing of group
            tmpene_lo.append(ene_lo[i])
            tmpene_hi.append(ene_hi[i])
            tmpcoun.append(coun[i])
            tmpcoun_err.append(coun_err[i])
        else: 
            raise Exception("`grpflg` not in standard format (`1` for start of group, `-1` for continuing of group)")
    
    # for the last energy bin
    grpene_lo.append(tmpene_lo[0])
    grpene_hi.append(tmpene_hi[-1])
    grpcoun.append(np.sum(tmpcoun))
    grpcoun_err.append(np.sqrt(np.sum(np.array(tmpcoun_err)**2)))
    
    grpene_lo = np.array(grpene_lo)
    grpene_hi = np.array(grpene_hi)
    grpcoun = np.array(grpcoun)
    grpcoun_err = np.array(grpcoun_err)
        
    return grpene_lo,grpene_hi,grpcoun,grpcoun_err
