import numpy as np
from astropy.io import fits
import os
import re
import shutil
from scipy.constants import c
import astropy.units as u
from astropy.cosmology import Planck18
import sfdmap
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel,delayed
from tqdm import tqdm
from Xstack.shift_rsp import get_prob,get_prob1d
from Xstack.shift_pi import get_bkgscal,get_expo


#===================================================
##################### grppi ########################
#===================================================
def make_grpflg(src_name,grp_name=None,method="EDGE",rmf_file="",eelo=None,eehi=None,bkg_name=None,min_net=0):
    """
    Add `GROUPING` column to the source PI file.
    
    Parameters
    ----------
    src_name : str
        Input source PI file name.
    grp_name : str, optional
        Output grouped PI file name. If not specified, will not create 
        output file.
    method : str, optional
        Grouping method. Available methods:
        - `EDGE`: Group by fixed energy bin edges. Edges provided by 
          `eelo` and `eehi`.
        - `MIN_NET`: Group by minimum net counts (src-bkg*bkgscal). 
           Needs to specify the bkg_name and min_net in each group.
    rmf_file : str, optional
        (for `EDGE` method) RMF file name. If not specified, the code 
        will automatically search the header of `src_name`.
    eelo : numpy.ndarray, optional
        (for `EDGE` method) Lower edge of fixed energy bin.
    eehi : numpy.ndarray, optional
        (for `EDGE` method) Upper edge of fixed energy bin.
    bkg_name : str, optional
        Background file name used in `MIN_NET` mode. Defaults to None. 
        If not specified, will look for it in the header of `src_name`.
    min_net : float or int, optional
        Minimum net counts in each group in `MIN_NET` mode. Defaults to 0.
    
    Returns
    -------
    grpflg : numpy.ndarray
        `GROUPING` column written in `grp_name`.

    Available Methods
    -----------------
    * `EDGE`: Group by fixed energy bin edges.
    * `MIN_NET`: Group by minimum net counts in each group.
    """
    if method == "EDGE":
        if (eelo is None) or (eehi is None):
            raise Exception("Please specify `eelo` and `eehi` in method `EDGE`!")
        # find channel energy in EBOUNDS extension of RMF file
        with fits.open(src_name) as hdu:
            data = hdu["SPECTRUM"].data
            head = hdu["SPECTRUM"].header
            chan = data["CHANNEL"]
            try:
                src_rmf = head["RESPFILE"]
            except Exception as e:
                src_rmf = ""
        
        # the RMF file must either be specified as `rmf_file`, or specified in the header of `src_name`
        if os.path.exists(rmf_file):
            pass
        elif os.path.exists(src_rmf):
            rmf_file = src_rmf
        else:
            raise Exception("Either the RMF file is not specified as `rmf_file`, or the one in %s does not exist!"%src_name)
        
        with fits.open(rmf_file) as hdu:
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
        if grp_name is not None:
            shutil.copy(src_name,grp_name)
            with fits.open(grp_name,mode="update") as hdu:
                SPECTRUM = hdu[1]
                if "GROUPING" in SPECTRUM.columns.names:
                    SPECTRUM.columns.del_col("GROUPING")    # remove "GROUPING" column if it exists beforehand
                GROUPING = fits.Column(name="GROUPING", format="I", array=grpflg)
                SPECTRUM.data = fits.BinTableHDU.from_columns(SPECTRUM.columns + GROUPING).data
        
        return grpflg
    
    elif method == "MIN_NET":
        with fits.open(src_name) as hdu:
            data = hdu["SPECTRUM"].data
            src_chan = data["CHANNEL"]
            src_coun = data["COUNTS"]
            head = hdu["SPECTRUM"].header
        if bkg_name is None:
            bkg_name = head["BACKFILE"]
        assert os.path.exists(bkg_name), "Background file does not exist!"
        with fits.open(bkg_name) as hdu:
            data = hdu["SPECTRUM"].data
            bkg_chan = data["CHANNEL"]
            bkg_coun = data["COUNTS"]
        assert len(src_chan) == len(bkg_chan), f"src channel ({len(src_chan)}) and bkg channel ({len(bkg_chan)}) do not match!"
        bkgscal = get_bkgscal(src_name,bkg_name)

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
        if grp_name is not None:
            shutil.copy(src_name,grp_name)
            with fits.open(grp_name,mode="update") as hdu:
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
    Rebin PI file according to `grpflg`.
    
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
        Grouping flag. Must have same length as `ene_lo` or `ene_hi`.
    
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


def make_dataarf_plot(src_name,bkg_name=None,arf_name=None,rmf_name=None,grp_name=None,normalize_at=None,outname=None,plot=False,ax=None,**kwargs):
    """
    Make data/arf plot to visualize the stacked spectral shape.

    Parameters
    ----------
    src_name : str
        Source spectrum file name.
    bkg_name : str, optional
        Background spectrum file name. If not specified, will look for 
        it in the header of `src_name`.
    arf_name : str, optional
        ARF file name. If not specified, will look for it in the header 
        of `src_name`.
    rmf_name : str, optional
        RMF file name. If not specified, will look for it in the header 
        of `src_name`.
    grp_name : str, optional
        Grouping file name. Only uses its "GROUPING" column.
    normalize_at : int or float, optional
        Output spectrum normalized at some energy (keV). Defaults to None.
    outname : str, optional
        Output file name. If not specified, will not create output file 
        name.
    plot : bool, optional
        Whether or not to make a plot. Defaults to False.
    ax : matplotlib.axes.Axes, optional
        The axes to make the plot. Defaults to None.
    **kwargs

    Returns
    -------
    grpene_lo : numpy.ndarray
        Lower bounds of grouped energy.
    grpene_hi : numpy.ndarray
        Upper bounds of grouped energy.
    ratio : numpy.ndarray
        Data/arf ratio, in units of [ct/cm^2/s/keV].
    ratioerr : numpy.ndarray
        Uncertainty in data/arf ratio.

    """
    with fits.open(src_name) as hdu:
        data = hdu["SPECTRUM"].data
        src_chan = data["CHANNEL"]
        src_coun = data["COUNTS"]
        src_counerr = np.sqrt(src_coun)
        head = hdu["SPECTRUM"].data
    expo = get_expo(src_name)
    
    if bkg_name is None:
        bkg_name = head["BACKFILE"]
    assert os.path.exists(bkg_name), "Background file does not exist!"
    with fits.open(bkg_name) as hdu:
        data = hdu["SPECTRUM"].data
        bkg_chan = data["CHANNEL"]
        bkg_coun = data["COUNTS"]
        bkg_counerr = np.sqrt(bkg_coun)
    bkgscal = get_bkgscal(src_name,bkg_name)

    if arf_name is None:
        arf_name = head["ANCRFILE"]
    assert os.path.exists(arf_name), "ARF file does not exist!"
    with fits.open(arf_name) as hdu:
        arf = hdu["SPECRESP"].data
    arfene_lo = arf["ENERG_LO"]
    arfene_hi = arf["ENERG_HI"]
    arfene_ce = (arfene_hi + arfene_lo)/2
    arfene_wd = arfene_hi - arfene_lo
    specresp = arf["SPECRESP"]

    if rmf_name is None:
        rmf_name = head["RESPFILE"]
    assert os.path.exists(rmf_name), "RMF file does not exist!"
    with fits.open(rmf_name) as hdu:
        mat = hdu["MATRIX"].data
        ebo = hdu["EBOUNDS"].data
    ene_lo = ebo["E_MIN"]
    ene_hi = ebo["E_MAX"]
    ene_ce = (ene_lo + ene_hi)/2
    ene_wd = ene_hi - ene_lo

    if grp_name is not None:
        with fits.open(grp_name) as hdu:
            data = hdu["SPECTRUM"].data
        grpflg = data["GROUPING"]
        assert len(grpflg) == len(src_chan), f"Channel number ({len(src_chan)}) and Grouping flag length ({len(grpflg)}) do not match!"
    else:
        grpflg = np.ones(len(ene_ce))

    grpene_lo,grpene_hi,grpsrc_coun,grpsrc_counerr = rebin_pi(ene_lo,ene_hi,src_coun,src_counerr,grpflg)
    grpene_lo,grpene_hi,grpbkg_coun,grpbkg_counerr = rebin_pi(ene_lo,ene_hi,bkg_coun,bkg_counerr,grpflg)
    grpene_wd = grpene_hi - grpene_lo
    grpene_ce = (grpene_lo + grpene_hi) / 2
    grpene_lo,grpene_hi,grpspecresp = rebin_arf(arfene_lo,arfene_hi,specresp,ene_lo,ene_hi,src_coun-bkg_coun*bkgscal,grpflg)

    subtract = grpsrc_coun - grpbkg_coun*bkgscal
    subtracterr = np.sqrt(grpsrc_counerr**2 + grpbkg_counerr**2*bkgscal**2)
    ratio = subtract / grpspecresp / grpene_wd / expo
    ratioerr = subtracterr / grpspecresp / grpene_wd / expo

    ene_ce_norm = 1
    if normalize_at is not None:
        if grp_name is not None:
            normalize_idx = np.argmin(abs(grpene_ce-normalize_at))
            ene_ce_norm = grpene_ce[normalize_idx]
            factor_norm = ratio[normalize_idx]
        else:   # take average
            normalize_idxs = np.argsort(abs(grpene_ce-normalize_at))[:50]   # TODO: better number than 50?
            ene_ce_norm = np.nanmedian(grpene_ce[normalize_idxs])
            factor_norm = np.nanmedian(ratio[normalize_idxs])
        ratio = ratio / factor_norm
        ratioerr = ratioerr / factor_norm

    if outname is not None:
        hdu_lst = fits.HDUList()
        hdu_primary = fits.PrimaryHDU()
        hdu_lst.append(hdu_primary)

        arrays = [grpene_lo,grpene_hi,ratio,ratioerr]
        colnames = ["GRPE_MIN","GRPE_MAX","RATIO","RATIO_ERR"]
        formats = ["D","D","D","D"]
        units = ["keV","keV","cts/s/cm^2/keV","cts/s/cm^2/keV"]

        columns = [fits.Column(name=colname_,array=array_,format=format_,unit=unit_) for colname_,array_,format_,unit_ in zip(colnames,arrays,formats,units)]
        hdu_data = fits.BinTableHDU.from_columns(columns,name="DATAARF")
        hdu_lst.append(hdu_data)

        hdu_lst.writeto(outname,overwrite=True)

    if plot:
        if ax is None:
            ax = plt.gca()
        ax.errorbar(grpene_ce,ratio*grpene_ce**2/ene_ce_norm**2,yerr=(ratioerr*grpene_ce**2/ene_ce_norm**2/2),**kwargs)

    return ax


#===================================================
################# first energy #####################
#===================================================
def fene_fits(srcid_lst,arffene_lst,fene_lst,fits_name):
    """
    Creating a fits storing the first energy of each source"s PI spectrum 
    and ARF specresp.

    Parameters
    ----------
    srcid_lst : list or numpy.ndarray
        The source ID list.
    arffene_lst : list or numpy.ndarray
        The first energy of each sources"s ARF specresp.
    fene_lst : list or numpy.ndarray
        The first energy of each source"s PI spectrum.
    fits_name : str
        The output fits name.

    Returns
    -------
    None

    """
    if fits_name is not None:
        hdu_lst = fits.HDUList()
        
        primary_hdu = fits.PrimaryHDU()
        hdu_lst.append(primary_hdu)
        
        cols = [fits.Column(name="srcid",format="I",array=srcid_lst),
                fits.Column(name="arffene",format="D",array=arffene_lst,unit="keV"),
                fits.Column(name="fene",format="D",array=fene_lst,unit="keV")]
        hdu_fene = fits.BinTableHDU.from_columns(cols, name="FENERGY")
        hdu_lst.append(hdu_fene)

        hdu_lst.writeto("%s"%(fits_name), overwrite=True)

    return


def valid_energy_range_plot(fene_name,src_name,grp_name,bkg_name,rmf_name,ax=None):
    """
    Plot:

    1) fraction of sources contributing, and
    2) fraction of net counts (total-background), 

    as a function of rest-frame energy. These two would facilitate 
    determining a valid energy range for stacked spectrum analysis. 
    Neither the fraction of sources contributing, nor the fraction 
    of net counts can be too low.

    Parameters
    ----------
    fene_name : str
        The name of the fits file containing first contributing energy. 
        Output of Xstack.
    src_name : str
        The source spectrum file name.
    grp_name : str
        The grouped source spectrum file to be created.
    bkg_name : str
        The background spectrum file name.
    rmf_name : str
        The RMF file name.
    ax : matplotlib.axes.Axes, optional
        The axes to make the plot. If not specified, will use the current 
        axes. Defaults to None.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    ax_twinx : matplotlib.axes.Axes
        The twin axes for background fraction plot.
    
    """
    if ax is None:
        ax = plt.gca()
    color_left = "red"
    color_right = "blue"

    # first energy
    with fits.open(fene_name) as hdu:
        data = hdu["FENERGY"].data
    arffene = data["arffene"]
    fene = data["fene"]

    sorted_arffene = np.sort(arffene)
    cdf_arffene = np.arange(1,len(sorted_arffene)+1)/len(sorted_arffene)
    extendene = np.logspace(-1,1,1000)
    extendcdf_arffene = np.interp(extendene,sorted_arffene,cdf_arffene,left=0,right=1)

    sorted_fene = np.sort(fene)
    cdf_fene = np.arange(1,len(sorted_fene)+1)/len(sorted_fene)
    extendene = np.logspace(-1,1,1000)
    extendcdf_fene = np.interp(extendene,sorted_fene,cdf_fene,left=0,right=1)

    ax.plot(extendene,extendcdf_arffene,ls="-",c=color_left,label="ARF")
    ax.plot(extendene,extendcdf_fene,ls="--",c=color_left,label="PHA")
    ax.legend(fontsize=10)

    ax.set_xlabel("Energy (keV)",fontsize=10)
    ax.tick_params("x",which="major",length=10,width=1.0,size=5,labelsize=8,pad=3)
    ax.tick_params("x",which="minor",length=10,width=1.0,size=5,labelsize=8,pad=3)
    ax.set_ylim(0,1.1)
    ax.set_ylabel("Frac sources",fontsize=10,color=color_left)
    ax.tick_params("y",which="major",length=10,width=1.0,size=5,labelsize=8,pad=3,labelcolor=color_left)
    ax.tick_params("y",which="minor",length=10,width=1.0,size=5,labelsize=8,pad=3,labelcolor=color_left)


    # net fraction
    with fits.open(src_name) as hdu:
        data = hdu["SPECTRUM"].data
    chan = data["CHANNEL"]
    pha = data["COUNTS"]
    phaerr = np.sqrt(pha)
    with fits.open(bkg_name) as hdu:
        data = hdu["SPECTRUM"].data
    chan = data["CHANNEL"]
    bkgpha = data["COUNTS"]
    bkgphaerr = np.sqrt(pha)
    with fits.open(rmf_name) as hdu:
        mat = hdu["MATRIX"].data
        ebo = hdu["EBOUNDS"].data
    ene_lo = ebo["E_MIN"]
    ene_hi = ebo["E_MAX"]
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = ene_hi - ene_lo

    eene = np.logspace(np.log10(0.2),np.log10(ene_ce.max()),18)
    eelo = eene[:-1]
    eehi = eene[1:]
    make_grpflg(src_name,grp_name,method="EDGE",rmf_file=rmf_name,eelo=eelo,eehi=eehi)
    with fits.open(grp_name) as hdu:
        data = hdu[1].data
    grpflg = data["GROUPING"]
    grpene_lo,grpene_hi,grppha,grpphaerr = rebin_pi(ene_lo,ene_hi,pha,phaerr,grpflg)
    grpene_lo,grpene_hi,grpbkgpha,grpbkgphaerr = rebin_pi(ene_lo,ene_hi,bkgpha,bkgphaerr,grpflg)
    with np.errstate(invalid='ignore'):
        grpbkgfrac = grpbkgpha/grppha
    grpene_wd = grpene_hi - grpene_lo
    grpene_ce = (grpene_lo + grpene_hi) / 2

    ax_twinx = ax.twinx()
    ax_twinx.plot(grpene_ce,1-grpbkgfrac,c=color_right)
    ax_twinx.set_ylim(0,1.1)
    ax_twinx.set_ylabel("Net/Total counts",fontsize=10,color=color_right)
    ax_twinx.tick_params("y",which="major",length=10,width=1.0,size=5,labelsize=8,pad=3,labelcolor=color_right)
    ax_twinx.tick_params("y",which="minor",length=10,width=1.0,size=5,labelsize=8,pad=3,labelcolor=color_right)

    return ax, ax_twinx


#===================================================
################# UV correction ####################
#===================================================
# NOTE: this section should be used with caution!
def dered(flux,RA,DEC,R=5.28):
    """
    Perform Galactic extinction correction. SFDMAP generated from 
    https://github.com/kbarbary/sfdmap.
    To use this function, please make sure `SFD_DIR` has been written 
    in ~/.bashrc as an environmental variable!
    See the above link for more details on `SFD_DIR`.
    
    Parameters
    ----------
    flux : float
        Flux to be dereddened.
    RA : float
        Right Ascension of the source.
    DEC : float
        Declination of the source.
    R : float, optional
        Extinction coefficient of the band in use (see e.g., 
        Schlafly&Finkbeiner2011, Fang+2023).

    Returns
    -------
    flux_der : float
        The dereddened flux.
    """
    # Assuming R(UVW1)=5.28 (Page+2013)
    m = sfdmap.SFDMap() # Assuming that SFD_DIR has been written in ~/.bashrc as an environmental variable!
    A = R * m.ebv(RA,DEC)
    flux_der = flux * 10**(A/2.5)
    return flux_der

def kcorr(flux,z,alpha=0.65):
    """
    K-correction involves spectral shift and distance correction. This 
    function only does the spectral shifting.
    
    Parameters
    ----------
    flux : float
        The observed-frame flux (erg/cm^2/s/AA) recorded by some filter.
    z : float
        Redshift of the source.
    alpha : float, optional
        Spectral slope (assuming F_nu ~ nu^{-alpha}, where F_nu in units 
        of [erg/cm^2/s/AA]). Defaults to 0.65 (Natali+1998).
    
    Returns
    -------
    flux_int : float
        The expected flux (erg/cm^2/s/AA) recorded by some filter, if 
        observed in rest-frame. 
        Note that the distance effect (F_nu ~ L_nu / 4/pi/d_L^2) has to 
        be considered separately.
    """
    # UV SED: F_nu~nu^-alpha
    # Assume alpha=0.65 (Natali+1998)
    K = 2.5*(alpha-1)*np.log10(1+z)
    flux_int = flux * 10**(0.4*K)
    return flux_int

def flux_sft(flux,lambda_fr=2315.7,lambda_to=2500,alpha=0.65):
    """
    Calculate the expected flux (erg/cm^2/s/Hz) at `lambda_to` based on 
    the flux at `lambda_fr`, and the assumed spectral slope `alpha`.
    
    Parameters
    ----------
    flux : float
    lambda_fr : float
        lambda (from).
    lambda_to : float
        lambda (to).
    alpha : float
        Spectral slope (assuming F_nu ~ nu^{-alpha}, where F_nu in units of [erg/cm^2/s/Hz]).

    Returns
    -------
    flux_sft : float

    """
    # UV SED: F_nu~nu^-alpha
    # Assume alpha=0.65 (Natali+1998)
    flux_sft = flux * (lambda_to / lambda_fr)**alpha
    return flux_sft

def flux2lum(flux,z):
    """
    Convert flux to luminosity.
    
    Parameters
    ----------
    flux : float
        Monochromatic flux (erg/cm^2/s/Hz).
    z : float
        Redshift.

    Returns
    -------
    lum : float
        Monochromatic luminosity (erg/s/Hz).
    """
    luminosity_distance = Planck18.luminosity_distance(z)
    # Calculate the luminosity using the formula: Luminosity = 4 * pi * (D_L^2) * F_observed / (1 + z)
    # Where Luminosity is in erg/s, D_L is the luminosity distance, F_observed is the observed flux, and z is the redshift.
    #luminosity = 4 * np.pi * (luminosity_distance.to(u.cm).value ** 2) * flux / (1 + z)
    lum = 4 * np.pi * (luminosity_distance.to(u.cm).value ** 2) * flux
    return lum

def restlum(flux,z,alpha,lambda_fr=2315.7,lambda_to=2500,R=None,RA=0,DEC=0):
    """
    A synthesized function converting obs-frame monochromatic flux (erg/cm^2/s/Hz) to rest-frame luminosity (erg/s).
    
    Parameters
    ----------
    flux : float
        Observed-frame monochromatic flux (erg/cm^2/s/Hz).
    z : float
        Redshift.
    alpha : float
        SED slope.
    lambda_fr : float
        Lambda (from) in units of AA.
    lambda_to : float
        Lambda (to) in units of AA.
    
    Returns
    -------
    lum : float
        Rest-frame luminosity, nu*F_nu or lambda*F_lambda (erg/s)
    """
    # deredden (optional)
    if R is not None:
        flux = dered(flux,RA,DEC,R) # erg/cm^2/s/Hz
    # K correction
    flux = kcorr(flux,z,alpha) # erg/cm^2/s/Hz
    # Shift to some common band (e.g. 2500AA)
    flux = flux_sft(flux,lambda_fr,lambda_to,alpha) # erg/cm^2/s/Hz
    flux = flux * c / (lambda_to * 1e-10) # erg/cm^2/s
    # Flux to Lum
    lum = flux2lum(flux,z) # erg/s
    return lum


#===================================================
################# GalNH (X-ray) ####################
#===================================================
def get_nh(RA,DEC):
    """
    Get the Galactic NH from NASA"s HEASARC tool `NH` (https://heasarc.gsfc.nasa.gov/Tools/w3nh_help.html).
    Please ensure the HEASOFT env has been set up.
    
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


#===================================================
############### RMF Visualization ##################
#===================================================
def view_rmf(
    rmf_file,n_grid_i=1000,n_grid=1000,
    fig=None,ax=None,fig_name=None,cmap="gray_r",log_scale=False,v_min_lbound=1e-6,
    x_label="Output photon energy (keV)",y_label="Input model energy (keV)",
    ):
    """
    A convenient tool for visualizing 2D RMF matrix. 2D interpolation 
    assumed. You can either call it inside your code to visualize RMF 
    alone side other plots you would like to plot; or you can use this 
    function to produce standalone PNG. 
    
    Parameters
    ----------
    rmf_file : str
        Name of the RMF file.
    n_grid_i : int, optional
        Number of grids for the input model energy (does not have to be 
        the same as the length of `ENERG_LO` or `ENERG_HI`). Defaults to 
        1000.
    n_grid : int, optional
        Number of grids for the output photon energy (does not have to be 
        the same as the length of `E_MIN` or `E_MAX`). Defaults to 1000.
    fig : matplotlib.figure.Figure, optional
        The current figure.
    ax : matplotlib.axes.Axes, optional
        The current axes.
    fig_name : str, optional
        Output figure name. If specified, will create an image.
    cmap : str, optional
        cmap. Defaults to "gray_r".
    log_scale : bool, optional
        If True, use log-scale for cmap.
    v_min_lbound : float, optional
        The lower bound of v_min for log-cmap. This means that 
        `LogNorm(vmin=np.max(np.min(prob_new),v_min_lbound),
        vmax=np.max(prob_new))`. Defaults to 1e-6.
    x_label : str, optional
        X label. Defaults to "Output photon energy (keV)".
    y_label : str, optional
        Y label. Defaults to "Input model energy (keV)".

    Returns
    -------
    ax : matplotlib.axes.Axes
        The current axes.
    """
    with fits.open(rmf_file) as hdu:
        mat = hdu["MATRIX"].data # `MATRIX` extension, determine the input model (=arf) energy bin (ENERG_LO,ENERG_HI)
        ebo = hdu["EBOUNDS"].data # `EBOUNDS` extension, determine the output (photon, or channel) energy bin (E_MIN,E_MAX)
    
    iene_lo = mat["ENERG_LO"] # input energy lower edge
    iene_hi = mat["ENERG_HI"] # input energy upper edge
    iene_ce = (iene_lo + iene_hi) / 2
    
    ene_lo = ebo["E_MIN"] # output energy lower edge
    ene_hi = ebo["E_MAX"] # output energy upper edge
    ene_ce = (ene_lo + ene_hi) / 2
    
    grid = np.meshgrid(ene_ce,iene_ce) # ( (len(iene_ce),len(ene_ce)), (len(iene_ce),len(ene_ce)) )
    prob = np.zeros(grid[0].shape)
    
    n_grp = mat["N_GRP"]
    f_chan = mat["F_CHAN"]
    n_chan = mat["N_CHAN"]
    mat = mat["MATRIX"]
    
    f_chan_0 = int(f_chan.min()) # the zero point of channel index
    for i in range(len(iene_ce)):
        f_mat = 0
        for grp_j in range(n_grp[i]):
            f_chan_j = f_chan[i][grp_j] - f_chan_0
            n_chan_j = n_chan[i][grp_j]
            e_chan_j = f_chan_j + n_chan_j # ending index of group_j in channel
            e_mat = f_mat + n_chan_j # ending index of group_j in matrix[i]
            
            prob[i][f_chan_j:e_chan_j] += mat[i][f_mat:e_mat]
            f_mat += n_chan_j
            
    # The energy bin width may not be uniform
    # e.g. smaller energy bin width near 0.05 keV, but larger energy bin width near 16 keV
    # For better visualization, we do 2d-interpolation!
    interp = RegularGridInterpolator((iene_ce, ene_ce), prob,
                                     bounds_error=False, fill_value=None)
    
    iene_new = np.linspace(min(iene_lo),max(iene_hi),n_grid_i+1)
    iene_lo_new = iene_new[:-1]
    iene_hi_new = iene_new[1:]
    iene_ce_new = (iene_lo_new + iene_hi_new) / 2
    
    ene_new = np.linspace(min(ene_lo),max(ene_hi),n_grid+1)
    ene_lo_new = ene_new[:-1]
    ene_hi_new = ene_new[1:]
    ene_ce_new = (ene_lo_new + ene_hi_new) / 2
    
    grid_new = np.meshgrid(ene_ce_new,iene_ce_new)
    prob_new = interp((grid_new[1],grid_new[0]))
    
    # normalize each row
    row_sum = np.sum(prob_new,axis=1)
    prob_new = prob_new / row_sum[:,np.newaxis]
    
    if ax is None:
        if fig_name is None:    # add axes to original plot
            ax = plt.gca()
        else:                   # only generate a plot (from command line)
            fig, ax = plt.subplots(1,1,figsize=(4,4))
        
    if log_scale==True:
        im = ax.imshow(prob_new[::-1],
                       extent=(ene_ce_new[0],ene_ce_new[-1],iene_ce_new[0],iene_ce_new[-1]),
                       norm=LogNorm(vmin=max(np.min(prob_new),v_min_lbound),vmax=np.max(prob_new)),
                       aspect="auto",cmap=cmap,)
    else:
        im = ax.imshow(prob_new[::-1],
                       extent=(ene_ce_new[0],ene_ce_new[-1],iene_ce_new[0],iene_ce_new[-1]),
                       aspect="auto",cmap=cmap,)
    
    ax.set_xlabel(x_label,fontsize=15)
    ax.tick_params("x",which="major",
                   length=10,width=1.0,size=5,labelsize=10,pad=3)
    ax.tick_params("x",which="minor",
                   length=10,width=1.0,size=3,labelsize=10,pad=3)
    
    ax.set_ylabel(y_label,fontsize=15)
    ax.tick_params("y",which="major",
                   length=10,width=1.0,size=5,labelsize=10)
    ax.tick_params("y",which="minor",
                   length=10,width=1.0,size=3,labelsize=10)
    
    spines = ax.spines
    for spine in spines.values():
        spine.set_linewidth(2.5)

    # inset colorbar
    # axins1 = inset_axes(ax,width="40%",height="4%",loc="lower right")
    axins1 = inset_axes(ax,width="100%",height="50%",bbox_to_anchor=(0.5,0.15,0.5,0.1),bbox_transform=ax.transAxes)
    if log_scale == True:
        ticks = np.logspace(np.log10(max(v_min_lbound,np.min(prob_new))),np.log10(np.max(prob_new)),3)
    else:
        ticks = np.linspace(max(v_min_lbound,np.min(prob_new)),np.max(prob_new),3)
    cbar = fig.colorbar(im,cax=axins1,orientation="horizontal",ticks=ticks)
    cbar.ax.set_xticklabels(["{:.0e}".format(c) for c in ticks])
    axins1.xaxis.set_ticks_position("top")
    axins1.tick_params(labelsize=12,pad=2,width=2,size=14)
    cbar.set_label("Probability",size=14)

    if fig_name is not None: 
        plt.savefig("%s"%fig_name,bbox_inches="tight",transparent=False,dpi=300)

    return ax


#===================================================
################ Concatenating RMFs ################
#===================================================
def concat_rmf(rmffile1,rmffile2,Es,Ee,Ngrid,out_name):
    """
    Concatenate two RMFs into a single large RMF.
    
    Parameters
    ----------
    rmffile1 : str
        Name of rmf with lower energy.
    rmffile2 : str
        Name of rmf with higher energy.
    Es : float
        Starting energy of the output rmf. Cannot be larger than minimum 
        energy of `rmffile1`.
    Ee : float
        Ending energy of the output rmf. Cannot be smaller than maximum 
        energy of `rmffile2`.
    Ngrid : int
        Number of grids between `Es` and `rmffile1` (also between 
        `rmffile1` and `rmffile2`, between `rmffile2` and `Ee`).
    out_name : str
        Output rmf name.

    Returns
    -------
    prob : numpy.ndarray
        The output 2D RMF matrix.
    """
    with fits.open(rmffile1) as hdu:
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
    f_chan1_0 = int(np.min([np.min(f_chan1[_]) for _ in range(len(f_chan1))])) # the zero point of channel index

    with fits.open(rmffile2) as hdu:
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
    f_chan2_0 = int(np.min([np.min(f_chan2[_]) for _ in range(len(f_chan2))])) # the zero point of channel index

    assert np.max(arfene1_hi) <= np.min(arfene2_lo), "Highest model energy of `rmffile1` (detected: %f) should be no greater than lowest model energy (detected: %f) of `rmffile2` !"%(np.max(arfene1_hi),np.min(arfene2_lo))
    assert np.max(arfene1_hi) <= np.min(arfene2_lo), "Highest model energy of `rmffile1` (detected: %f) should be no greater than lowest model energy (detected: %f) of `rmffile2` !"%(np.max(arfene1_hi),np.min(arfene2_lo))
    assert np.max(ene1_hi) <= np.min(ene2_lo), "Highest channel energy of `rmffile1` (detected: %f) should be no greater than lowest channel energy (detected: %f) of `rmffile2` !"%(np.max(ene1_hi),np.min(ene2_lo))

    arfenes1 = np.logspace(np.log10(Es),np.log10(np.min(arfene1_lo)),Ngrid) # model energy grid from Es to 1st min model energy of rmffile1
    arfene12 = np.logspace(np.log10(np.max(arfene1_hi)),np.log10(np.min(arfene2_lo)),Ngrid) # model energy grid from last max model energy of rmffile1 to 1st min model energy of rmffile2
    arfene2e = np.logspace(np.log10(np.max(arfene2_hi)),np.log10(Ee),Ngrid) # model energy grid from last max model energy of rmffile2 to Ee
    arfene_lo = np.concatenate((arfenes1[:-1],arfene1_lo,arfene12[:-1],arfene2_lo,arfene2e[:-1]))   # model lower energy of the new arfene grid 
    arfene_hi = np.concatenate((arfenes1[1:],arfene1_hi,arfene12[1:],arfene2_hi,arfene2e[1:]))      # model upper energy of the new arfene grid 
    arfene_ce = (arfene_lo + arfene_hi) / 2
    arfene_wd = arfene_hi - arfene_lo
    arfene_id = np.arange(len(arfene_ce))
    didx_arfene1 = len(arfenes1) - 1    # 1st idx of rmffile1 in the new arfene grid
    didx_arfene2 = len(arfenes1) - 1 + len(arfene1_lo) + len(arfene12) - 1  # 1st idx of rmffile2 in the new arfene grid

    enes1 = np.logspace(np.log10(Es),np.log10(np.min(ene1_lo)),Ngrid)
    ene12 = np.logspace(np.log10(np.max(ene1_hi)),np.log10(np.min(ene2_lo)),Ngrid)
    ene2e = np.logspace(np.log10(np.max(ene2_hi)),np.log10(Ee),Ngrid)
    ene_lo = np.concatenate((enes1[:-1],ene1_lo,ene12[:-1],ene2_lo,ene2e[:-1]))
    ene_hi = np.concatenate((enes1[1:],ene1_hi,ene12[1:],ene2_hi,ene2e[1:]))
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = ene_hi - ene_lo
    ene_id = np.arange(len(ene_ce))
    didx_ene1 = len(enes1) - 1    # 1st idx of rmffile1 in the new ene grid
    didx_ene2 = len(enes1) - 1 + len(ene1_lo) + len(ene12) - 1  # 1st idx of rmffile2 in the new ene grid


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
    # RMF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
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
    # RMF header following OGIP standards (https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/caldb_doc.html, CAL/GEN/92-002: "The Calibration Requirements for Spectral Analysis")
    hdu_ebounds.header["TELESCOP"] = "CONCAT"
    hdu_ebounds.header["INSTRUME"] = "CONCAT"
    hdu_ebounds.header["CHANTYPE"] = "PI"
    hdu_ebounds.header["DETCHANS"] = prob.shape[1]
    hdu_ebounds.header["HDUCLASS"] = "OGIP"
    hdu_ebounds.header["HDUCLAS1"] = "RESPONSE"
    hdu_ebounds.header["HDUCLAS2"] = "EBOUNDS"
    hdu_ebounds.header["HDUVERS"] = "1.2.0"
    hdu_lst.append(hdu_ebounds)

    hdu_lst.writeto("%s"%(out_name), overwrite=True)

    return prob


#===================================================
################ Concatenating ARFs ################
#===================================================
def concat_arf(arffile1,arffile2,Es,Ee,Ngrid,out_name):
    """
    Concatenate two ARFs into a single large ARF.
    
    Parameters
    ----------
    arffile1 : str
        Name of arf with lower energy.
    arffile2 : str
        Name of arf with higher energy.
    Es : float
        Starting energy of the output arf. Cannot be larger than minimum 
        energy of `arffile1`.
    Ee : float
        Ending energy of the output arf. Cannot be smaller than maximum 
        energy of `arffile2`.
    Ngrid : int
        Number of grids between `Es` and `arffile1` (also between 
        `arffile1` and `arffile2`, between `arffile2` and `Ee`).
    out_name : str
        Output ARF name.

    Returns
    -------
    specresp : numpy.ndarray
        The output ARF specresp.
    """
    with fits.open(arffile1) as hdu:
        arf1 = hdu["SPECRESP"].data
        expo = hdu["SPECRESP"].header["EXPOSURE"]
    arfene1_lo = arf1["ENERG_LO"]
    arfene1_hi = arf1["ENERG_HI"]
    arfene1_ce = (arfene1_lo + arfene1_hi) / 2
    arfene1_wd = arfene1_hi - arfene1_lo
    specresp1 = arf1["SPECRESP"]

    with fits.open(arffile2) as hdu:
        arf2 = hdu["SPECRESP"].data
    arfene2_lo = arf2["ENERG_LO"]
    arfene2_hi = arf2["ENERG_HI"]
    arfene2_ce = (arfene2_lo + arfene2_hi) / 2
    arfene2_wd = arfene2_hi - arfene2_lo
    specresp2 = arf2["SPECRESP"]

    arfenes1 = np.logspace(np.log10(Es),np.log10(np.min(arfene1_lo)),Ngrid) # model energy grid from Es to 1st min model energy of arffile1
    arfene12 = np.logspace(np.log10(np.max(arfene1_hi)),np.log10(np.min(arfene2_lo)),Ngrid) # model energy grid from last max model energy of rmffile1 to 1st min model energy of arffile2
    arfene2e = np.logspace(np.log10(np.max(arfene2_hi)),np.log10(Ee),Ngrid) # model energy grid from last max model energy of arffile2 to Ee
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

    hdu_lst.writeto("%s"%(out_name), overwrite=True)

    return specresp


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
################# Folding Model ####################
#===================================================
def align_model(oarfene_lo,oarfene_hi,omodel,narfene_lo,narfene_hi):
    """
    Original model (defined on `oarfene` grid) --> New model (defined on 
    `narfene` grid).

    Parameters
    ----------
    oarfene_lo : numpy.ndarray
        Lower edge of original model energy bin.
    oarfene_hi : numpy.ndarray
        Upper edge of original model energy bin.
    omodel : numpy.ndarray
        Model flux defined on original model energy bin.
    narfene_lo : numpy.ndarray
        Lower edge of new model energy bin.
    narfene_hi : numpy.ndarray
        Upper edge of new model energy bin.

    Returns
    -------
    nmodel : numpy.ndarray
        Model flux defined on new model energy bin.
    """
    oarfene_wd = oarfene_hi - oarfene_lo
    narfene_wd = narfene_hi - narfene_lo
    nmodel = np.zeros(len(narfene_lo))    # aligned model
    for i in range(len(nmodel)):
        mask = (narfene_lo[i] <= oarfene_hi) & (narfene_hi[i] > oarfene_lo)
        if np.all(mask==False):
            print(i)
            continue
        oarfene_mask_lo = oarfene_lo[mask].copy()
        oarfene_mask_hi = oarfene_hi[mask].copy()
        oarfene_mask_wd = oarfene_wd[mask].copy()
        omodel_mask = omodel[mask].copy()
        
        # for the first and last masked channel, we need to recalculate their widths
        oarfene_mask_wd[0] = oarfene_mask_hi[0] - narfene_lo[i]
        oarfene_mask_wd[-1] = narfene_hi[i] - oarfene_mask_lo[-1]

        if len(omodel_mask) == 1:
            oarfene_mask_wd[0] = narfene_wd[i]

        nmodel[i] = np.mean(omodel_mask)
        #nmodel[i] = np.sum(oarfene_mask_wd * omodel_mask) / narfene_wd[i]
        #nmodel[i] = np.sum(oarfene_mask_wd * omodel_mask) / np.sum(oarfene_mask_wd)

    return nmodel

# def process_entry_fmodel(ext_idx,hdu,modelfile,rmffile,prob,ene_lo,ene_hi,arfene_lo,arfene_hi):
#     """
#     Parallel function to be called in `fold_model`.
#     """
#     data = hdu[ext_idx].data

#     arfene_ce = (arfene_hi + arfene_lo) / 2
#     arfene_wd = arfene_hi - arfene_lo
#     ene_ce = (ene_lo + ene_hi) / 2
#     ene_wd = ene_hi - ene_lo
    
#     oarfene_lo = data["ENERG_LO"]
#     oarfene_hi = data["ENERG_HI"]
#     oarfene_ce = (oarfene_lo + oarfene_hi) / 2
#     oarfene_wd = oarfene_hi - oarfene_lo

#     fmodel_lst = [ene_lo,ene_hi]
#     parname_lst = [colname for colname in data.columns.names if colname not in ["ENERG_LO","ENERG_HI"]]
#     colname_lst = ["E_MIN","E_MAX"] + parname_lst
#     for parname in parname_lst:
#         omodel = data[parname]
#         model = align_model(oarfene_lo,oarfene_hi,omodel,arfene_lo,arfene_hi)   # model flux based on arfene_ce grid
#         ctrate = model * arfene_wd
#         fctrate = np.sum(ctrate[:,np.newaxis]*prob,axis=0)
#         fmodel_lst.append(fctrate/ene_wd)    # folded model
#     format_lst = ["D" for _ in range(len(colname_lst))]
#     unit_lst = ["keV","keV"] + ["cts/s/cm^2/keV" for _ in range(len(fmodel_lst))]

#     columns = [fits.Column(name=colname_,format=format_,array=array_,unit=unit_) for colname_,format_,array_,unit_ in zip(colname_lst,format_lst,fmodel_lst,unit_lst)]

#     hdu_data = fits.BinTableHDU.from_columns(columns,name=hdu[ext_idx].name)
#     hdu_data.header["DESCRIPT"] = "FOLDED MODEL"
#     hdu_data.header["MODEFILE"] = modelfile
#     hdu_data.header["RESPFILE"] = rmffile
#     hdu_data.header["CREATOR"] = "XSTACK"

#     return hdu_data

def fold_model(modelfile,rmffile,arffile,out_name):
    """
    Fold the input models ([erg/cm^2/s/keV], input model energy) through 
    response (ARF+RMF) files ([ct/s/keV], output channel energy).
    
    Different extensions store different models (models should be defined 
    in `modelfile`). Different columns store `E_MIN`, `E_MAX`, and flux 
    of different components in a model.
    
    Parameters
    ----------
    modelfile : str
        Name of file storing input models to be folded. Different 
        extensions store different models. Different columns store 
        different components. 
    rmffile : str
        Name of RMF file.
    arffile : str
        Name of ARF file.
    out_name : str
        Output fits name.
    usecpu : int
        Number of CPUs used in folding process.

    Returns
    -------
    None
    """
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
    prob = get_prob(mat,ebo)

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

    with fits.open(arffile) as hdu:
        arf = hdu["SPECRESP"].data
    specresp = arf["SPECRESP"]
    specresp_ali = align_arf(ene_lo,ene_hi,arfene_lo,arfene_hi,specresp)

    with fits.open(modelfile) as hdu:
        hdu_lst = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        hdu_lst.append(primary_hdu)

        for ext_idx in range(1,len(hdu)):
            data = hdu[ext_idx].data
            
            oarfene_lo = data["ENERG_LO"]
            oarfene_hi = data["ENERG_HI"]
            oarfene_ce = (oarfene_lo + oarfene_hi) / 2
            oarfene_wd = oarfene_hi - oarfene_lo

            fmodel_lst = [ene_lo,ene_hi]
            parname_lst = [colname for colname in data.columns.names if colname not in ["ENERG_LO","ENERG_HI"]]
            colname_lst = ["E_MIN","E_MAX"] + parname_lst
            for parname in parname_lst:
                omodel = data[parname]
                model = align_model(oarfene_lo,oarfene_hi,omodel,arfene_lo,arfene_hi)   # model flux based on arfene_ce grid
                ctrate = model * arfene_wd * specresp
                fctrate = np.sum(ctrate[:,np.newaxis]*prob,axis=0)
                # the folded model is not divided by effective area
                # as there may be 2 ways of aligning arf (RMF-weighted or not; specified by `prob` in `align_arf`)
                # and both of them could be biased at the energy where the intrinsic spectrum becomes very steep
                # or the effective area drops drastically (e.g., 0.1-0.3 keV)
                fmodel_lst.append(fctrate/ene_wd)    # folded model (cts/s/keV)
            format_lst = ["D" for _ in range(len(colname_lst))]
            unit_lst = ["keV","keV"] + ["cts/s/keV" for _ in range(len(fmodel_lst))]

            columns = [fits.Column(name=colname_,format=format_,array=array_,unit=unit_) for colname_,format_,array_,unit_ in zip(colname_lst,format_lst,fmodel_lst,unit_lst)]

            hdu_data = fits.BinTableHDU.from_columns(columns,name=hdu[ext_idx].name)
            hdu_data.header["DESCRIPT"] = "FOLDED MODEL"
            hdu_data.header["MODEFILE"] = modelfile
            hdu_data.header["RESPFILE"] = rmffile
            hdu_data.header["ANCRFILE"] = arffile
            hdu_data.header["CREATOR"] = "XSTACK"

            hdu_lst.append(hdu_data)

    # write fits file
    hdu_lst.writeto("%s"%(out_name), overwrite=True)

    return


#===================================================
##################### XSPEC ########################
#===================================================
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
