#!/usr/bin/env python3
"""

"""
import numpy as np
from astropy.io import fits
import os
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from Xstack.utils.pi import get_bkgscal,get_expo,rebin_pi,make_grpflg
from Xstack.utils.rsp import get_prob,rebin_arf,get_tlmin_from_header



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

    f_chan_0 = get_tlmin_from_header(rmf_file)
    prob = get_prob(mat,ebo,f_chan_0)   # 2D RNF probability matrix
            
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
        plt.savefig(fig_name,bbox_inches="tight",transparent=False,dpi=300)

    return ax



#===================================================
########### Quick spectral shape check #############
#===================================================
def make_dataarf_plot(
        src_name,bkg_name=None,arf_name=None,rmf_name=None,grp_name=None,
        normalize_at=None,outname=None,plot=False,ax=None,**kwargs
):
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
    ax : matplotlib.axes.Axes
        The axes for the plot. None if not specified.

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
            if factor_norm <= 0:
                raise Exception(f"Grouped data at ~{normalize_at}keV is 0 or negative --- maybe try a larger bin size when grouping data?")
            if not np.isfinite(factor_norm):
                raise Exception(f"Grouped data at ~{normalize_at}keV is NaN.")
        else:   # take average
            factor_norm = np.nan
            for window_length in [50,100,150]:
                wl = min(window_length,len(ratio))
                normalize_idxs = np.argsort(abs(grpene_ce-normalize_at))[:wl]   # TODO: better number than 50?
                ene_ce_norm = np.nanmedian(grpene_ce[normalize_idxs])
                factor_norm = np.nanmedian(ratio[normalize_idxs])
                if factor_norm > 0:
                    break
            if not np.isfinite(factor_norm):
                raise Exception(f"Grouped data at ~{normalize_at}keV is NaN.")

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

    # plot energy*energy*data
    # therefore additional normalization (ene_ce_norm**2) is needed to ensure value at ene_ce_norm is 1
    if plot:
        if ax is None:
            ax = plt.gca()
        ax.errorbar(grpene_ce,ratio*grpene_ce**2/ene_ce_norm**2,yerr=(ratioerr*grpene_ce**2/ene_ce_norm**2/2),**kwargs)

    return grpene_lo,grpene_hi,ratio,ratioerr,ax



#===================================================
########### Which energy range to use ##############
#===================================================
def valid_energy_range_plot(
        fene_name,src_name,grp_name,bkg_name,rmf_name,
        ax=None,
):
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
    make_grpflg(src_name,grp_name,method="EDGE",rmf_fname=rmf_name,eelo=eelo,eehi=eehi)
    with fits.open(grp_name) as hdu:
        data = hdu[1].data
    grpflg = data["GROUPING"]
    grpene_lo,grpene_hi,grppha,grpphaerr = rebin_pi(ene_lo,ene_hi,pha,phaerr,grpflg)
    grpene_lo,grpene_hi,grpbkgpha,grpbkgphaerr = rebin_pi(ene_lo,ene_hi,bkgpha,bkgphaerr,grpflg)
    with np.errstate(invalid="ignore"):
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
############## Make Dispersion Map #################
#===================================================
def gaussian(x, amplitude, mean, stddev):
    """
    A gaussian function.

    Parameters
    ----------
    x : float or numpy.ndarray
    amplitude : float
    mean : float
    stddev : float

    Returns
    -------
    pdf : float or numpy.ndarray
        The probability at x.
    """
    pdf = amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)
    return pdf


def get_ene_dsp(ene_ce,prob_lst,fixed_mean=True):
    """
    Get energy dispersion.

    Parameters
    ----------
    ene_ce : numpy.ndarray
        Output central channel energy.
    prob_lst : numpy.ndarray
        Probability profile for some input model energy (this is a function of output channel energy). Must have same length as `ene_ce`.
    fixed_mean : bool
        If true, the mean energy of the Gaussian will be fixed at the nominal energy (which corresponds to maximal probability).

    Returns
    -------
    norm : float
        The Gaussian normalization.
    ene_nom : float
        The Gaussian central energy (this is the nominal energy for some input energy).
    ene_dsp : float
        The Gaussian width (this is the energy dispersion for some input energy).
    """
    from scipy.optimize import curve_fit
    ene_nom = ene_ce[np.argmax(prob_lst)] # nominal energy
    if fixed_mean:
        mlo = ene_nom # lower bound for mean
        mhi = ene_nom + 1e-6 # upper bound for mean
    else:
        mlo = 0
        mhi = np.inf
    popt, pcov = curve_fit(
        gaussian, ene_ce, prob_lst,
        p0=[1, ene_nom, 1], bounds=([0, mlo, 0], [np.inf, mhi, np.inf])
    )
    norm = popt[0]
    ene_dsp = popt[2]
    return norm,ene_nom,ene_dsp


def make_dspmap(mat,ebo,out_name,f_chan_0=None):
    """
    Make energy dispersion map.
    
    Parameters
    ----------
    mat : astropy.io.fits.FITS_rec
        The `MATRIX` HDU data from a standard RMF file.
    ebo : astropy.io.fits.FITS_rec
        The `EBOUNDS` HDU data from a standard RMF file.
    out_name : str
        The output dispersion map name.
    f_chan_0 : int, optional
		First channel index. Defaults to None. 
		If not specified, will be determined from rmf file.

    Returns
    -------
    None.
    """
    iene_lo = mat["ENERG_LO"]
    iene_hi = mat["ENERG_HI"]
    iene_ce = (iene_lo + iene_hi) / 2
    iene_wd = (iene_hi - iene_lo)
    
    ene_lo = ebo["E_MIN"]
    ene_hi = ebo["E_MAX"]
    ene_ce = (ene_lo + ene_hi) / 2
    ene_wd = (ene_hi - ene_lo)
    
    # get prob_lst
    prob = get_prob(mat,ebo,f_chan_0)   # 2D RNF probability matrix
    prob_ene = prob / ene_wd # probability per energy bin
    
    # get nominal energy and energy dispersion
    print("****************** Generating dspmap ********************")
    norm = []
    ene_nom = []
    ene_dsp = []
    for i in tqdm(range(len(iene_ce))):
        norm_i,ene_nom_i,ene_dsp_i = get_ene_dsp(ene_ce,prob_ene[i],fixed_mean=True)
        norm_i /= (gaussian(ene_ce,norm_i,ene_nom_i,ene_dsp_i)*ene_wd).sum() # renormalize the gaussian profile
        norm.append(norm_i)
        ene_nom.append(ene_nom_i)
        ene_dsp.append(ene_dsp_i)
    norm = np.array(norm)
    ene_nom = np.array(ene_nom)
    ene_dsp = np.array(ene_dsp)
    
    # make fits file
    column_names = ["ENERG_LO","ENERG_HI","norm","ene_nom","ene_dsp"]
    formats = ["D","D","D","D","D"]
    arrays = [iene_lo,iene_hi,norm,ene_nom,ene_dsp]
    columns = [fits.Column(name=col_name,format=format_,array=array_) for col_name,format_,array_ in zip(column_names,formats,arrays)]
    coldefs = fits.ColDefs(columns)
    table = fits.BinTableHDU.from_columns(coldefs)
    table.writeto(out_name,overwrite=True)
    print("****************** dspmap successfully generated! ********************")
    
    return








