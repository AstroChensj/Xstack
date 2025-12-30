#!/usr/bin/env python3
"""

"""
import numpy as np
from astropy.io import fits
from Xstack.utils.rsp import get_prob,align_arf,get_tlmin_from_header



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
    f_chan_0 = get_tlmin_from_header(rmffile)
    prob = get_prob(mat,ebo,f_chan_0)

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
    hdu_lst.writeto(f"{out_name}", overwrite=True)

    return