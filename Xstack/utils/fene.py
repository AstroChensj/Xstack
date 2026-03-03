#!/usr/bin/env python3
"""
===================================
Module for First energy file (FENE)
===================================
:Authors:   Shi-Jiang Chen (MPE, USTC)
            Johannes Buchner (MPE)
            Teng Liu (USTC)
:Email:     JohnnyCsj666@gmail.com


"""
from astropy.io import fits
from Xstack.utils.logger import utc_now_iso,add_run_cmd_history
from Xstack.config import VERSION,LASTUPDATE,WEB


def write_fene(srcid_lst,arffene_lst,fene_lst,fene_fname="./stacked_fene.fits",run_cmd=None):
	"""
	Creating a fits storing the first energy of each source's PI spectrum 
	and ARF specresp.

	Parameters
	----------
	srcid_lst : list or numpy.ndarray
		The source ID list.
	arffene_lst : list or numpy.ndarray
		The first energy of each sources's ARF specresp.
	fene_lst : list or numpy.ndarray
		The first energy of each source's PI spectrum.
	fene_fname : str
		The output fits name.
	run_cmd : str, optional
		Full command string recorded in ``HISTORY`` for provenance.

	Returns
	-------
	None
	"""
	hdu_lst = fits.HDUList()
	
	primary_hdu = fits.PrimaryHDU()
	hdu_lst.append(primary_hdu)
	
	cols = [
		fits.Column(name="srcid",format="I",array=srcid_lst),
		fits.Column(name="arffene",format="D",array=arffene_lst,unit="keV"),
		fits.Column(name="fene",format="D",array=fene_lst,unit="keV"),
	]
	hdu_fene = fits.BinTableHDU.from_columns(cols, name="FENERGY")
	hdu_fene.header["CREATOR"] = "XSTACK"
	hdu_fene.header["HISTORY"] = f"{utc_now_iso()}: stacked first energy diagnostic file created by Xstack v{VERSION} [{LASTUPDATE}] [{WEB}]"
	add_run_cmd_history(hdu_fene.header,run_cmd)
	hdu_lst.append(hdu_fene)

	hdu_lst.writeto(f"{fene_fname}", overwrite=True)

	return
