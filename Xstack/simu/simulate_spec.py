import numpy as np
import os
import subprocess
fkspec_dir = os.path.join(os.path.dirname(__file__),"fkspec_sh")

#===================================================
################## SIMULTION #######################
#===================================================
def make_fkspec(spec_model,par_lst,seed,spec_dir,src_file,rmf_file,arf_file,out_pre,out_dir,log_dir,src_expo=None,bkg_expo=None):
    """
    Make fake spectrum.

    Parameters
    ----------
    spec_model : str
        The spectral model.
        * `po` : single powerlaw
        * `abspo` : single absorbed powerlaw
        * `pobb` : primary continuum (powerlaw) + soft excess (bbody)
        * `abspobb` : absorbed primary continuum (powerlaw) + soft excess (bbody)
        * `popo` : primary continuum (powerlaw) + soft excess (powerlaw)
        * `abspopo` : absorbed primary continuum (powerlaw) + soft excess (powerlaw)
        * `abspocpl` : absorbed primary continuum (powerlaw) + soft excess (cutoffpl)
    
    par_lst : array_like
        The parameter list.
        * `po` : z, PC lum (0.5-10), gamma
        * `abspo` : nh, z, PC lum (0.5-10), gamma
        * `pobb` : z, PC lum (0.5-2), gamma, SE strength q (0.5-2)
        * `abspobb` : nh, z, PC lum (0.5-2), gamma, se strength q (0.5-2)
        * `popo` : z, PC lum (0.5-2), gamma, SE strength q (0.5-2), SE gamma
        * `abspopo` : nh, z, PC lum (0.5-2), gamma, SE strength q (0.5-2), SE gamma
        * `abspocpl` : nh, z, PC lum (0.5-2), gamma, SE strength q (0.5-2), SE gamma, SE ecut
    
    seed : int
        Seed for `fakeit` in XSPEC.

    spec_dir : str
        The directory of your sample input src PI spectrum, BKG spectrum, ARF and RMF.

    src_file : str
        The name of the sample src PI spectrum.

    rmf_file : str
        The name of the sample RMF file.

    arf_file : str
        The name of your sample input ARF file.

    out_pre : str
        The prefix of all output files (src PI spectrum, bkg PI spectrum, ARF, RMF).

    log_dir : str
        The directory to store the log file.

    src_expo : float or None
        Source exposure time for the fake spectrum (optional). If None, exposure time from `src_file` will be used. Default is None.

    bkg_expo : float or None
        Background exposure time for the fake spectrum (optional). If None, exposure time from `src_file` will be used. Default is None.
    
    Returns
    -------
    None
    
    """
    os.makedirs(out_dir,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    if src_expo is None:
        src_expo = " "
    else:
        src_expo = str(src_expo)
    if bkg_expo is None:
        bkg_expo = " "
    else:
        bkg_expo = str(bkg_expo)
    
    sh_file = f"{fkspec_dir}/{spec_model}.sh"

    par_lst = [str(ele) for ele in par_lst]
    # par_str = ' '.join(par_lst)

    result = subprocess.run([
        sh_file,
        *par_lst,
        str(seed),
        src_expo,
        bkg_expo,
        spec_dir,
        src_file,
        rmf_file,
        arf_file,
        out_pre,
        out_dir,
    ], capture_output=True, text=True)

    # Write captured output to log file
    with open(f"{log_dir}/{out_pre}.log", "w") as logfile:
        logfile.write(result.stdout)

    # Handle errors if necessary
    if result.stderr:
        print(f"Error: {result.stderr}")

    # if run_sh == True:  # if you want to generate the spectrum right away
    #     os.system("bash %s %s %d %s %s %s %s %s %s %s > %s/%s.log 2>&1"%(sh_file,par_str,seed,spec_dir,src_file,arf_file,out_pre,out_dir,src_expo,bkg_expo,log_dir,out_pre))
    # else:   # if you only want to generate the shell script (for later use, e.g. run with parallel)
    #     os.system("echo 'sh %s %s %d %s %s %s %s %s %s %s > %s/%s.log 2>&1' >> %s"%(sh_file,par_str,seed,spec_dir,src_file,arf_file,out_pre,out_dir,src_expo,bkg_expo,log_dir,out_pre,pipeline_file))

    return

def pofluxconv(flux1,start1,end1,start2,end2,alpha):
    """
    
    Parameters
    ----------
    flux1 : float
        original flux (erg/s/cm^2)

    start1 : float
        starting energy for original flux

    end1 : float
        ending energy for original flux

    start2 : float
        starting energy for new flux

    end2 : float
        ending energy for new flux

    alpha : float
        spectral slope (F_E~E^{-alpha})

    Returns
    -------
    flux2 : float
        the converted flux (erg/s/cm^2)

    """
    flux2 = flux1 * (start2**(-alpha+1) - end2**(-alpha+1))/(start1**(-alpha+1) - end1**(-alpha+1))
    return flux2