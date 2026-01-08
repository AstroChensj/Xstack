# Xstack: X-ray Spectral Stacking

:checkered_flag: **If you are in a hurry, please jump to [this link](#wrench-prerequisites-and-installation) for installation, and [this link](#ledger-how-to-use-xstack) for basic usage of this code. [Demo notebook](https://nbviewer.org/github/AstroChensj/Xstack/blob/main/demo/demo.ipynb) gives you a quick walk-through of <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>.**

## :pirate_flag: What is <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> ?

<u><span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is a comprehensive standalone pipeline code for **X-ray spectral (rest-frame) shifting and stacking**.</u>

In the era of eROSITA All Sky X-ray Survey (eRASS), the code should be very useful, if you have a special sample (point or extended sources) selected in other bands (*infra-red color, optical line/line ratios, variability*, etc), and you would like to see how their **averaged X-ray spectral shape** looks like. You simply download your targets' spectra from [eROSITA archive](https://erosita.mpe.mpg.de/dr1/erodat/data/download/), and <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> them (see [below](#ledger-how-to-use-xstack) for examples). 

## :bulb: How <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> works: a very brief introduction

### Difficulties, and solutions

X-ray spectral stacking is **non-trivial** compared to optical spectral stacking. The difficulties arise from two facts:

:cold_sweat: X-ray has much fewer photon counts (Poisson), meaning that spectral **counts and uncertainties cannot be scaled simultaneously** (as compared to optical);

:cold_sweat: X-ray has non-diagonal, complex response, meaning that the **response needs to be taken into account** when stacking.

To tackle these issues, we develop **<span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>**: a standalone pipeline code for X-ray spectral stacking. The methodology is to first sum all (rest-frame) PI spectra, without any scaling; and then sum the response files (ARFs and RMFs), each with appropriate weighting factors to preserve the overall spectral shape. The preservation of Poisson statistics for the data (and Gaussian for the background) ensures the validity of subsequent spectral fitting (via e.g., XSPEC).

### Key features of <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>

:star_struck: properly preserve X-ray spectral shape, by assigning optimal (data-driven) response weighting factors; 

:star_struck: preserve Poisson statistics; 

:star_struck: support Galactic absorption correction, if an additional ***nH*** value (in units of 1 $\text{cm}^{-2}$) for each spectrum is given.

You can find in the appendix of [S. Chen, J. Bucher, T. Liu, et al., 2025, A&A, 701, A144 (2025)](https://ui.adsabs.harvard.edu/abs/2025A%26A...701A.144C/exportcitation) more technical details!


## :wrench: Prerequisites and Installation

To install <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> (along with the prerequisite packages), simply put:
```shell
git clone https://github.com/AstroChensj/Xstack.git
cd Xstack
python -m pip install .
```
`python -m` ensures that all required packages are installed for your current conda environment only (i.e., the path where your python is called, `which python`).

Troubleshooting:

- If you encounter network issues (e.g., `Port 443`) when installing with `https`, try `ssh` instead (see [this link](https://stackoverflow.com/questions/2643502/git-how-to-solve-permission-denied-publickey-error-when-using-git) for setup of github `ssh`):
  
  ```shell
  ssh -T git@github.com
  git clone git@github.com:AstroChensj/Xstack.git
  ```

## :ledger: How to use <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>
Stacking X-ray spectra with <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is simple: you can either call it from [**command line**](#one-command-line-version), or invoke it as a [**python module**](#two-python-module-version). 

In either case, <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> requires the following as input:

- individual source `PI` spectra, with proper headers following OGIP standards; additional redshift file (with `.z` extension) and Galactic nH file (with `.nh` extension) under the same directory as each source `PI` spectrum;

  + An example of `.nh` file:

    ```
    119000000000000000000.000000
    ```

  + An example of `.z` file:

    ```
    0.1
    ```
  
- background `PI` spectra (with proper `BACKSCAL` parameters);
  
- effective area curve `ARF`s (extracted for source region);

- and response matrix `RMF`s.

The output will be: 

- stacked source `PI` spectrum;

- stacked background `BKGPI` spectrum (already scaled);

- stacked `ARF`;

- stacked `RMF`;
  
- and first contributing energy file `FENE`.

### :one: Command line version

- A simple and quick example:

  ```shell
  runXstack your_filelist.txt --prefix ./results/stacked_
  ```
  
  - And you will get the stacked spectra `./results/stacked_pi.fits`, `./results/stacked_bkgpi.fits`, and stacked response files `./results/stacked_arf.fits`, `./results/stacked_rmf.fits`, and `./results/stacked_fene.fits` which stores the first contributing energy of each individual source. 
  
  - `runXstack` is the alias for `python3 /path/to/your/Xstack/Xstack_scripts/Xstack_autoscript.py`, which should be set automatically after `python -m pip install .`.
  
  - `your_filelist.txt` stores the absolute path of the PI spectrum file for each source. The PI spectrum should follow OGIP standards – its header (of extension `SPECTRUM`) should have keywords helping <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> to find the corresponding bkg PI spectrum file (`BACKFILE`), the RMF (`RESPFILE`) and ARF (`ANCRFILE`). An example of `your_filelist.txt` would be:
 
    ```
    /path/to/your/PI_001.fits
    /path/to/your/PI_002.fits
    /path/to/your/PI_003.fits
    # ...
    ```
    Note that under each directory, it is assumed there exist a redshift file and Galactic nH file, with naming convention like `/path/to/your/PI_001.fits.z` and `/path/to/your/PI_001.fits.nh`. The redshift value and Galactic nH value ([1 cm^-2]) are stored in these two files, separately.
  
  - `1.0` and `2.3`  are lower/upper end of the energy range [keV] for computing flux. The flux represents the contribution from each source’s PI spectrum to the total stacked spectrum, and will be used as the weighting factors when stacking ARFs/RMFs.
  
  - `SHP` is the response weighting method, assuming all sources to be stacked have the same spectral shape (the minimum assumption). Under this method, the response weighting factor is calculated from flux (in a data-driven way). 
  
- Or more sophisticatedly, specify more parameters:

  ```shell
  runXstack your_filelist.txt --prefix ./results/stacked_ --rsp_weight_method SHP --rsp_proj_gamma 2.0 --flux_energy_lo 1.0 --flux_energy_hi 2.3 --nthreads 20 --ene_trc 0.2 --extended --same_rmf AllSourcesUseSameRMF.rmf
  ```

  -  `nthreads` specifies the number of CPUs used for shifting RMF.

  - `ene_trc` specifies the energy (keV) below which the ARF is unreliable and should manually be truncated. For example, for eROSITA there may be some calibration issues below 0.2 keV, so you can set this parameter to `0.2`.

  - use `--extended` only if you are stacking extended sources (methods following [X. Zhang+2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.234Z/abstract))

  - `same_rmf` : the RMF files are usually large, and sometimes all sources to be stacked could share the same RMF in order to save space. Under this case, you can specify the file name of the common RMF with `same_rmf`.

- If you want to do bootstrap, that is also easy:

  ```shell
  runXstack your_filelist.txt --prefix ./results/stacked_ --rsp_weight_method SHP --rsp_project_gamma 2.0 --flux_energy_lo 1.0 --flux_energy_hi 2.3 --nthreads 20 --ene_trc 0.2 --extended --same_rmf AllSourcesUseSameRMF.rmf --bootstrap --num_bootstrap 100
  ```

- You can run `runXstack -h` to get the documentation of all the above parameters. Or equivalently check below:

  | Parameters | Description | Default values|
  |---|---|---|
  |`filelist`|text file containing the file names|--|
  |`--prefix`|prefix for output stacked PI, BKGPI, ARF, and RMF files|`./results/stacked_`|
  |`--rsp_weight_method`|method to calculate RSP weighting factor for each source; 'SHP': assuming all sources have same spectral shape, 'FLX': assuming all sources have same shape and energy flux (weigh by exposure time), 'LMN': assuming all sources have same shape and luminosity (weigh by exposure/dist^2)|`SHP`|
  |`--rsp_project_gamma`|prior photon index value for projecting RSP matrix onto the output energy channel. This is used in the `SHP` method, to calculate the weight of each response. Defaults to 2.0 (typical for AGN).|2.0|
  |`--flux_energy_lo`|lower end of the energy range in keV for computing flux|1.0|
  |`--flux_energy_hi`|upper end of the energy range in keV for computing flux|2.3|
  |`--nthreads`|number of cpus used for non-parametric response shifting|10|
  |`--num_bkg_groups`|number of background groups|10|
  |`--ene_trc`|energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)|0.0|
  |`--extended`|whether or not this is an extended source|`False` (point source)|
  |`--same_rmf`|specify the name of common rmf, if all sources are to use the same rmf|None|
  |`--bootstrap`|activate `bootstrap` mode|False|
  |`--num_bootstrap`|number of bootstrap experiments in `bootstrap` mode|10|
  |`--bootstrap_portion`|portion of sources to resample in each bootstrap experiment|1.0|


### :two: Python module version
- An example:

  ```python
  from Xstack.Xstack import XstackRunner
  from Xstack.config import default_nh_file
  
  ## specify the input PIs, bkg PIs, RMFs, ARFs, redshifts, Galactic nHs ...
  #pifile_lst = [...]
  #bkgpifile_lst = [...]
  #rmffile_lst = [...]
  #arffile_lst = [...]
  #z_lst = [...]
  #nh_lst = [...]
  
  # run Xstack
  XstackRunner(
      pifile_lst=pifile_lst,                          # PI file list
      arffile_lst=arffile_lst,                        # ARF file list
      rmffile_lst=rmffile_lst,                        # RMF file list
      z_lst=z_lst,                                    # redshift list
      bkgpifile_lst=bkgpifile_lst,                    # bkg PI file list
      nh_lst=nh_lst,                                  # nh list
      rspwt_method="SHP",                             # method to calculate response weighting factor for each source (recommended: SHP)
      rspproj_gamma=2.0,                              # prior photon index for projecting RSP matrix onto the output energy channel.
      int_rng=(1.0,2.3),                              # if `rspwt_method`=`SHP`, choose the range to calculate flux
      nh_file=default_nh_file,                        # the Galactic absorption profile (absorption factor vs. energy)
      Nbkggrp=10,                                     # the number of background groups to calculate uncertainty of background
      ene_trc=0.2,                                    # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
      extended=False,                                 # whether or not this is an extended source
      nthreads=50,                                    # number of cpus used for response shifting
      prefix="./results/stacked_",                    # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
  ).run()
  ```

  - see `help(XstackRunner)` for the documentation for each input parameter.

- Or bootstrap:

  ```python
  from Xstack.Xstack import XstackRunner
  from Xstack.config import default_nh_file
  
  ## specify the input PIs, bkg PIs, RMFs, ARFs, redshifts, Galactic nHs ...
  #pifile_lst = [...]
  #bkgpifile_lst = [...]
  #rmffile_lst = [...]
  #arffile_lst = [...]
  #z_lst = [...]
  #nh_lst = [...]
  
  # run Xstack
  XstackRunner(
      pifile_lst=pifile_lst,                          # PI file list
      arffile_lst=arffile_lst,                        # ARF file list
      rmffile_lst=rmffile_lst,                        # RMF file list
      z_lst=z_lst,                                    # redshift list
      bkgpifile_lst=bkgpifile_lst,                    # bkg PI file list
      nh_lst=nh_lst,                                  # nh list
      rspwt_method="SHP",                             # method to calculate ARF weighting factor for each source (recommended: SHP)
      rspproj_gamma=2.0,                              # prior photon index for projecting RSP matrix onto the output energy channel.
      int_rng=(1.0,2.3),                              # if `rspwt_method`=`SHP`, choose the range to calculate flux
      nh_file=default_nh_file,                        # the Galactic absorption profile (absorption factor vs. energy)
      Nbkggrp=10,                                     # the number of background groups to calculate uncertainty of background
      ene_trc=0.2,                                    # energy below which the ARF is manually truncated (e.g., 0.2 keV for eROSITA)
      extended=False,                                 # whether or not this is an extended source
      nthreads=50,                                    # number of cpus used for RMF shifting
      bootstrap=True,              		                # resample method: `bootstrap` or `KFold`
      num_bootstrap=20,							                  # number of bootstrap experiments in `bootstrap` method
      bootstrap_portion=1.0,							            # portion to resample in `bootstrap` method
      prefix="./results/stacked_",                    # prefix for output stacked PI, BKGPI, ARF, RMF, FENE
  ).run()
  ```

  - see `help(resample_XstackRunner)` for the documentation for each input parameter.

- View [`./demo/demo.ipynb`](https://nbviewer.org/github/AstroChensj/Xstack/blob/main/demo/demo.ipynb) for a quick walk-through and more examples!


## :bow_and_arrow: What to do with your stacked spectra

Please take a look at the `Step 3` of primary example in [`./demo/demo.ipynb`](https://nbviewer.org/github/AstroChensj/Xstack/blob/main/demo/demo.ipynb). Basically, you could do the following:

  - Sanity check: from which energy should you use for stacked spectrum analysis

  - Simple visualization: a quick visualization of stacked spectral shape via data/arf plot

  - XSPEC fitting


## :warning: Limitations so far ... and contributions are welcome!

<span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is a great tool for stacking large number of low-counts X-ray spectra, especially when focusing on average spectral shapes. While we acknowledge a few limitations, they are beyond current scope and won't affect the core functionality of the code. That said, contributions are still welcome!

- **Preserving only spectral **shape**; **normalization** information is lost**
  - Each spectrum carries both **normalization** and **shape** information. For **X-ray** spectral stacking, it is in principle not possible to preserve both simultaneously, due to the complex response. <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> is designed to preserve the **shape** (by assigning optimized response weighting factors), which necessarily results in the loss of **absolute normalization**. One possible improvement could be to stack the shape (with <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span>) and normalization (with e.g., 1.0-2.3 keV image stacking) separately, and then rescale the shape spectrum using the average luminosity/flux computed independently.

- **You can only stack spectra from one instrument for now**
  - <span style="font-family: 'Courier New', Courier, monospace; font-weight: 700;">Xstack</span> assumes all spectra to be stacked share the same energy grids (from RMF). This means that you can only stack spectra from only one instrument (eROSITA or XMM or Chandra or EP...), as different instruments generally have different energy grid settings. Potential improvement could focus on creating a common energy grid for all spectra before shifting and stacking.
 
## :books: If you find our code useful, please consider citing our work ... DANKE! :smiling_face_with_three_hearts:

```latex
@ARTICLE{2025A&A...701A.144C,
       author = {{Chen}, Shi-Jiang and {Buchner}, Johannes and {Liu}, Teng and {Hagen}, Scott and {Waddell}, Sophia G.~H. and {Nandra}, Kirpal and {Salvato}, Mara and {Igo}, Zsofi and {Aydar}, Catarina and {Merloni}, Andrea and {Ni}, Qingling and {Kang}, Jia-Lai and {Cai}, Zhen-Yi and {Wang}, Jun-Xian and {Li}, Ruancun and {Ramos-Ceja}, Miriam E. and {Sanders}, Jeremy and {Georgakakis}, Antonis and {Zhang}, Yi},
        title = {The average soft X-ray spectra of eROSITA active galactic nuclei},
      journal = {\aap},
     keywords = {galaxies: active, X-rays: galaxies, High Energy Astrophysical Phenomena},
         year = 2025,
        month = sep,
       volume = {701},
          eid = {A144},
        pages = {A144},
          doi = {10.1051/0004-6361/202554737},
archivePrefix = {arXiv},
       eprint = {2506.17150},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025A&A...701A.144C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
