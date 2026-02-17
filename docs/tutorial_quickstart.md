# Tutorial / Quickstart

This page gives a practical start from `demo/demo.ipynb` and `README.md`, covering both CLI and Python workflows.

## 1. Prepare Input Files

For each source PI file, Xstack expects:

- source PI spectrum (OGIP-compliant `SPECTRUM` extension),
- background PI (from header `BACKFILE`),
- RMF (from header `RESPFILE`),
- ARF (from header `ANCRFILE`),
- redshift text file at `your.pi.z`,
- optional Galactic NH text file at `your.pi.nh`.

`filelist.txt` should contain one source PI path per line.

## 2. Quick CLI Run

```bash
runXstack your_filelist.txt --prefix ./results/stacked_
```

Main outputs:

- `./results/stacked_pi.fits`
- `./results/stacked_bkgpi.fits`
- `./results/stacked_arf.fits`
- `./results/stacked_rmf.fits`
- `./results/stacked_fene.fits`

### Common CLI Example

```bash
runXstack your_filelist.txt \
  --prefix ./results/stacked_ \
  --rsp_weight_method SHP \
  --rsp_project_gamma 2.0 \
  --flux_energy_lo 1.0 \
  --flux_energy_hi 2.3 \
  --nthreads 20 \
  --ene_trc 0.2 \
  --same_rmf AllSourcesUseSameRMF.rmf \
  --do_cache
```

### Bootstrap Example

```bash
runXstack your_filelist.txt \
  --prefix ./results/stacked_ \
  --bootstrap \
  --num_bootstrap 100 \
  --bootstrap_portion 1.0
```

## 3. Python Module Workflow

```python
from Xstack.Xstack import XstackRunner
from Xstack.config import default_nh_file

# Provide your own file lists and values:
# pifile_lst, bkgpifile_lst, arffile_lst, rmffile_lst, z_lst, nh_lst

runner = XstackRunner(
    pifile_lst=pifile_lst,
    arffile_lst=arffile_lst,
    rmffile_lst=rmffile_lst,
    z_lst=z_lst,
    bkgpifile_lst=bkgpifile_lst,
    nh_lst=nh_lst,
    rspwt_method="SHP",
    rspproj_gamma=2.0,
    int_rng=(1.0, 2.3),
    nh_file=default_nh_file,
    Nbkggrp=10,
    ene_trc=0.2,
    extended=False,
    nthreads=20,
    bootstrap=False,
    prefix="./results/stacked_",
    do_cache=False,
)
runner.run()
```

## 4. What To Check Next

After stacking:

- inspect valid energy range (source-contribution fraction and net-count fraction),
- make a data/ARF quick-look spectrum,
- fit in XSPEC (recommended statistic: PG-stat for Poisson source + Gaussian background treatment).

See also `demo/demo.ipynb` and `demo/xspec_sh/*`.

