# 2.3 Input Parameters (CLI And Python)

This page documents inputs for:

- CLI: `runXstack` (`Xstack_scripts/Xstack_autoscript.py`)
- Python API: `Xstack.Xstack.XstackRunner`

## Required Inputs

### CLI

- `filelist` (positional): text file listing PI files.

### Python

- `pifile_lst`
- `arffile_lst`
- `rmffile_lst`
- `z_lst`

## Core Workflow Parameters

| CLI argument | Python argument | Meaning | Default |
|---|---|---|---|
| `--prefix` | `prefix` | Output file prefix | `./results/stacked_` |
| `--rsp_weight_method` | `rspwt_method` | Full-response weighting method: `SHP`, `FLX`, `LMN` | `SHP` |
| `--rsp_project_gamma` | `rspproj_gamma` | Prior photon index used by `SHP` projection | `2.0` |
| `--flux_energy_lo` + `--flux_energy_hi` | `int_rng` | Integration range for `SHP` weights (rest-frame keV) | `1.0`, `2.3` keV |
| `--nthreads` | `nthreads` | CPU threads for shifting/stacking pipeline | `10` (CLI), `1` (Python) |
| `--ene_trc` | `ene_trc` | Truncate unreliable low-energy ARF/PI bins below threshold | `0.0` / `None` |
| `--extended` | `extended` | Treat sample as extended sources | `False` |
| `--same_target` | `same_target` | Stack multiple exposures of one target in observed frame (skip rest-frame/NH/background pre-scaling; use FLX response weighting) | `False` |
| `--do_cache` | `do_cache` | Save/reuse per-source rest-frame cache files (`.rf*`) | `False` |

## Data And Background Parameters

| CLI argument | Python argument | Meaning | Default |
|---|---|---|---|
| (auto from PI header) | `bkgpifile_lst` | Background PI spectrum list | `None` |
| (auto from `*.nh`) | `nh_lst` | Galactic NH list | `None` |
| (not in CLI) | `nh_file` | Galactic absorption profile template file | `None` |
| `--num_bkg_groups` | `Nbkggrp` | Number of groups for background-uncertainty estimation | `10` |
| `--same_rmf` | (preprocessed in CLI loader) | Use one common RMF path for all sources | `None` |

## Bootstrap Parameters

| CLI argument | Python argument | Meaning | Default |
|---|---|---|---|
| `--bootstrap` | `bootstrap` | Enable bootstrap mode | `False` |
| `--num_bootstrap` | `num_bootstrap` | Number of realizations | `10` |
| `--bootstrap_portion` | `bootstrap_portion` | Fraction of sources resampled in each realization | `1.0` |

## Advanced Python-only Parameters

| Python argument | Meaning | Default |
|---|---|---|
| `srcid_lst` | Source IDs stored in output metadata/logging | auto index |
| `sample_rmf` | RMF used to define reference grids | first RMF |
| `sample_arf` | ARF used for reference input-energy grid | first ARF |

## Output Files

For non-bootstrap runs, Xstack writes:

- `<prefix>pi.fits` (stacked PI spectrum)
- `<prefix>bkgpi.fits` (stacked background PI spectrum)
- `<prefix>arf.fits`
- `<prefix>rmf.fits`
- `<prefix>fene.fits`
- `<prefix>runXstack.log`

In bootstrap mode, each realization gets an index in the output names.

## Same-target Mode Notes

When `same_target=True`:

- `rspwt_method` is effectively forced to `FLX`,
- `do_cache` is ignored,
- output src/bkg `AREASCAL`, `BACKSCAL`, `CORRSCAL` are means of input values,
- a log warning is raised if `var(x/mean(x))` is large for these scale keywords.
