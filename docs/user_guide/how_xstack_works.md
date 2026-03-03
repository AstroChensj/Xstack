# 2.2 How Xstack Works

This section is a pedagogical walkthrough of Appendix A.2-A.4 in `2025Chen_Xstack.pdf`, aligned with the implementation in `Xstack/Xstack.py`.

## 2.2.1 Goal Of The Pipeline

Xstack does three things in one pipeline:

1. shift each source to rest frame,
2. stack source and background PI spectra with statistically consistent handling,
3. stack full responses with explicit weighting so spectral shape is preserved.

## 2.2.2 Step 1: Rest-frame Shifting

### PI and background shifting

For each source with redshift $z$, observed-frame channel bins are mapped to rest frame by factor $(1+z)$.
Counts are redistributed to overlapping output bins proportionally to overlap width.

Background PI is shifted the same way as the paired source PI.

### Full-response shifting

Xstack forms full response:

$$
P_{ij} = R_{ij}A_j.
$$

Then it applies two shifts:

1. horizontal shift along output-channel axis (dispersion broadens by $(1+z)$),
2. vertical shift along input-model-energy axis (mapped to rest-frame grid).

Why this matters: PI and response remain physically coupled after shifting.

## 2.2.3 Step 2: Stacking Strategy

- Source PI spectra are summed directly (no per-source flux normalization before summation).
- Background PI spectra are stacked with source-to-background scaling and separate uncertainty treatment.
- Shifted full responses are weighted and summed, then decomposed into stacked ARF and RMF products for fitting.

This gives a final data package similar to single-source spectroscopy:

- stacked PI,
- stacked background PI,
- stacked ARF,
- stacked RMF.

## 2.2.4 Step 3: Weighting Models (Why Three Modes?)

Xstack supports three response-weighting assumptions (`rspwt_method`):

1. `SHP` (recommended):
   - assumes sources share spectral shape,
   - uses data-driven weights from counts and full response over an integration band
     (`int_rng` / CLI `flux_energy_lo`, `flux_energy_hi`),
   - generally least biased for shape recovery in heterogeneous samples.

2. `FLX`:
   - assumes same shape + same flux level,
   - response weights scale mainly with exposure
     (and region-area terms in extended mode).

3. `LMN`:
   - assumes same shape + same luminosity,
   - response weights scale with exposure / distance$^2$
     (plus region-area terms in extended mode).

Pedagogical rule of thumb:

- start from `SHP` unless you have a strong physical reason for `FLX` or `LMN`.

## 2.2.5 Background Treatment And Uncertainty

Background handling is intentionally separate from source stacking:

- background PI is shifted and accumulated with source/background scaling,
- sources are grouped by similar scaling ratio (`Nbkggrp`),
- grouped approximation is used to propagate background uncertainty robustly.

This is why `Nbkggrp` is a real analysis parameter, not just a runtime detail.

## 2.2.6 Point Sources vs Extended Sources

`extended=False` (default):

- point-source-style weighting definitions.

`extended=True`:

- includes region-area factors in weighting logic,
- better matches extended-source assumptions (surface-brightness style behavior).

## 2.2.7 Computational Controls

### Parallelism

- `nthreads` controls parallel work, especially response shifting.

### Bootstrap

- `bootstrap=True` activates resampled realizations,
- `num_bootstrap` sets realization count,
- `bootstrap_portion` sets fraction of sources drawn each realization.

### Caching

- `do_cache=True` stores and reuses per-source rest-frame intermediates (`.rf*`),
- reduces rerun time for repeated experiments on same source list,
- increases disk usage and I/O.

You can remove cache files with:

- `clear_rf_files`.

## 2.2.8 Outputs And Interpretation

For each run (or each bootstrap realization), Xstack writes:

- stacked PI spectrum,
- stacked background PI spectrum,
- stacked ARF,
- stacked RMF,
- first-contributing-energy metadata (`fene`),
- and log files documenting settings and weights.

For reproducibility, stacked FITS outputs also store the executed command in header `HISTORY` cards.

Interpretation reminder:

- the pipeline is designed to preserve average spectral shape robustly;
- do not over-interpret absolute normalization without additional calibration context.

## 2.2.9 Same-target Mode (Multiple Exposures of One Target)

Xstack also provides `same_target` mode for coadding repeated exposures of one target.

Conceptual difference from standard mode:

- standard mode: rest-frame shifting + multi-target population stacking,
- `same_target` mode: observed-frame exposure coadd for one target.

In `same_target` mode:

- rest-frame shifting is skipped,
- Galactic NH correction is skipped,
- source/background PI are summed directly (no pre-scaling of background PI),
- full response is stacked using `FLX` weighting,
- stacked `EXPOSURE` is the summed exposure,
- stacked src/bkg `AREASCAL`, `BACKSCAL`, `CORRSCAL` are written as input means.

Quality guard in logs:

- if `var(x/mean(x))` is large for any of `AREASCAL`, `BACKSCAL`, or `CORRSCAL`, a warning is logged.
