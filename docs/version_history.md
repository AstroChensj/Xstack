# Version History

## v1.1.2 (latest)

- Current documented release in `Xstack/config.py`.
- Last update field in code: `2026/2/3`.

Download link is [here](https://github.com/AstroChensj/Xstack).

## Earlier Versions

### v1.1.1

This release (v1.1.1) includes slight improvements to the code:

1) better treatment for extremely low data by properly assigning photon counts in each shift (individual count can be float; only round to integer for the final stacked spectrum) [Thanks Y. Zhang for helpful discussion];
2) fix stacking Swift-XRT / XMM MOS issues [Thanks S. Mondal for helpful discussion];
3) speed up initial file reading process;
4) input bkg files can be none now.

Download link is [here](https://github.com/AstroChensj/Xstack/releases/tag/v1.1.1).


### v1.1.0

Stacking extended sources is now supported, with methods inspired from Eq. 9 to Eq. 12 of [X. Zhang+2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...691A.234Z/abstract). However, some slight differences are:

- Eq. 10, the bkg spectra are scaled not only by `REGAREA`, but additionally by the averaged exposure per pixel, which effectively results in an exactly same scaling formula as for the point sources (`BACKSCAL` * `EXPOSURE` * `AREASCAL`). Note, the averaged exposure = `BACKSCAL` / `REGAREA` * `EXPOSURE`.
- Eq. 11 and Eq. 12, the separate stacking of ARF and RMF are replaced by the stacking of RSP=ARF * RMF in one step.

Overall, for extended sources, the only difference is how the response weighting factors are computed under `FLX` mode.

Download link is [here](https://github.com/AstroChensj/Xstack/releases/tag/v1.1.0).


### v1.0.0

Public version used for X-ray spectral stacking, with method detailed in Appendix A of S. Chen, J. Buchner, T. Liu et al. 2025, A&A, *The average soft X-ray spectra of eROSITA active galactic nuclei*.

Download link is [here](https://github.com/AstroChensj/Xstack/releases/tag/v1.0.0).