# 2.1 X-ray ABC

In this section, we will walk through some basic concepts of X-ray observation. 

If you are already familiar with these topics, feel free to skip to the next section on {doc}`./how_xstack_works`.


## 2.1.1 Big Picture: What We Actually Measure

Imagine a distant astronomical source emitting radiation with a luminosity density in units of $\mathrm{erg}\ \mathrm{s}^{-1}\ \mathrm{keV}^{-1}$. 

Our telescope receives this signal as a flux density in $\mathrm{erg}\ \mathrm{s}^{-1}\ \mathrm{cm}^{-2}\ \mathrm{keV}^{-1}$. 

This flux spectrum, which encodes the detailed physics of the astronomical source, is what we ultimately want to know. However unfortunately, we never really measure it directly.

The reason is that, the intrinsic flux spectrum is significantly modified by the instrumental response. Very roughly speaking:

- Photons are first absorbed by telescope optics and filters (related to ARF);
- And then converted into electronic signals inside the detector (related to RMF). 

A probably helpful analogy for the second effect is a *Galton board*: incoming photons are redistributed into different measured energy channels. 

What we actually obtain in X-ray astronomy is a histogram — the number of detected counts in each energy channel: $\mathrm{ct}\ \mathrm{s}^{-1}\ \mathrm{Channel}^{-1}$, commonly known as a "PI" (Pulse Invariant) or "PHA" spectrum.


(sec212-x-ray-instrumental-responses)=
## 2.1.2 X-ray Instrumental Responses

In X-ray astronomy, the full response is often separated into two parts:

### ARF ($A_j$): *Ancillary Response File*

This tells you the effective collecting area at input photon energy bin $j$. 

Three example ARFs are shown in {numref}`fig-arf`:

- purple: *XMM/PN* 
- red: *XMM/RGS*
- yellow: *XRISM/Resolve*

The drop-off and absorption edge at lower energies are mainly due to the instrumental filter designed to block UV/optical contamination. The sharp decline at higher energies results from:

- The decreasing reflectivity of grazing-incidence mirrors (the critical angle for total reflection decreases with energy);
- The drop in detector quantum efficiency (how likely each photon is captured inside the CCD/Calorimeter detector, roughly scales as $E^{-3}$ at high energies). 

You may notice that the X-ray instrumental effective areas are very small compared to optical telescopes --- even for flagship missions like *XMM-Newton* ($\lesssim 2\times 10^{3}\ \mathrm{cm}^2$). This makes X-ray observations **photon-starved** --- every photon truly "counts".

```{figure} fig/arf_comparison.png
:alt: Comparison of ARF
:width: 50%
:name: fig-arf

ARF comparison example.
```

### RMF ($R_{ij}$): *Response Matrix File*

The RMF is a two-dimensional matrix that describes the probability that an incoming photon with true energy in bin $j$ is detected in output channel $i$.

This is manifested in {numref}`fig-rmf`. 
Photons enter from the y-axis (input photon energy; the true energy) and are redistributed into detected channels on the x-axis. The color encodes the transition probability.

Ideally we would like a perfect 1:1 mapping between the input photon energy and output detected energy, i.e., the RMF matrix is perfectly diagonal. 
However in reality, X-ray RMFs are only approximately diagonal, with significant off-diagonal components. That means for example, a $6\ \mathrm{keV}$ photon is most likely detected near $6\ \mathrm{keV}$, but there is also non-negligible probability that it appears at $3\ \mathrm{keV}$ or elsewhere.

From left to right in {numref}`fig-rmf` we show RMF matrix for:

- *XMM/PN*: Although XMM/PN has the largest effective area, it has the thickest diagonal — meaning lower spectral resolution ($R=E/\Delta E\sim15$ at 2 keV).
- *XMM/RGS*: XMM/RGS has a thinner diagonal ($R\sim200$), but shows strong higher-order diffraction components due to its grating design.
- *XRISM/Resolve*: a microcalorimeter, achieves the best energy resolution ($R\sim400$ at 2 keV), producing an RMF that is nearly perfectly diagonal. 

```{figure} fig/rmf_comparison.png
:alt: Comparison of RMF
:width: 100%
:name: fig-rmf

RMF comparison example.
```


## 2.1.3 Core Equation Connecting Flux Spectrum And PI Spectrum

The intrinsic flux spectrum $F_j$ (what we want) and the observed PI spectrum $C_i$ are related by:

$$
C_i \sim \mathrm{Poisson}(\lambda_i + b_i),
$$

where $b_i$ is background contribution, and $\lambda_i$ is the expected source counts in channel $i$:

$$
\lambda_i = \sum_j R_{ij}\times A_j\times F_j\times \Delta t\times \Delta E_j
= \sum_j P_{ij}\times F_j\times \Delta t\times \Delta E_j
$$

with:

- $F_j$: intrinsic source model at input energy bin $j$,
- $\Delta t$: exposure time,
- $\Delta E_j$: input-bin width.

Two key complications arise:

1.	The counts follow Poisson statistics.
2.	The RMF is non-diagonal.

Therefore, $F_j$ cannot be directly inverted from $C_i$.
Instead, X-ray analysis relies on forward modeling: assume a model $F_j$, convolve it with the response, compare to data, and infer the posterior distribution of model parameters.


## 2.1.4 Why Stacking Is Useful (And What It Means)

Two major modes of X-ray observations: 

- Pointed observations: small samples with long exposures and high-quality spectra.
- Survey observations: large samples with short exposures and limited per-source quality.

Traditional X-ray studies focused on pointed observations, yielding deep physical insight into individual objects. However, such samples may represent only the "tip of the iceberg".

In the era of *eROSITA* (and *Einstein Probe*), we can test whether conclusions drawn from individual sources hold at the population level. Because individual survey spectra often lack sufficient counts for detailed fitting, stacking becomes essential.

But note the caveat: a stacked spectrum is a population-level average and may not match any one individual source exactly.


## 2.1.5 Why X-ray Is Harder Than Optical Stacking

That said, X-ray spectral stacking is non-trivial, and is in fact more complex than traditional optical spectral stacking. We have already learned from {ref}`sec212-x-ray-instrumental-responses`, that X-ray spectrum features in:

1. **Low-count regime**: X-ray has much fewer photon counts than optical, therefore counts follow Poisson rather than Gaussian. Unlike optical spectra, we cannot arbitrarily rescale PI spectra without invalidating uncertainty propagation.
2. **Non-diagonal response**: True photon energy does not map directly to detected channel. Off-diagonal RMF components further complicate the mapping between intrinsic flux spectrum and observed PI spectrum.

In optical workflows, Gaussian approximations and simple rescaling are often workable.
In X-ray, naïve rescaling can break uncertainty treatment and bias interpretation.

This is why we have developed `Xstack`.


## 2.1.6 Further Reading

- Statistical aspects of X-ray astronomy: [Buchner&Boorman2023](https://scixplorer.org/abs/2023hxga.book..150B/abstract)
- PI (PHA/PI) format guidance: [OGIP_92_007](https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007.pdf)
- RMF/ARF format guidance: [OGIP_92_002](https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.pdf)
- See also our appendix: [Chen&Buchner2025](https://scixplorer.org/abs/2025A%26A...701A.144C/abstract)
