# 2.2 How Xstack Works

An X-ray spectrum is not simply a blurred photograph of the photons emitted by
a source. A photon begins at a **model energy**, passes through the telescope's
energy-dependent effective area, and is finally recorded in a detector
**channel** that may not correspond exactly to its original energy. The ARF
describes the collecting area; the RMF describes the probability of landing in
each detector channel.

This is why stacking only the PI spectra is not enough. If Xstack moves the
counts into the rest frame but leaves their responses behind in the observed
frame, the stacked spectrum and the model used to fit it would describe two
different experiments. Xstack therefore moves the data and the full response
together, and only then combines the sources.

The method follows Appendix A of [Chen et al.
(2025)](https://doi.org/10.1051/0004-6361/202554737). What follows is a
plain-language tour of the same procedure, using the notation and array layout
of the code.

## 2.2.1 Two energy axes, not one

It helps to keep the two axes separate from the beginning:

- **Input model energy** is the energy of a photon before the detector
  redistributes it. It is the energy axis of the ARF and the rows of the Xstack
  response matrix.
- **Output channel energy** is the detector channel in which that photon is
  recorded. It is the energy axis of the PI spectrum and the columns of the
  response matrix.

```{figure} fig/rmf_two_energy_axes.png
:alt: A single RMF with input model energy on the vertical axis and output channel energy on the horizontal axis.
:width: 72%
:align: center
:name: fig-rmf-two-energy-axes

A single RMF shown using Xstack's index convention. Rows $i$ represent input
model energy, while columns $j$ represent output detector-channel energy. The
diagonal band shows that photons are usually detected near their input energy;
its finite width and off-diagonal structure encode detector redistribution.
This image shows the RMF probability $R_{ij}$, before multiplication by the ARF.
```

Let $A_i$ be the ARF effective area in model-energy bin $i$, and let $R_{ij}$
be the probability that a photon in model-energy bin $i$ is detected in
channel-energy bin $j$. Xstack first combines them into the **full response**

$$
P_{ij} = A_i R_{ij}.
$$

Each row of $P$ answers a useful question: *if photons arrive in this one
model-energy bin, how much effective collecting area contributes to every
detector channel?* The row sum is therefore the ARF `SPECRESP` at that model
energy.

```{note}
**Index convention.** Chen et al. (2025) use $i$ for output channel energy and
$j$ for input model energy in Eqs. A.1--A.3. This guide follows the Xstack
array layout instead: $i$ is the input model-energy row and $j$ is the output
channel-energy column. Thus $P_{ij}$ here corresponds to $P_{ji}$ under the
paper's convention. The physics is unchanged; only the index names are
transposed.
```

## 2.2.2 Moving the observed counts into the rest frame

Consider one observed channel with boundaries $E_\mathrm{lo}$ and
$E_\mathrm{hi}$. For a source at redshift $z$, Xstack maps that interval to

$$
[(1+z)E_\mathrm{lo},\ (1+z)E_\mathrm{hi}].
$$

The mapped interval will rarely line up perfectly with the fixed output grid.
When it overlaps several channels, Xstack divides the counts among them in
proportion to the overlap widths. A bin that overlaps two destination channels
by 40% and 60%, for example, contributes 40% of its counts to the first and 60%
to the second.

This redistribution can temporarily produce fractional counts. Xstack keeps
those fractions while sources are being shifted and summed, rather than
rounding every source separately and accumulating rounding noise. It calculates
the final PI uncertainties and writes integer source counts only after the
stack is complete.

The paired background PI spectrum is shifted through the same channel map. It
is then scaled to the source extraction region before entering the standard
multi-source stack. Keeping the source and background on the same rest-frame
grid is essential: otherwise a background feature would be subtracted from the
wrong energy.

## 2.2.3 Why the response needs two shifts

The full response is two-dimensional, so multiplying both axes by $1+z$ is not
one operation. Xstack performs it in two deliberate passes.

### First pass: move the channel-energy direction

Xstack takes one model-energy row at a time and shifts its output-channel bins
horizontally. The value from each old channel is distributed over the new
channels according to their normalized overlap widths. Contributions from all
old channels accumulate in the destination row.

This pass does more than move the response peak: it broadens the detector's
energy dispersion by $1+z$, exactly as the rest-frame channel scale requires.
After edge clipping, Xstack renormalizes each non-empty row so that its
horizontal sum still equals the ARF effective area at that model energy. This
matters at the high-energy boundary, where part of a shifted response may fall
outside the available grid.

```{figure} fig/rsp_shift_horizontal.png
:alt: Horizontal full-response shift distributing one channel value between two overlapping destination channels.
:width: 100%
:align: center
:name: fig-rsp-shift-horizontal

Horizontal shifting at one fixed model-energy row $i$. The original channel
bin $j=1$, whose matrix value is 1, overlaps two destination bins after its
boundaries are multiplied by $1+z$. It therefore contributes
$1\times0.4$ and $1\times0.6$ to their accumulators. Values shown in the coral
row identify the original row for reference; the blue `+=` terms illustrate
one contribution to the newly accumulated row.
```

In {numref}`fig-rsp-shift-horizontal`, the `+=` is important: a destination
channel generally receives contributions from more than one old channel.

### Second pass: move the model-energy direction

After the horizontal pass, the channel dispersion is in the correct frame, but
the rows still refer to the observed model-energy scale. Xstack therefore
shifts the input model-energy bins vertically by $1+z$.

Here the desired quantity is not a conserved count total along the vertical
axis. It is the response height at a particular model energy. For every fixed
output channel $j$, Xstack takes an overlap-weighted average of the shifted
model-energy rows. If destination row 0 overlaps shifted rows 0 and 1 with
weights 60% and 40%, then

```{figure} fig/rsp_shift_vertical.png
:alt: Vertical full-response shift taking an overlap-weighted average of two model-energy rows at a fixed output channel.
:width: 78%
:align: center
:name: fig-rsp-shift-vertical

Vertical shifting at one fixed output channel $j$. The destination
model-energy row $i=0$ overlaps shifted rows $i=0$ and $i=1$ by 60% and 40%,
so its new response value is their overlap-weighted average. Unlike PI counts
in the horizontal direction, response height—not a vertical integral—is being
preserved here.
```

$$
P'_{0j} = 0.6\,H_{0j} + 0.4\,H_{1j},
$$

where $H$ is the matrix after the horizontal pass, as illustrated in
{numref}`fig-rsp-shift-vertical`. This keeps the effective-area height tied to
model energy while completing the rest-frame transformation.

The two passes together keep three things consistent: the shifted PI channel
grid, the detector redistribution, and the effective area seen by the input
model.

## 2.2.4 Galactic absorption belongs with the response

Galactic absorption removes photons before they reach the telescope. Xstack
therefore treats it as an additional energy-dependent filter, rather than
trying to "correct" the observed PI counts directly. When an NH absorption
template and source NH values are supplied, the absorption factor is mapped to
the ARF grid and multiplied into `SPECRESP`. Xstack then forms and shifts the
full response from this Galactic-absorption-corrected ARF and the RMF.

The practical advantage is that the stacked PI remains an honest count
spectrum. The absorption correction lives where a forward-folded spectral
model expects it: in the response that connects an incident model to detected
counts.

The optional `ene_trc` setting follows the same philosophy. It removes an
unreliable low-energy range consistently from the PI and response; for eROSITA,
0.2 keV is a common choice.

## 2.2.5 Stacking: add the evidence, weight the experiments

Once every source is on the common rest-frame grid, the source PI spectra are
summed directly. Xstack does not normalize each low-count PI spectrum before
addition, because scaled Poisson counts no longer have a simple Poisson
uncertainty.

The responses cannot simply be averaged, however. A long exposure, a bright
source, and a distant source do not contribute to a stack in the same way.
Xstack therefore multiplies each shifted full response by a scalar weight,
sums the weighted full responses, and only at the end decomposes the result
back into a stacked ARF and RMF. Weighting the full response is important:
averaging ARFs and RMFs separately does not in general preserve their product.

Xstack offers three scientific assumptions:

### `SHP`: preserve a common spectral shape

`SHP` makes the weakest assumption: the sources share an average spectral
shape, but they may have different normalizations. Xstack estimates a
data-driven weight from the background-subtracted counts and the projected
full response in the chosen integration band. This is usually the best starting
point when individual spectra have enough counts (roughly ten or more).

The trade-off is intentional: the final vertical normalization does not carry
a simple physical flux unit. `SHP` is designed to answer *what does the average
shape look like?*

### `FLX`: assume a common flux

`FLX` assumes the sources have a similar spectral shape **and** flux. For point
sources, the response weight is proportional to exposure and includes the
rest-frame $(1+z)$ factor. Extended-source mode additionally includes region
area.

This can be a useful approximation for a flux-limited sample in which most
objects lie near the survey limit. If the true fluxes span a wide range, the
assumption can bias the recovered average shape.

### `LMN`: assume a common luminosity

`LMN` makes the analogous assumption in luminosity rather than observed flux.
Its weights include luminosity distance, with region area included for extended
sources. It is appropriate only when treating the sources as having comparable
luminosities is scientifically defensible.

These modes are not merely numerical recipes. Each one states what is being
held comparable across the population. When in doubt, begin with `SHP` and
write down why a stronger `FLX` or `LMN` assumption is justified.

## 2.2.6 Background uncertainty is kept separate

The stacked background is not just another source spectrum. Different
observations may have very different source-to-background extraction ratios.
Xstack scales each background to its corresponding source region, then groups
sources with similar scaling ratios when propagating the background
uncertainty. `num_bkg_groups` controls this approximation.

More groups retain more detail in the scaling distribution but may leave few
spectra per group; fewer groups are faster and more stable but coarser. This is
why `num_bkg_groups` is an analysis choice, not simply a performance switch.

## 2.2.7 What comes out

The result looks like an ordinary single-observation spectral package:

- a stacked source PI spectrum,
- a stacked background PI spectrum,
- a stacked ARF,
- a stacked RMF,
- an `fene` diagnostic recording where each source begins to contribute, and
- a run log containing settings, timings, warnings, and response weights.

The ARF and RMF also contain `WEIGHT` information so that the contribution of
each source can be audited. FITS `HISTORY` cards record the command used to
create the products.

## 2.2.8 Repeated observations of one target are a different problem

Sometimes the inputs are FPMA/FPMB modules or repeated exposures of one target,
not a population at different redshifts. In that case use `same_target` mode.
It coadds the observations in their observed frame:

- no rest-frame shift is performed,
- no Galactic-NH correction is applied,
- source and background PI spectra are summed directly,
- the full responses are combined with `FLX` weighting, and
- the output exposure is the sum of the input exposures.

`same_target` also writes mean `AREASCAL`, `BACKSCAL`, and `CORRSCAL` values and
warns when those quantities vary strongly among exposures. It is useful for
building one observation-level product before that target enters a standard
multi-source rest-frame stack.

## 2.2.9 Making a large run practical

Response shifting is normally the most expensive part of a standard stack.
`nthreads` shifts independent sources in parallel. `do_cache` saves the
per-source rest-frame PI, background, and response products beside the inputs,
which can make repeated experiments much faster at the cost of disk space.
Cached redshifts are checked when files are reused.

Bootstrap realizations are drawn from the already shifted products, so Xstack
does not repeat the expensive response transformation for every realization.
`num_bootstrap` sets the number of realizations and `bootstrap_portion` sets the
fraction of the sample drawn into each one.

The guiding idea throughout is simple: move each observation into a common
physical frame once, preserve the count statistics, and carry enough response
information forward that the final stacked spectrum can still be fitted like a
real X-ray observation.
