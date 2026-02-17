# Xstack

Xstack is an open-source pipeline for X-ray spectral rest-frame shifting and stacking.

## Motivation

X-ray spectral stacking is harder than optical stacking because:

- X-ray spectra are often low-count and follow Poisson statistics.
- RMF/ARF responses are non-diagonal and must be treated consistently with the spectrum.

Xstack addresses this by stacking rest-frame PI counts directly and stacking full responses with physically motivated weights.

## Download And Installation

```bash
git clone https://github.com/AstroChensj/Xstack.git
cd Xstack
python -m pip install .
```

If HTTPS cloning fails due to network constraints, use SSH:

```bash
ssh -T git@github.com
git clone git@github.com:AstroChensj/Xstack.git
```

```{toctree}
:maxdepth: 2
:caption: Contents

tutorial_quickstart
user_guide/index
api_reference
version_history
acknowledgements
```