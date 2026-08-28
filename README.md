# Kilonova Afterglow Inference (KAI)

[![Tests](https://gitlab.com/Jean-damas/kilonova-afterglow-inference/actions/workflows/tests.yml/badge.svg)](https://gitlab.com/Jean-damas/kilonova-afterglow-inference/actions)
[![Coverage](https://codecov.io/gh/Jean-damas/kilonova-afterglow-inference/branch/main/graph/badge.svg)](https://codecov.io/gh/Jean-damas/kilonova-afterglow-inference)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**KAI** is an open-source Python package for Bayesian inference of kilonova
ejecta parameters from multiwavelength photometry. It implements a
multi-component kilonova lightcurve model based on
[Villar et al. 2017](https://doi.org/10.3847/2041-8213/aa9c84),
coupled to a [bilby](https://lscsoft.docs.ligo.org/bilby/) inference
framework, and is validated against public GW170817 photometry.

This package accompanies the paper:

> **Jean [Mbarubucyeye] et al. (2026)**
> *KAI: Bayesian multiwavelength inference of kilonova ejecta properties
> applied to GW170817*
> Astronomy & Astrophysics, submitted. [arXiv:XXXX.XXXXX]

---

## Science overview

Kilonovae are transient optical/NIR counterparts of neutron star mergers,
powered by the radioactive decay of r-process nuclei synthesised in the
ejecta. Modelling their multiwavelength lightcurves provides direct
constraints on:

- Ejecta **mass** and **velocity** of each component
- Lanthanide fraction (via the **opacity** κ)
- r-process **nucleosynthesis** yields
- The **neutron star merger rate** and its contribution to heavy element
  enrichment

KAI provides a clean, tested, and reproducible implementation of this
modelling pipeline, from raw photometry to posterior distributions.

---

## Features

- **Multi-component kilonova model** — 1, 2, or 3 ejecta components
  (blue/purple/red) with physically motivated heating rates
  (Korobkin et al. 2012, Barnes et al. 2016)
- **Bayesian inference** via `bilby` + `dynesty` nested sampling
- **Public data loaders** for GW170817 photometry (Villar et al. 2017)
- **Systematic error handling** — per-band calibration offset nuisance
  parameters
- **Publication-quality figures** — fully reproducible from a single
  notebook
- **Modern software practices** — type hints, docstrings, pytest suite,
  GitHub Actions CI, Sphinx docs

---

## Installation

```bash
# Clone the repository
git clone https://gitlab.com/Jean-damas/kilonova-afterglow-inference
cd kilonova-afterglow-inference

# Install (standard)
pip install .

# Install with development dependencies (for testing and docs)
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, numpy, scipy, pandas, astropy, bilby, dynesty.
See `pyproject.toml` for the full dependency list.

---

## Quickstart

### 1. Load GW170817 photometry

```python
from kai.data.loaders import load_gw170817_photometry

df = load_gw170817_photometry(bands=["g", "r", "i", "J", "H"])
print(df.head())
```

### 2. Evaluate the kilonova model

```python
import numpy as np
from kai.models.kilonova import KilonovaComponent, MultiComponentKilonova

blue  = KilonovaComponent(mej=0.025, vej=0.27, kappa=0.5,  name="blue")
red   = KilonovaComponent(mej=0.040, vej=0.15, kappa=10.0, name="red")
model = MultiComponentKilonova([blue, red])

t   = np.linspace(0.5, 15, 100)
mag = model.magnitude(t, band="r", distance_mpc=40.0)
```

### 3. Run Bayesian inference

```python
from kai.inference.likelihood import KilonovaLikelihood
from kai.inference.priors    import gw170817_priors
from kai.inference.sampler   import run_inference

likelihood = KilonovaLikelihood(data=df, n_components=2, distance_mpc=40.0)
priors     = gw170817_priors(n_components=2)

result = run_inference(likelihood, priors, nlive=500)
result.plot_corner()
```

### 4. Reproduce all paper figures

```bash
jupyter nbconvert --to notebook --execute \
    notebooks/04_reproduce_paper_figures.ipynb
```

---

## Repository structure
kilonova-afterglow-inference/
├── src/kai/
│ ├── models/ # kilonova physics model
│ ├── inference/ # likelihood, priors, sampler
│ ├── data/ # GW170817 data loaders
│ └── plotting/ # figure utilities
├── tests/ # pytest suite
├── notebooks/ # exploratory + paper figure notebooks
├── data/GW170817/ # cached public photometry
├── results/ # posterior samples (not tracked by git)
├── docs/ # Sphinx documentation
├── pyproject.toml
└── README.md

---

## Running the tests

```bash
pytest                        # run full suite with coverage
pytest tests/test_models.py   # run a single module
pytest -v -k "distance"       # run tests matching a keyword
```

The CI runs the full test suite on Python 3.10, 3.11, and 3.12 on
every push via GitHub Actions.

---

## Reproducing the paper results

All results in the paper can be reproduced from scratch:

```bash
# Full inference run (~55 min with nlive=1000)
python run_inference_gw170817.py

# All paper figures from posterior samples
jupyter nbconvert --to notebook --execute \
    notebooks/04_reproduce_paper_figures.ipynb
```

Saved posterior samples are available on Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## Citation

If you use KAI in your research, please cite:

```bibtex
@article{Jean2026kai,
  author  = {[Mbarubucyeye], Jean Damascene},
  title   = {{KAI}: A {B}ayesian multiwavelength inference of kilonova
             ejecta properties applied to {GW170817}},
  journal = {Astronomy \& Astrophysics},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX}
}
```

Please also cite the underlying physics and inference frameworks this
package builds on: Villar et al. 2017, Korobkin et al. 2012,
Barnes et al. 2016, and Ashton et al. 2019 (bilby).

---

## Contributing

Contributions are welcome. Please open an issue before submitting a
pull request. All contributions must include tests and pass the CI.

---

## License

MIT License. See [LICENSE](LICENSE) for details.