# cosmax

by [Andrin Rehmann](https://github.com/andrinr)

[![Downloads](https://pepy.tech/badge/cosmax)](https://pepy.tech/project/cosmax)
[![Monthly Downloads](https://pepy.tech/badge/cosmax/month)](https://pepy.tech/project/cosmax)
[![PyPI](https://img.shields.io/pypi/v/cosmax.svg)](https://pypi.python.org/pypi/cosmax)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

Fast and differentiable tools for analysis and optimization on structured and unstructured data in cosmology. Powered by JAX. 

Install with `pip install cosmax`. If you want to leverage GPU acceleration, install [jax for GPU](https://jax.readthedocs.io/en/latest/installation.html) prior to installing cosmax.

## What can I do with this package?

### Classical Use Cases

Generate the [Matter Power Spectrum](examples/power_spectrum.ipynb) from a 3D densitiy field:

<img src="https://raw.githubusercontent.com/andrinr/cosmax/refs/heads/main/data/power.png" alt="drawing" width="400"/>

Generate the [ICs for a given matter power spectrum](examples/generate_ic.ipynb):

<img src="https://raw.githubusercontent.com/andrinr/cosmax/refs/heads/main/data/ic.png" alt="drawing" width="400"/>

Cloud in a cell mass assignment scheme to [convert unstructured particles to a 3D density field](examples/cic.ipynb):

<img src="https://raw.githubusercontent.com/andrinr/cosmax/refs/heads/main/data/cic.png" alt="drawing" width="400"/>

### Optimization

With gradient optimization over the cic mass assignment scheme, we can find the particle positions that best [fit an observed density field](examples/fit.ipynb):

<img src="https://raw.githubusercontent.com/andrinr/cosmax/refs/heads/main/data/fit.png" alt="drawing" width="400"/>

We can also use the power spectrum as a loss function to find a [conditional IC density field ](examples/conditional_ic.ipynb):

<img src="https://raw.githubusercontent.com/andrinr/cosmax/refs/heads/main/data/cond_ic.png" alt="drawing" width="400"/>

## Benchmark

When measuring the execution time of the power spectrum calculation, cosmax is faster than PowerBox even **without gpu** acceleration:

<img src="https://raw.githubusercontent.com/andrinr/cosmax/refs/heads/main/data/performance.png" alt="drawing" width="300"/>

This is suprising, since PowerBox is based on FFTW, a highly optimized C library for Fourier Transforms. We have excluded the warmup execution time of the JAX JIT compiler, which includes optimization and compilation of the function. For this reason, you might not see a speedup but a slowdown if powerbox is replaced with cosmax naively.
Generally speaking, the performance gains of cosmax are felt, when the power spectrum calculations are done **repeatedly**, e.g. in optimization loops.

## Limitations

- The package is currently in development and the API is not stable.
- The package generally ONLY works with 3D square boxes and periodic boundary conditions.
- If you are looking for a python library to obtain the power spectrum of a density field, without differentiability, consider using [PowerBox](https://powerbox.readthedocs.io/en/latest/). Its API is more stable and it is more feature complete.
- I am not a physisist and even less an astrophysicist. My background is CS, hence there might be some mistakes and some of the examples are possibly not useful as better approaches already exist. If you find any mistakes, feel free to open an issue or a PR.

## Development

To develop, clone the repository and install the package in editable mode:

```
pip install -e .
```

To release as pip package, tests, docs and builds are handled automatically by github actions as defined in
.github/workflows. To make a new release:

```
git tag v*.*.*
git push origin v*.*.*
```
and change the version number in pyproject.toml.

### Test

```
pytest
```

### Build 

```
python -m build
```

### Local Docs

With the pip package sphinx installed, run

```
sphinx-apidoc -o docs/source cosmax/
sphinx-build -b html docs/source docs/_build
```

to view locally

```
cd docs/_build
python -m http.server
```

## Acknowledgements

- [PowerBox](https://powerbox.readthedocs.io/en/latest/) was used as a reference implementation of the matter power spectrum.
