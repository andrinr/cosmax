# cosmax

Fast and differentiable tools for analysis and inference on structured and unstructured data in cosmology.

```pip install cosmax```

The main contributions of this package are:

* Fast and fully differentiable Power Spectrum Computation. This can be used for optiziation with loss functions that depend on the power spectrum.
* Mapping unstructured particle distribution to 3D grids with differentiable mass assignment.
* Mapping from 3D grids to unstructured particle distributions with gradient-based optimization over mass assignment.

## What can I do with this package?

## Development

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