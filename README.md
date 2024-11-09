# cosmax
Fast and differentiable implementations of operations needed for inference and analysis in cosmology. Powered by JAX.

## Development

Build 

```
python -m build
```

### Documentation

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