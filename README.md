# cosmax
Fast and differentiable implementations of operations needed for inference and analysis in cosmology. Powered by JAX.

## Development

### Documentation

With the pip package sphinx installed, run

```
sphinx-apidoc -o docs/source cosmax/
sphinx-build -b html docs/source docs/_build
```

to view locally

```
python -m http.server
```