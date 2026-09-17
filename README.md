# ForsetiMultivariate.jl

Multivariate statistics: PCA, clustering, and MANOVA.

Part of the [Forseti](https://github.com/natapol) statistical analysis package family.

## Dependencies

This package depends on the following sibling Forseti packages, which are not
yet registered and must be added via local dev paths:

- `ForsetiCore`

## Local development

```julia
using Pkg
Pkg.develop(path="../ForsetiCore.jl")
Pkg.develop(path=".")
Pkg.test("ForsetiMultivariate")
```
