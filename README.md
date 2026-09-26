# ForsetiMultivariate.jl

Multivariate statistics: PCA, clustering, and MANOVA.

Part of the [Forseti](https://github.com/Forseti-jl) statistical analysis package family.

## Usage

```julia
using ForsetiCore, ForsetiMultivariate, DataFrames

df = DataFrame(x1 = [1.0, 2.0, 3.0, 5.0], x2 = [2.0, 1.0, 4.0, 3.0], x3 = [5.0, 3.0, 6.0, 2.0],
               group = ["A", "A", "B", "B"])

# PCA: tidy() takes a `matrix` kwarg (:loadings default, :scores, :eigenvalues)
pca_fit = df |> pca(:x1, :x2, :x3)
tidy(pca_fit)                          # loadings, long format
tidy(pca_fit; matrix = :eigenvalues)   # variance explained per component
augment(pca_fit)                       # data + PC1, PC2, ... columns

# k-means clustering
df |> kmeans_cluster(2, :x1, :x2, :x3) |> tidy      # per-cluster size/center/withinss
df |> kmeans_cluster(2, :x1, :x2, :x3) |> augment   # data + cluster assignment

# one-way MANOVA (Wilks' Lambda)
df |> manova(:group, :x1, :x2) |> tidy
```

An exact F-transform of Wilks' Lambda is used when there is one response
variable (reduces to univariate ANOVA) or exactly 2 groups (the
Hotelling's T² case); otherwise Bartlett's large-sample chi-square
approximation is used (`exact = false` on the result).

## Dependencies

This package depends on the following sibling Forseti packages, which are not
yet registered and must be added via local dev paths:

- `ForsetiCore`

## Local development

```julia
using Pkg
Pkg.develop(path="../ForsetiCore.jl")
Pkg.instantiate()
Pkg.test()
```
