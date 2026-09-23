"""
    ForsetiMultivariate

Multivariate statistics: PCA, k-means clustering, and one-way MANOVA.

Follows the family-wide pipe-curried calling convention and returns
`ForsetiFit` subtypes usable with [`tidy`](@ref)/[`glance`](@ref)/[`augment`](@ref):

```julia
df |> pca(:x1, :x2, :x3) |> tidy               # loadings, by default
df |> kmeans_cluster(3, :x1, :x2) |> augment   # data + cluster assignment
df |> manova(:group, :y1, :y2) |> tidy         # Wilks' Lambda test
```
"""
module ForsetiMultivariate

using DataFrames
using Statistics
using LinearAlgebra
using Random
using Distributions
using ForsetiCore

export PCAFit, pca
export KMeansFit, kmeans_cluster
export ManovaFit, manova
export tidy, glance, augment

function _numeric_matrix(df::AbstractDataFrame, cols)
    cs = isempty(cols) ? Symbol.(names(df)) : collect(cols)
    X = reduce(hcat, [Float64.(df[!, c]) for c in cs])
    return X, cs
end

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

"""
    PCAFit <: ForsetiFit

Result of [`pca`](@ref). [`tidy`](@ref) accepts a `matrix` keyword
(`:loadings` (default), `:scores`, or `:eigenvalues`), mirroring
`broom::tidy.prcomp`.
"""
struct PCAFit <: ForsetiFit
    loadings::Matrix{Float64}    # p x k
    scores::Matrix{Float64}      # n x k
    center::Vector{Float64}
    scale::Union{Vector{Float64},Nothing}
    sdev::Vector{Float64}        # length k
    var_explained::Vector{Float64}
    n::Int
    p::Int
    data::DataFrame
    cols::Vector{Symbol}
end

"""
    pca(X::AbstractMatrix{<:Real}; center = true, scale = false) -> PCAFit

Principal component analysis of the columns of `X` (observations in
rows), via SVD of the (optionally centered/scaled) data matrix.

# Formula

For (optionally centered/scaled) data matrix `Xc = UΣVᵀ` (SVD):

```
loadings = V
scores   = UΣ
sdev_j   = Σⱼⱼ / √(n-1)
var_explained_j = Σⱼⱼ² / Σᵢ Σᵢᵢ²
```

Computing PCA via the SVD of the data matrix directly (rather than the
eigendecomposition of the covariance matrix) avoids ever forming `XᵀX`,
which is the numerically preferred approach.

# References

- Pearson, K. (1901). On lines and planes of closest fit to systems of
  points in space. *Philosophical Magazine, Series 6*, 2(11), 559–572.
- Hotelling, H. (1933). Analysis of a complex of statistical variables
  into principal components. *Journal of Educational Psychology*, 24(6),
  417–441.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th
  ed.). Johns Hopkins University Press. (SVD-based computation)
"""
function pca(X::AbstractMatrix{<:Real}; center::Bool = true, scale::Bool = false)
    n, p = size(X)
    n >= 2 || throw(ArgumentError("pca requires at least 2 observations"))
    Xf = Float64.(X)
    mu = center ? vec(mean(Xf; dims = 1)) : zeros(p)
    Xc = Xf .- mu'
    sc = nothing
    if scale
        sc = vec(std(Xf; dims = 1))
        any(iszero, sc) && throw(ArgumentError("cannot scale a zero-variance column"))
        Xc = Xc ./ sc'
    end
    F = svd(Xc)
    S = F.S
    k = length(S)
    scores = F.U .* S'
    loadings = Matrix(F.V)
    sdev = S ./ sqrt(n - 1)
    total_var = sum(abs2, S)
    var_explained = total_var > 0 ? (S .^ 2 ./ total_var) : zeros(k)
    return PCAFit(loadings, scores, mu, sc, sdev, var_explained, n, p, DataFrame(), Symbol[])
end

"""
    pca(df::AbstractDataFrame, cols::Symbol...; kwargs...) -> PCAFit

Run PCA on the given columns of `df` (or all columns if `cols` is empty).
"""
function pca(df::AbstractDataFrame, cols::Symbol...; kwargs...)
    X, cs = _numeric_matrix(df, cols)
    fit = pca(X; kwargs...)
    return PCAFit(fit.loadings, fit.scores, fit.center, fit.scale, fit.sdev,
                  fit.var_explained, fit.n, fit.p, DataFrame(df), cs)
end

"""
    pca(cols::Symbol...; kwargs...)

Pipe-curried form: `df |> pca(:x1, :x2, :x3)`.
"""
pca(cols::Symbol...; kwargs...) = df -> pca(df, cols...; kwargs...)

function ForsetiCore.tidy(fit::PCAFit; matrix::Symbol = :loadings)
    k = length(fit.sdev)
    component_names = [Symbol("PC$j") for j in 1:k]
    if matrix == :loadings
        terms = isempty(fit.cols) ? ["x$i" for i in 1:fit.p] : string.(fit.cols)
        rows = [(term = t, component = component_names[j], value = fit.loadings[i, j])
                for (i, t) in enumerate(terms) for j in 1:k]
        return DataFrame(rows)
    elseif matrix == :eigenvalues
        return DataFrame(component = component_names, std_dev = fit.sdev,
                          percent = fit.var_explained, cumulative = cumsum(fit.var_explained))
    elseif matrix == :scores
        return DataFrame(fit.scores, component_names)
    else
        throw(ArgumentError("matrix must be :loadings, :eigenvalues, or :scores, got $matrix"))
    end
end

function ForsetiCore.glance(fit::PCAFit)
    return DataFrame(n_obs = [fit.n], n_variables = [fit.p], n_components = [length(fit.sdev)])
end

function ForsetiCore.augment(fit::PCAFit, data::AbstractDataFrame = fit.data)
    out = DataFrame(data)
    for j in 1:length(fit.sdev)
        out[!, Symbol("PC$j")] = fit.scores[:, j]
    end
    return out
end

# ---------------------------------------------------------------------------
# k-means clustering
# ---------------------------------------------------------------------------

"""
    KMeansFit <: ForsetiFit

Result of [`kmeans_cluster`](@ref).
"""
struct KMeansFit <: ForsetiFit
    centers::Matrix{Float64}    # k x p
    assignments::Vector{Int}
    withinss::Vector{Float64}
    totss::Float64
    tot_withinss::Float64
    betweenss::Float64
    iterations::Int
    converged::Bool
    k::Int
    n::Int
    data::DataFrame
    cols::Vector{Symbol}
end

function _kmeans_pp_init(X::Matrix{Float64}, k::Int, rng::AbstractRNG)
    n = size(X, 1)
    centers = zeros(k, size(X, 2))
    centers[1, :] = X[rand(rng, 1:n), :]
    for c in 2:k
        dists = [minimum(sum(abs2, X[i, :] .- centers[j, :]) for j in 1:(c - 1)) for i in 1:n]
        total = sum(dists)
        if total == 0
            centers[c, :] = X[rand(rng, 1:n), :]
            continue
        end
        r = rand(rng) * total
        cum = 0.0
        chosen = n
        for i in 1:n
            cum += dists[i]
            if r <= cum
                chosen = i
                break
            end
        end
        centers[c, :] = X[chosen, :]
    end
    return centers
end

function _kmeans_run(X::Matrix{Float64}, centers::Matrix{Float64}, k::Int, max_iter::Int)
    n = size(X, 1)
    assignments = zeros(Int, n)
    converged = false
    iter = 0
    for it in 1:max_iter
        iter = it
        changed = false
        for i in 1:n
            dists = [sum(abs2, view(X, i, :) .- view(centers, c, :)) for c in 1:k]
            c_new = argmin(dists)
            if assignments[i] != c_new
                assignments[i] = c_new
                changed = true
            end
        end
        new_centers = copy(centers)
        for c in 1:k
            idx = findall(==(c), assignments)
            isempty(idx) || (new_centers[c, :] = vec(mean(X[idx, :]; dims = 1)))
        end
        centers = new_centers
        if !changed && it > 1
            converged = true
            break
        end
    end
    withinss = zeros(k)
    for c in 1:k
        idx = findall(==(c), assignments)
        isempty(idx) || (withinss[c] = sum(sum(abs2, X[i, :] .- centers[c, :]) for i in idx))
    end
    return centers, assignments, withinss, sum(withinss), iter, converged
end

"""
    kmeans_cluster(X::AbstractMatrix{<:Real}, k::Int; max_iter = 100, n_init = 10,
                   init = nothing, rng = Random.default_rng()) -> KMeansFit

K-means clustering of the rows of `X` into `k` clusters via Lloyd's
algorithm, using k-means++ initialization and `n_init` restarts (keeping
the run with the lowest total within-cluster sum of squares) unless
explicit starting `centers` are given via `init` (a `k x size(X,2)`
matrix), in which case a single deterministic run is performed.

# Formula

Lloyd's algorithm alternates, until assignments stop changing:

```
assign each point to the nearest center (Euclidean distance)
recompute each center as the mean of its assigned points
```

minimizing total within-cluster sum of squares
`Σₖ Σᵢ∈cluster k ‖xᵢ - centerₖ‖²`. k-means++ initialization picks the
first center uniformly at random, then each subsequent center from the
remaining points with probability proportional to its squared distance
to the nearest already-chosen center — this spreads the initial centers
out and gives better expected quality than uniform random init.

# References

- Lloyd, S. P. (1982). Least squares quantization in PCM. *IEEE
  Transactions on Information Theory*, 28(2), 129–137. (written 1957,
  published 1982)
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of
  careful seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on
  Discrete Algorithms*, 1027–1035.
"""
function kmeans_cluster(X::AbstractMatrix{<:Real}, k::Int; max_iter::Int = 100, n_init::Int = 10,
                         init::Union{Nothing,AbstractMatrix} = nothing,
                         rng::AbstractRNG = Random.default_rng())
    n, p = size(X)
    k >= 1 || throw(ArgumentError("k must be >= 1"))
    n >= k || throw(ArgumentError("need at least k observations for k=$k clusters"))
    Xf = Float64.(X)

    n_runs = init === nothing ? n_init : 1
    best = nothing
    for _ in 1:n_runs
        start = init === nothing ? _kmeans_pp_init(Xf, k, rng) : Matrix{Float64}(init)
        centers, assignments, withinss, tot_withinss, iterations, converged =
            _kmeans_run(Xf, start, k, max_iter)
        if best === nothing || tot_withinss < best[4]
            best = (centers, assignments, withinss, tot_withinss, iterations, converged)
        end
    end
    centers, assignments, withinss, tot_withinss, iterations, converged = best

    grand_mean = vec(mean(Xf; dims = 1))
    totss = sum(sum(abs2, Xf[i, :] .- grand_mean) for i in 1:n)
    betweenss = totss - tot_withinss

    return KMeansFit(centers, assignments, withinss, totss, tot_withinss, betweenss,
                      iterations, converged, k, n, DataFrame(), Symbol[])
end

"""
    kmeans_cluster(df::AbstractDataFrame, k::Int, cols::Symbol...; kwargs...) -> KMeansFit

Cluster the given columns of `df` (or all columns if `cols` is empty).
"""
function kmeans_cluster(df::AbstractDataFrame, k::Int, cols::Symbol...; kwargs...)
    X, cs = _numeric_matrix(df, cols)
    fit = kmeans_cluster(X, k; kwargs...)
    return KMeansFit(fit.centers, fit.assignments, fit.withinss, fit.totss, fit.tot_withinss,
                      fit.betweenss, fit.iterations, fit.converged, k, fit.n, DataFrame(df), cs)
end

"""
    kmeans_cluster(k::Int, cols::Symbol...; kwargs...)

Pipe-curried form: `df |> kmeans_cluster(3, :x1, :x2)`.
"""
kmeans_cluster(k::Int, cols::Symbol...; kwargs...) = df -> kmeans_cluster(df, k, cols...; kwargs...)

function ForsetiCore.tidy(fit::KMeansFit)
    cs = isempty(fit.cols) ? ["x$i" for i in 1:size(fit.centers, 2)] : string.(fit.cols)
    sizes = [count(==(c), fit.assignments) for c in 1:fit.k]
    out = DataFrame(cluster = collect(1:fit.k), size = sizes)
    for (j, name) in enumerate(cs)
        out[!, Symbol(name)] = fit.centers[:, j]
    end
    out[!, :withinss] = fit.withinss
    return out
end

function ForsetiCore.glance(fit::KMeansFit)
    return DataFrame(totss = [fit.totss], tot_withinss = [fit.tot_withinss],
                      betweenss = [fit.betweenss], iterations = [fit.iterations],
                      converged = [fit.converged])
end

function ForsetiCore.augment(fit::KMeansFit, data::AbstractDataFrame = fit.data)
    out = DataFrame(data)
    out[!, :cluster] = fit.assignments
    return out
end

# ---------------------------------------------------------------------------
# One-way MANOVA (Wilks' Lambda)
# ---------------------------------------------------------------------------

"""
    ManovaFit <: ForsetiFit

Result of [`manova`](@ref). An exact F-transform of Wilks' Lambda is used
when `p == 1` (reduces to univariate ANOVA) or when there are exactly 2
groups (`dfH == 1`, the Hotelling's T² case); otherwise Bartlett's
large-sample chi-square approximation is used (`exact == false`).
"""
struct ManovaFit <: ForsetiFit
    wilks_lambda::Float64
    statistic::Float64
    statistic_type::Symbol   # :F or :chisq
    df1::Float64
    df2::Union{Float64,Missing}
    p_value::Float64
    df_hypothesis::Int
    df_error::Int
    p::Int
    g::Int
    exact::Bool
    method::String
end

"""
    manova(Y::AbstractMatrix{<:Real}, groups::AbstractVector) -> ManovaFit

One-way MANOVA testing whether the mean vector of the columns of `Y`
differs across the levels of `groups`, via Wilks' Lambda.

# Formula

```
B = Σₖ nₖ(x̄ₖ - x̄)(x̄ₖ - x̄)ᵀ    (between-groups SSCP matrix)
W = Σₖ Σᵢ (xᵢₖ - x̄ₖ)(xᵢₖ - x̄ₖ)ᵀ  (within-groups SSCP matrix)
Λ = det(W) / det(W + B)          (Wilks' Lambda)
```

Two of Λ's exact F-transforms are used here (`exact = true`): `p = 1`
(reduces algebraically to the univariate [`anova`](@ref) F-statistic),
and `dfH = g-1 = 1` (two groups), the latter derived directly from the
Hotelling's `T²` two-sample relationship `T² = dfE(1-Λ)/Λ` and
`F = (dfE-p+1)/(p·dfE) · T²`. Other combinations of `p`/group count fall
back to Bartlett's large-sample chi-square approximation (`exact =
false`): `χ² = -(dfE - (p-dfH+1)/2) ln Λ ~ χ²(p·dfH)`.

# References

- Wilks, S. S. (1932). Certain generalizations in the analysis of
  variance. *Biometrika*, 24(3/4), 471–494.
- Hotelling, H. (1931). The generalization of Student's ratio. *Annals
  of Mathematical Statistics*, 2(3), 360–378. (`T²`, the 2-group case)
- Bartlett, M. S. (1938). Further aspects of the theory of multiple
  regression. *Mathematical Proceedings of the Cambridge Philosophical
  Society*, 34(1), 33–40. (chi-square approximation)
"""
function manova(Y::AbstractMatrix{<:Real}, groups::AbstractVector)
    n, p = size(Y)
    length(groups) == n || throw(ArgumentError("groups must have one entry per row of Y"))
    levels = unique(groups)
    g = length(levels)
    g >= 2 || throw(ArgumentError("manova requires at least 2 groups"))

    Yf = Float64.(Y)
    grand_mean = vec(mean(Yf; dims = 1))
    B = zeros(p, p)
    W = zeros(p, p)
    for lev in levels
        idx = findall(==(lev), groups)
        Yg = Yf[idx, :]
        gm = vec(mean(Yg; dims = 1))
        d = gm .- grand_mean
        B .+= length(idx) .* (d * d')
        for i in idx
            dv = Yf[i, :] .- gm
            W .+= dv * dv'
        end
    end

    wilks_lambda = det(W) / det(W + B)
    df_hyp = g - 1
    df_err = n - g

    if p == 1
        statistic = (df_err / df_hyp) * (1 - wilks_lambda) / wilks_lambda
        df1, df2 = Float64(df_hyp), Float64(df_err)
        pval = ccdf(FDist(df1, df2), statistic)
        return ManovaFit(wilks_lambda, statistic, :F, df1, df2, pval, df_hyp, df_err, p, g,
                          true, "One-way MANOVA (Wilks' Lambda, exact F, p=1 reduces to ANOVA)")
    elseif df_hyp == 1
        statistic = ((df_err - p + 1) / p) * (1 - wilks_lambda) / wilks_lambda
        df1, df2 = Float64(p), Float64(df_err - p + 1)
        pval = ccdf(FDist(df1, df2), statistic)
        return ManovaFit(wilks_lambda, statistic, :F, df1, df2, pval, df_hyp, df_err, p, g,
                          true, "One-way MANOVA (Wilks' Lambda, exact F, 2 groups)")
    else
        statistic = -(df_err - (p - df_hyp + 1) / 2) * log(wilks_lambda)
        df1 = Float64(p * df_hyp)
        pval = ccdf(Chisq(df1), statistic)
        return ManovaFit(wilks_lambda, statistic, :chisq, df1, missing, pval, df_hyp, df_err,
                          p, g, false, "One-way MANOVA (Wilks' Lambda, Bartlett chi-square approximation)")
    end
end

"""
    manova(df::AbstractDataFrame, group_col::Symbol, response_cols::Symbol...) -> ManovaFit

Run one-way MANOVA of the given response columns across the levels of
`group_col` in `df`.
"""
function manova(df::AbstractDataFrame, group_col::Symbol, response_cols::Symbol...)
    Y, _ = _numeric_matrix(df, response_cols)
    return manova(Y, df[!, group_col])
end

"""
    manova(group_col::Symbol, response_cols::Symbol...)

Pipe-curried form: `df |> manova(:group, :y1, :y2)`.
"""
manova(group_col::Symbol, response_cols::Symbol...) = df -> manova(df, group_col, response_cols...)

function ForsetiCore.tidy(fit::ManovaFit)
    return DataFrame(
        wilks_lambda = [fit.wilks_lambda],
        statistic = [fit.statistic],
        statistic_type = [fit.statistic_type],
        df1 = [fit.df1],
        df2 = [fit.df2],
        p_value = [fit.p_value],
        exact = [fit.exact],
    )
end

function ForsetiCore.glance(fit::ManovaFit)
    return DataFrame(
        wilks_lambda = [fit.wilks_lambda],
        statistic = [fit.statistic],
        p_value = [fit.p_value],
        df_hypothesis = [fit.df_hypothesis],
        df_error = [fit.df_error],
        p = [fit.p],
        g = [fit.g],
        method = [fit.method],
    )
end

end # module ForsetiMultivariate
