using Test
using ForsetiCore
using ForsetiMultivariate
using DataFrames
using Statistics
using LinearAlgebra
using Random
using Distributions

@testset "ForsetiMultivariate" begin

    @testset "pca: degenerate (perfectly correlated) data" begin
        x = [1.0, 2.0, 3.0, 4.0]
        y = [2.0, 4.0, 6.0, 8.0]  # y = 2x exactly -> rank-1 after centering
        X = hcat(x, y)
        fit = pca(X)
        @test fit isa PCAFit
        @test fit isa ForsetiFit

        # sign-agnostic reconstruction check: scores * loadings' recovers centered data
        Xc = X .- mean(X; dims = 1)
        @test fit.scores * fit.loadings' ≈ Xc atol = 1e-8

        # hand-derived: eigenvalues of cov matrix are 25/3 and 0
        @test fit.var_explained ≈ [1.0, 0.0] atol = 1e-8
        @test fit.sdev[1] ≈ sqrt(25.0 / 3.0) atol = 1e-8
        @test fit.sdev[2] ≈ 0.0 atol = 1e-8
    end

    @testset "pca: DataFrame form, tidy matrix kwarg, glance, augment" begin
        df = DataFrame(a = [1.0, 2.0, 3.0, 5.0], b = [2.0, 1.0, 4.0, 3.0], c = [5.0, 3.0, 6.0, 2.0])
        fit = df |> pca(:a, :b, :c) |> identity
        @test sum(fit.var_explained) ≈ 1.0 atol = 1e-10

        t_load = tidy(fit)  # default matrix = :loadings
        @test Set(t_load.term) == Set(["a", "b", "c"])
        @test nrow(t_load) == 3 * length(fit.sdev)

        t_eig = tidy(fit; matrix = :eigenvalues)
        @test t_eig.percent ≈ fit.var_explained atol = 1e-10
        @test t_eig.cumulative[end] ≈ 1.0 atol = 1e-10

        t_scores = tidy(fit; matrix = :scores)
        @test size(t_scores) == (4, length(fit.sdev))

        @test_throws ArgumentError tidy(fit; matrix = :bogus)

        g = glance(fit)
        @test g.n_obs[1] == 4
        @test g.n_variables[1] == 3

        a = augment(fit)
        @test a.a == df.a
        @test a.PC1 ≈ fit.scores[:, 1] atol = 1e-10
    end

    @testset "kmeans_cluster: deterministic run with explicit init centers" begin
        # two well-separated clusters
        X = [0.0 0.0; 1.0 0.0; 0.0 1.0; 10.0 10.0; 11.0 10.0; 10.0 11.0]
        init = [0.0 0.0; 10.0 10.0]
        fit = kmeans_cluster(X, 2; init = init)
        @test fit isa KMeansFit
        @test fit isa ForsetiFit
        @test fit.converged

        @test fit.assignments == [1, 1, 1, 2, 2, 2]
        @test fit.centers ≈ [1/3 1/3; 31/3 31/3] atol = 1e-10
        @test fit.tot_withinss ≈ 8.0 / 3.0 atol = 1e-10
        @test fit.totss ≈ fit.tot_withinss + fit.betweenss atol = 1e-10  # SS decomposition identity

        t = tidy(fit)
        @test nrow(t) == 2
        @test sort(t.size) == [3, 3]

        g = glance(fit)
        @test g.tot_withinss[1] ≈ fit.tot_withinss

        # DataFrame/pipe form + augment
        df = DataFrame(x = X[:, 1], y = X[:, 2])
        a = df |> kmeans_cluster(2, :x, :y; init = init) |> augment
        @test a.cluster == [1, 1, 1, 2, 2, 2]
    end

    @testset "kmeans_cluster: random init still converges and is reproducible with a seeded rng" begin
        X = [0.0 0.0; 1.0 0.0; 0.0 1.0; 10.0 10.0; 11.0 10.0; 10.0 11.0]
        fit1 = kmeans_cluster(X, 2; rng = MersenneTwister(42))
        fit2 = kmeans_cluster(X, 2; rng = MersenneTwister(42))
        @test fit1.assignments == fit2.assignments
        @test fit1.tot_withinss ≈ 8.0 / 3.0 atol = 1e-8  # global optimum for this well-separated data

        @test_throws ArgumentError kmeans_cluster(X, 10)  # more clusters than points
    end

    @testset "manova: exact F, 2 groups (dfH = 1, Hotelling's T^2 case)" begin
        Y = [1.0 2.0; 2.0 4.0; 3.0 3.0; 5.0 7.0; 6.0 9.0; 7.0 8.0]
        groups = ["A", "A", "A", "B", "B", "B"]
        fit = manova(Y, groups)
        @test fit isa ManovaFit
        @test fit isa ForsetiFit
        @test fit.exact
        @test fit.statistic_type == :F

        # hand-derived: Wilks' Lambda = det(W)/det(W+B) = 12/138 = 2/23
        @test fit.wilks_lambda ≈ 12.0 / 138.0 atol = 1e-10
        @test fit.statistic ≈ 15.75 atol = 1e-8
        @test fit.df1 == 2.0
        @test fit.df2 == 3.0
        @test fit.p_value ≈ ccdf(FDist(2, 3), 15.75) atol = 1e-12
    end

    @testset "manova: exact F, p = 1 reduces to univariate ANOVA" begin
        values = reshape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], :, 1)
        groups = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        fit = manova(values, groups)
        @test fit.exact
        # hand-derived (matches the ForsetiHypothesis anova example): F = 27, df = (2, 6)
        @test fit.statistic ≈ 27.0 atol = 1e-8
        @test fit.df1 == 2.0
        @test fit.df2 == 6.0

        df = DataFrame(value = vec(values), group = groups)
        fit2 = df |> manova(:group, :value) |> identity
        @test fit2.statistic ≈ 27.0 atol = 1e-8
    end

    @testset "manova: Bartlett chi-square approximation, general case" begin
        Y = [1.0 2.0; 2.0 4.0; 3.0 3.0; 5.0 7.0; 6.0 9.0; 7.0 8.0; 9.0 10.0; 10.0 12.0; 11.0 11.0]
        groups = ["A", "A", "A", "B", "B", "B", "C", "C", "C"]
        fit = manova(Y, groups)
        @test !fit.exact
        @test fit.statistic_type == :chisq
        @test fit.df1 == 4.0  # p * dfH = 2 * 2
        @test 0.0 < fit.p_value < 1.0

        # internal consistency: recompute Bartlett's statistic from the reported Lambda
        p, dfH, dfE = fit.p, fit.df_hypothesis, fit.df_error
        expected = -(dfE - (p - dfH + 1) / 2) * log(fit.wilks_lambda)
        @test fit.statistic ≈ expected atol = 1e-10

        t = tidy(fit)
        @test t.statistic_type[1] == :chisq
        g = glance(fit)
        @test g.g[1] == 3
    end

    @testset "manova: requires at least 2 groups" begin
        Y = [1.0 2.0; 2.0 3.0]
        @test_throws ArgumentError manova(Y, ["A", "A"])
    end

end
