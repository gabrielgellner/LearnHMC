# # Stan Economics Tutorial
# http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/IntroToStan_basics_workflow.ipynb

using Distributions
using Parameters
using DynamicHMC
using ContinuousTransformations
using MCMCDiagnostics
using DiffWrappers
using PyPlot

# Generate the data from a more complex model
function dgp_rng(X::Matrix, β::Vector, ν, σ)
    y = fill(0.0, size(X, 1))
    for n in 1:size(X, 1)
        #TODO: do a pull request to add the non-centered version
        y[n] = σ * (rand(TDist(ν))) + sum(X[n, :] .* β)
    end
    return y
end

#TODO: add random data
# Number of observations
N = 1000
# Number of covariates
P = 10
X = rand(Normal(), N, P)

# Degrees of freedom for Cauchy noise (\nu)
ν = 5.0
# scale parameter
σ_true = 5.0
# generate some random coefficients that we'll try to recover
β_true = rand(Normal(), 10)
# Make sure the first element of  eta is positive as in our chosen DGP
β_true[1] = abs(β_true[1])
# fixed values
β_true = [1.16, -0.19, -0.29, -0.39, 0.7, -1.6, 0.77, -1.29, 0.89, -0.17]

y_sim = dgp_rng(X, β_true, ν, σ_true)
begin
    figure()
    plt[:hist](y_sim, bins = 30)
end

struct EcoRegProblem
    X::Matrix{Float64}
    data::Vector{Float64}
end

function (prob::EcoRegProblem)(θ)
    β, σ = θ

    # Parameter Priors
    βp = Normal(0, 5)
    σp = Cauchy(0, 2.5)
    #β1, βrest, σ = θ
    #β = vcat(β1, βrest)
    llike = logpdf(σp, σ)
    for i = 1:length(β)
        llike += logpdf(βp, β[i])
    end

    # Likelihood
    # y ~ normal(x*beta, sigma)
    for (i, μ) in enumerate(prob.X * β)
        llike += logpdf(Normal(μ, σ), prob.data[i])
    end

    return llike
end

# # "Incorrect" Model
prob = EcoRegProblem(X, y_sim)
θ_transform = TransformationTuple(ArrayTransformation(IDENTITY, P), bridge(ℝ, ℝ⁺))
#TODO: the first element of β needs to be positive
tprob = TransformLogLikelihood(prob, θ_transform)
tprob∇ = ForwardGradientWrapper(tprob, fill(0.0, length(tprob)))

@time tprob∇(randn(P + 1))

#TODO: this seems to much slower than stan
iter = 2000
@time chain1, NUTS_tuned = NUTS_init_tune_mcmc(tprob∇, fill(1.1, P + 1), iter)
@time chain2, NUTS_tuned = NUTS_init_tune_mcmc(tprob∇, fill(0.1, P + 1), iter)
# Thin
chain1 = chain1[floor(Int, iter / 2):end]
chain2 = chain2[floor(Int, iter/ 2):end]
samp = vcat(chain1, chain2)

# # # "Correct" Model
# θ_transform = TransformationTuple(bridge(ℝ, ℝ⁺),                     # β₁
# ArrayTransformation(IDENTITY, P - 1),  # β rest
# bridge(ℝ, ℝ⁺))                     # σ

NUTS_statistics(samp)
ESS = squeeze(mapslices(effective_sample_size, ungrouping_map(Array, get_position, samp), 1), 1)

# Note the use of ungrouping_map, a utility function that maps the collects posterior results,
# and groups then in tuples of vectors or arrays. You could do this manually, but you have to
# use get_position to extract the posterior position for each point. Also, in order to do inference,
# you would want to transform the "raw" values in $\mathbb{R}^n$ using the parameter transformation.
#β1, βrest, σ = ungrouping_map(Array, θ_transform ∘ get_position, samp)
#β = hcat(β1, βrest)
β, σ = ungrouping_map(Array, θ_transform ∘ get_position, samp)

# Posterior means are fairly close to the parameters (note that they will not be the same,
# because of sampling variation):
for icol = 1:size(β, 2)
    @show mean(β[:, icol]), β_true[icol]
end
mean(σ), σ_true

for icol = 1:size(β, 2)
    @show abs(mean(β[:, icol]) - β_true[icol])
end
mean(σ), σ_true

# # Plot some results
# ## Posterior
function plot_posterior(known_value, posterior_draws, varname; nbins = 20)
    #postr = fit(Histogram, posterior_draws; closed = :right, args...)
    plt[:hist](posterior_draws, bins = nbins, label = "posterior")
    xlabel(varname)
    axvline([known_value], label = "known value", c = "red")
    #legend()
end

begin
    figure()
    for i = 1:size(β, 2)
        subplot(3, 4, i)
        plot_posterior(β_true[i], β[:, i], "β$i"; nbins = 20)
    end
    subplot(3, 4, P + 1)
    plot_posterior(σ_true, σ, "σ"; nbins = 20)
    tight_layout()
end

# ## MCMC sampling
begin
    figure()
    for i = 1:size(β, 2)
        subplot(3, 4, i)
        plot(β[:, i], label = "β$i")
        axhline([β_true[i]], c = "red")
    end
    tight_layout()
end
