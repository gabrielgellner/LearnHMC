# # Implement this example
# https://theoreticalecology.wordpress.com/2010/09/17/metropolis-hastings-mcmc-in-r/
using Distributions
using StatsBase
using Parameters
using DynamicHMC
using ContinuousTransformations
using MCMCDiagnostics
using DiffWrappers
using PyPlot

# # Creating test data
true_a = 5.0
true_b = 0.0
true_sd = 10.0
sample_size = 31

# create independent x-values
x = -(sample_size - 1) / 2 : (sample_size - 1) / 2
# create dependent values according to ax + b + N(0, sd)
y = true_a .* x + true_b + rand(Normal(0, true_sd), sample_size)
plot(x, y, "o")

struct LinRegProblem{T}
    x::Vector{T}
end

function likelihood(prob::LinRegProblem, θ)
    a, b, sd = θ
    return sum(logpdf(Normal(a * prob.x[i] + b, sd), y[i]) for i in 1:length(y))
end

function priors(θ)
    a, b, sd = θ
    return logpdf(Uniform(0, 10), a) + logpdf(Normal(0, 5), b) + logpdf(Uniform(0, 30), sd)
end

prob = LinRegProblem(collect(x))
likelihood(prob, [true_a, true_b, true_sd])

slope_values(x) = likelihood(prob, [x, true_b, true_sd])
slope_likelihoods = slope_values.(3:0.05:7)
begin
    plot(3:0.05:7, slope_likelihoods)
    xlabel("values of slope parameter a")
    ylabel("Log Likelihood")
end

"posterier of `LinRegProblem`"
function (prob::LinRegProblem)(θ)
    return likelihood(prob, θ) + priors(θ)
end

prob([true_a, true_b, true_sd])

# for the priors I will want:
# in the domain for the Uniform priors
# All of ℝ for the Normal
θ_transform = TransformationTuple((bridge(ℝ, ℝ), bridge(ℝ, ℝ), bridge(ℝ, ℝ⁺)))
tprob = TransformLogLikelihood(prob, θ_transform)
tprob([true_a, true_b, true_sd])

tprob∇ = ForwardGradientWrapper(tprob, fill(0.0, length(tprob)))
tprob∇(ones(length(tprob∇)))

samp, NUTS_tuned = NUTS_init_tune_mcmc(tprob∇, fill(0.0, length(tprob)), 20000)

NUTS_statistics(samp)
ESS = squeeze(mapslices(effective_sample_size, ungrouping_map(Array, get_position, samp), 1), 1)

# Note the use of ungrouping_map, a utility function that maps the collects posterior results,
# and groups then in tuples of vectors or arrays. You could do this manually, but you have to
# use get_position to extract the posterior position for each point. Also, in order to do inference,
# you would want to transform the "raw" values in $\mathbb{R}^n$ using the parameter transformation.
(a, b, sd) = ungrouping_map(Array, θ_transform ∘ get_position, samp)

# Posterior means are fairly close to the parameters (note that they will not be the same,
# because of sampling variation):
mean(a[5000:end]), true_a
mean(b[500:end]), true_b
mean(sd[5000:end]), true_sd

median(a), true_a
median(b), true_b
median(sd), true_sd

linreg(x, y)

# # Plot some results
# ## Posterior
function plot_posterior(known_value, posterior_draws, varname; nbins = 20)
    #postr = fit(Histogram, posterior_draws; closed = :right, args...)
    plt[:hist](posterior_draws, bins = nbins, label = "posterior")
    xlabel(varname)
    axvline([known_value], label = "known value", c = "red")
    legend()
end

begin
    figure()
    subplot(3, 1, 1)
    plot_posterior(true_a, a, L"a"; nbins = 20)
    subplot(3, 1, 2)
    plot_posterior(true_b, b, L"b"; nbins = 20)
    subplot(3, 1, 3)
    plot_posterior(true_sd, sd, L"\sigma"; nbins = 20)
    tight_layout()
end

# ## MCMC sampling
begin
    figure()
    subplot(3, 1, 1)
    plot(a, label = L"a")
    axhline([true_a], c = "red")
    legend()
    subplot(3, 1, 2)
    plot(b, label = L"b")
    axhline([true_b], c = "red")
    legend()
    subplot(3, 1, 3)
    plot(sd, label = L"\sigma")
    axhline([true_sd], c = "red")
    legend()
    tight_layout()
end
