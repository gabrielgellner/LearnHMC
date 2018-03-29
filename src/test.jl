# # Stan Economics Tutorial
# http://nbviewer.jupyter.org/github/QuantEcon/QuantEcon.notebooks/blob/master/IntroToStan_basics_workflow.ipynb

using Distributions
using Parameters
using DynamicHMC
using ContinuousTransformations
using MCMCDiagnostics
using DiffWrappers
using PyPlot

# It is convenient to define a structure that holds the data,
struct EcoRegProblem
    "Quantity"
    X::Matrix{Float64}
    σ::Float64
    data::Vector{Float64}
end

function (problem::EcoRegProblem)(θ)
    β = θ[1:2]
    @unpack X, σ, data = problem
    llike = 1.0
    for (i, μ) in enumerate(X * β)
        llike += logpdf(Normal(μ, σ), data[i])
    end
    return llike + logpdf(Uniform(0, 5), β[1]) + logpdf(Uniform(0, 5), β[2])
end

sim_data = [1 2; 3 4] * [0.5, 2.2]

prob = EcoRegProblem([1 2; 3 4], 0.1, sim_data)
prob([0.5, 2.1])

#TODO: I need to add transforamtions so that the β can not go outside of (0, 5)

prob∇ = ForwardGradientWrapper(prob, [0.0, 0.0])
prob∇(ones(length(prob∇)))
samp, NUTS_tuned = NUTS_init_tune_mcmc(prob∇, [1, 1.0], 1000)

NUTS_statistics(samp)
