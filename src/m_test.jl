include("GMCMC/metropolis.jl")

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
