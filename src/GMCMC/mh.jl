using Distributions
using PyPlot

function mh_mcmc(g, x0; iter = 100, scale = 0.9)
    xs = fill(0.0, length(x0), iter)
    xs[:, 1] = x0
    for t = 2:iter
        xstar = rand(MvNormal(xs[:, t - 1], scale))
        if g(xstar) - g(xs[:, t - 1]) > log(rand())
            xs[:, t] = xstar
        else
            xs[:, t] = xs[:, t - 1]
        end
    end
    return xs
end

llike(x) = logpdf(Gamma(1, 2), x[1]) + logpdf(Gamma(9, 0.5), x[2]) + logpdf(Uniform(1, 2), x[3])

iter = 10000
samps = mh_mcmc(llike, [0.5, 0.1, 0.1]; iter = iter, scale = [1.3, 1.3, 0.5])
plot(samps'[floor(Int, iter / 2):end, :])

plt[:hist](samps[1, floor(Int, iter / 2):end], bins = 20, density = true)
plt[:hist](samps[2, floor(Int, iter / 2):end], bins = 20, density = true)
plt[:hist](samps[3, floor(Int, iter / 2):end], bins = 20, density = true)
