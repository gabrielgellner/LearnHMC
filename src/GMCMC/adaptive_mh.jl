using Parameters

@with_kw mutable struct AdaptiveMHSampler
    # `m`: dimension
    dim::Int
    # `scale_est`, estimated covariance matrix
    # `μ_est`, estimated mean
    # `restart_at`, last iteration that restarted the adaptive search
    # `initruns1`, how many iterations before starting the adaptive search
    # `theta`, estimated to give wanted acceptance rate
    # `theta0`, initial theta
    # `dim`, how many parameters that are updated in the Gaussian RW
    # `c_star`, step length constant when searching for theta. Use `get_cstar_for_multivar` to calculate it.
    # `p_star`, wanted acceptance rate
    p_star::Float64
    # `p_accept`, probability of accepting the proposed values of the MCMC
    # `n0`, to initially reduce to large step length in the search for theta. n0=round(5/(pstar*(1-pstar)));
    # `nrrestarts`, how many times the search for theta has been restarted.
    # `include_restarts`, true if the option of restarting the search should be included, otherwise false
    # `maxrestarts`, maximum number of possible restarts
    # `A`, covariance matrix to be used for proposals
end

"""
# Arguments
* `pstar`: wanted acceptance probability
* `m`: dimension
"""
function get_cstar_for_multivar(pstar, m)
    α = -quantile(Normal(), pstar / 2)
    cstar = (1 - 1 / m) * (2 * π) ^ 0.5 * exp(α ^ 2 / 2) * ((2 * α) ^ (-1)) + (m * pstar * (1 - pstar)) ^ (-1)
    return cstar
end

"""
# Arguments
* `σ0`: ?
* `x`: new data point, column vector
* `μ0`: previous mean, column vector
"""
function get_SIGMA_for_multivar_adaptive(σ0, t, μ0::Vector, x::Vector)
    μ1 = t ^ (-1) * ((t - 1) * μ0 + x)
    σ1 = (t - 2) * ((t - 1) ^ (-1)) * σ0 + μ0 * μ0' - t * ((t - 1) ^ (-1)) * μ1 * μ1' + ((t - 1) ^ (-1)) * x * x'
    return @NT(μ = μ1, σ = σ1)
end

#TODO: rename to `propose_theta`?
function theta_for_adaptive_multivar_proposal(θ, pstar, cstar, accprob, t, m)
    divby = max(200, t * m ^ (-1)) #TODO: why 200, should this be an optional parameter?
    θ = θ + cstar * (accprob - pstar) * divby ^ -1
    return θ
end

#TODO: names like SIGMA and MEAN seem unnessary to me, and give the wrong impression that
#      these are constants. Rationalize the naming.
"""
# Arguments
* `X`, current values of parameters in the MCMC chain
* `t`, iteration of the MCMC
* `SIGMAest`, estimated covariance matrix
* `MEANest`, estimated mean
* `restart_at`, last iteration that restarted the adaptive search
* `initruns1`, how many iterations before starting the adaptive search
* `theta`, estimated to give wanted acceptance rate
* `theta0`, initial theta
* `dim`, how many parameters that are updated in the Gaussian RW
* `cstar`, step length constant when searching for theta. Use `get_cstar_for_multivar` to calculate it.
* `pstar`, wanted acceptance rate
* `accprob`, probability of accepting the proposed values of the MCMC
* `n0`, to initially reduce to large step length in the search for theta. n0=round(5/(pstar*(1-pstar)));
* `nrrestarts`, how many times the search for theta has been restarted.
* `include_restarts`, true if the option of restarting the search should be included, otherwise false
* `maxrestarts`, maximum number of possible restarts
* `A`, covariance matrix to be used for proposals

# Example inputs to run:
```julia
out = adaptive_mcmc(X=1:3, chain_step=2345, SIGMAest=,MEANest,restart_at,initruns1,theta,theta0,dim,cstar,pstar,accprob,n0,nrrestarts,include_restarts,maxrestarts)
```
"""
#NOTE: this just does a single step
function adaptive_mcmc(X,
                       chain_step,
                       S,
                       IGMAest,
                       MEANest,
                       restart_at,
                       initruns1,
                       theta,
                       theta0,
                       dim::Int,
                       cstar,
                       pstar,
                       accprob,
                       n0,
		               nrrestarts,
                       include_restarts::Bool,
                       max_restarts::Int)
    tt = chain_step - restart_at
    ttn0 = tt + n0
    t0 = initruns1 + restart_at
    if chain_step <= initruns1
        # run get_SIGMA_for_multivar_adaptive function
        MS = get_SIGMA_for_multivar_adaptive(SIGMAest, chain_step, MEANest, X)
        sig2 = exp(2 * theta)
        A = diag(dim) + sig2 * chain_step ^ -1 * diag(dim)
        A = sig2 * A
    else
        MS = get_SIGMA_for_multivar_adaptive(SIGMAest, chain_step, MEANest, X)
        tmod = max(200, ttn0 * dim ^ -1)
        theta = theta_for_adaptive_multivar_proposal(theta, pstar, cstar, accprob, tmod, dim)
        sig2 = exp(2 * theta)
        A = MS[:σ] + sig2 * chain_step ^ -1 * diag(dim)
        A = sig2 * A
    end

    if include_restarts
	    if (abs(theta - theta0) > 1.1) && (maxrestarts > nrrestarts)
	        theta0 = theta
	        restart_at = chain_step
	        nrrestarts += 1
        end
    end

    #TODO: return a NamedTuple?
    return Dict(:σ => MS[:σ], :μ => MS[:μ], :Θ => theta, :A => A,:Θ0 => theta0, :restart_at => restart_at, :nrrestarts => nrrestarts)
end
