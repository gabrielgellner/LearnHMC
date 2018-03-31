# In R:
# Load necessary libraries and set up multi-core processing for Stan
options(warn=-1, message =-1)
library(dplyr) 
library(ggplot2)
library(rstan)
library(reshape2)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# In R, or you could save the contents of the string in a file with .stan file type

dgp_string <- "

functions {
    /**
   * Return draws from a linear regression with data matrix X,
   * coefficients beta, and student-t noise with degrees of freedom nu
   * and scale sigma.
   *
   * @param X Data matrix (N x P)
   * @param beta Coefficient vector (P x 1)
   * @param nu Residual distribution degrees of freedom.
   * @param sigma Residual distribution scale.
   * @return Return an N-vector of draws from the model.
   */

    vector dgp_rng(matrix X, vector beta, real nu, real sigma) {
      vector[rows(X)] y; // define the output vector to be as long as the number of rows in X
      
      // Now fill it in
      for (n in 1:rows(X))
        y[n] = student_t_rng(nu, X[n] * beta, sigma);
      return y;
   }
}
data {
 // If we were estimating a model, we'd define the data inputs here
}
parameters {
  // ... and the parameters we want to estimate would go in here
}
model {
  // This is where the probability model we want to estimate would go
}
"

# Generate a matrix of random numbers, and values for beta, nu and sigma

set.seed(42) # Set the random number generator seed so that we get the same parameters
N <- 1000 # Number of observations
P <- 10 # Number of covariates
X <- matrix(rnorm(N*P), N, P) # generate an N*P covariate matrix of random data
nu <- 5 # Set degrees of freedom
sigma <- 5 # And scale parameter
beta <- rnorm(10) # Generate some random coefficients that we'll try to recover
# Make sure the first element of beta is positive as in our chosen DGP
beta[1] <- abs(beta[1])

# Compile the script
compiled_function <- stan_model(model_code = dgp_string) # you could use file = "path/to/yourfile.stan" if you have saved it as so
# And make the function available to the user in R
expose_stan_functions(compiled_function)

# Draw a vector of random numbers for known Xs and parameters
y_sim <- dgp_rng(nu = nu, X = X, sigma = sigma, beta = beta)

# Plot the data
data_frame(y_sim = y_sim) %>% # Declare a data frame and pipe it into a ggplot
  ggplot(aes( x = y_sim)) + # Where we state the x-axis aesthetic (our simulated values)
  geom_histogram(binwidth = 3) # And tell ggplot what sort of chart to build

# In R, or in your .stan file (contents from within the quotes only)

incorrect_model <- "
data {
  // In this section, we define the data that must be passed to Stan (from whichever environment you are using)

  int N; // number of observations
  int P; // number of covariates
  matrix[N, P] X; //covariate matrix
  vector[N] y; //outcome vector
}
parameters {
  // Define the parameters that we will estimate, as well as any restrictions on the parameter values (standard deviations can't be negative...)

  vector[P] beta; // the regression coefficients
  real<lower = 0> sigma; // the residual standard deviation (note that it's restricted to be non-negative)
}
model {
  // This is where we write out the probability model, in very similar form to how we would using paper and pen

  // Define the priors
  beta ~ normal(0, 5); // same prior for all betas; we could define a different one for each, or use a multivariate prior
  sigma ~ cauchy(0, 2.5);
  
  // The likelihood
  y ~ normal(X*beta, sigma);
}
generated quantities {
  // For model comparison, we'll want to keep the likelihood contribution of each point
  // We will also generate posterior predictive draws (yhat) for each data point. These will be elaborated on below. 
  
  vector[N] log_lik;
  vector[N] y_sim;
  for(i in 1:N){
    log_lik[i] = normal_lpdf(y[i] | X[i,]*beta, sigma);
    y_sim[i] = normal_rng(X[i,]*beta, sigma);
  }
}
"

# In R, or in your .stan file (contents from within the quotes only)

correct_model <- "
data {
  int N; // number of observations
  int P; // number of covariates
  matrix[N, P] X; //covariate matrix
  vector[N] y; //outcome vector
}
parameters {
  // We need to define two betas--the first is the restricted value, the next are the others. We'll join these in the next block
  real<lower = 0> beta_1;
  vector[P-1] beta_2; // the regression coefficients
  real<lower = 0> sigma; // the residual scale (note that it's restricted to be non-negative)
  real<lower = 0> nu; 
}
transformed parameters {
  vector[P] beta;
  beta = append_row(rep_vector(beta_1, 1), beta_2);
}
model {
  // Define the priors
  beta ~ normal(0, 5); // same prior for all betas; we could define a different one for each, or use a multivariate prior. The first beta will have a prior of the N+(0, 5)
  sigma ~ cauchy(0, 2.5);
  nu ~ cauchy(7, 5);

  // The likelihood
  y ~ student_t(nu, X*beta, sigma);
}
generated quantities {
  // For model comparison, we'll want to keep the likelihood contribution of each point
  vector[N] log_lik;
  vector[N] y_sim;
  for(i in 1:N){
    log_lik[i] = student_t_lpdf(y[i] | nu, X[i,]*beta, sigma);
    y_sim[i] = student_t_rng(nu, X[i,]*beta, sigma);
  }
}
"

# In R

# Specify the data list that we will pass to Stan. This gives Stan everything declared in the data{} block. 
data_list_2 <- list(X = X, N = N, y = y_sim, P = P)

# Call Stan. You'll need to give it either model_code (like the ones we defined above), a file (.stan file), 
# or a fitted Stan object (fit)
# You should also pass Stan a data list, number of cores to estimate on (jupyter only has access to one), 
# the number of Markov chains to run (4 by default)
# and number of iterations (2000 by default). 
# We use multiple chains to make sure that the posterior distribution that we converge on 
# is stable, and not affected by starting values. 

# The first time you run the models, they will take some time to compile before sampling. 
# On subsequent runs, it will only re-compile if you change the model code. 

incorrect_fit <- stan(model_code = incorrect_model, data = data_list_2, cores = 1, chains = 2, iter = 2000)
correct_fit <- stan(model_code = correct_model, data = data_list_2, cores = 1, chains = 2, iter = 2000)

# In R:

print(incorrect_fit, pars = c("beta", "sigma"))
# Notice that we specify which parameters we want; else we'd get values for `log_lik` and `yhat` also

# Some things to note: 

# - mean is the mean of the draws for each observation
# - se_mean is the Monte Carlo error (standard error of the Monte Carlo estimate from the true mean)
# - sd is the standard deviation of the parameter's draws
# - the quantiles are self-explanatory
# - n_eff is the effective number of independent draws. If there is serial correlation between sequential draws, 
#   the draws cannot be considered independent. In Stan, high serial correlation is typically a problem in 
#   poorly specified models
# - Rhat: this is the Gelman Rubin convergence diagnostic. Values close to 1 indicate that the multiple chains
#   that you estimated have converged to the same distribution and are "mixing" well.

# In R

# Declare a data frame that contains the known parameter names in one column `variable` and their known values
known_parameters <- data_frame(variable = c(paste0("beta[",1:P,"]"),"sigma", "nu"), real_value = c(beta, sigma, nu))


# Extract params as a (draws * number of chains * number of params) array
extract(correct_fit, permuted = F, pars = c("beta", "sigma", "nu")) %>% 
  # Stack the chains on top of one another and drop the chains label
  plyr::adply(2) %>% 
  dplyr::select(-chains) %>% 
  # Convert from wide form to long form (stack the columns on one another)
  melt() %>% 
  # Perform a left join with the known parameters
  left_join(known_parameters, by = "variable") %>%
  # Generate the plot
  ggplot(aes(x = value)) + 
  geom_density(fill = "orange", alpha = 0.5) + # Make it pretty
  facet_wrap(~ variable, scales = "free") +
  geom_vline(aes(xintercept = real_value), colour = "red") +
  ggtitle("Actual parameters and estimates\ncorrectly specified model\n")

extract(incorrect_fit, permuted = F, pars = c("beta", "sigma")) %>% 
  # Extract params as a (draws * number of chains * number of params) array
  plyr::adply(2) %>% 
  dplyr::select(-chains) %>% 
  # Stack the chains on top of one another and drop the chains label
  melt() %>% 
  left_join(known_parameters, by = "variable") %>% # Join the known parameter table
  # Convert from wide form to long form (stack the columns on one another)
  # Write out the plot
  ggplot(aes(x = value)) + 
  geom_density(fill = "orange", alpha = 0.5) + # Make it pretty
  facet_wrap(~ variable, scales = "free") + # small sub-plots of each variable
  geom_vline(aes(xintercept = real_value), colour = "red") + # red vertical lines for the known parameters
  ggtitle("Actual parameters and estimates\nincorrectly specified model\n") # A title
