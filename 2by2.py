import numpy as np
import pystan

x1 = np.random.normal(0, 2, 100)
x2 = np.random.normal(-1, 1, 100)

x_array = [x1, x2]
x = np.stack(x_array, axis=1)

y1 = x1 + 2*x2 + 3
y2 = 2*x1 + 3 * x2 + 1


y_array = [y1, y2]
y = np.stack(y_array, axis=1)


regress_cod = """
data {
	int<lower=0> N;
	int<lower=1> J;
	int<lower=1> K;
	vector[J] x[N];
	vector[K] y[N];
}

parameters {
	matrix[K, J] beta;
	cholesky_factor_corr[K] L_Omega;
	vector<lower=0>[K] L_sigma;
}

transformed parameters {
	vector[K] mu[N];
	matrix[K, K] L_Sigma;
	for (n in 1:N)
	mu[n] = beta * x[n];
	L_Sigma = diag_pre_multiply(L_sigma, L_Omega);	
}

model {
	to_vector(beta) ~ normal(0, 5);
	L_Omega ~ lkj_corr_cholesky(4);
	L_sigma ~ cauchy(0, 2.5);
	y ~ multi_normal_cholesky(mu, L_Sigma);
}
"""

regress_dat = {"x":x, "y":y, "N":100, "J": 2, "K": 2}

fit = pystan.stan(model_code=regress_cod, data=regress_dat, iter=2000, chains=4)

print fit

