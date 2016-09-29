import numpy as np
import pystan

x1 = np.random.normal(0,5,300)
x2 = np.random.normal(1,3, 300)
x3 = np.random.normal(2, 4, 300)
x4 = np.random.normal(1, 7, 300)
x5 = np.random.normal(-1, 6, 300)

x_array = [x1, x2, x3, x4, x5]
x = np.stack(x_array, axis=1)
print x.shape

y1 = x1 + 2*x2 + 3*x3 + 4*x4 + 5*x5 + 8 + np.random.randn(300) * 4
y2 = 5*x1 + 4 * x2 + 3 * x3 + 2 * x2 + x5 + 6 + np.random.randn(300) * 2
y3 = 3 * x1 + 5 * x2 + 4 * x3 + x4 +  2* x5 + 4 + np.random.randn(300) * 5
y4 = 4 * x1 + 3 * x2 + 2 * x3 + 5 * x4 + x5 + 10 + np.random.randn(300) * 3
y5 = 2 * x1 + x2 + 5 * x3 + 4 * x4 + 3 * x5 + 3 + np.random.randn(300) * 2

y_array = [y1, y2, y3, y4, y5]
y = np.stack(y_array, axis=1)


regress_cod = """
data {
	int<lower=1> K;
	int<lower=1> J;
	int<lower=0> N;
	vector[J] x[N];
	vector[K] y[N];
}

parameters {
	matrix[K, J] beta;
	cholesky_factor_corr[K] L_Omega;
	vector<lower=0>[K] L_sigma;
}

# transformed parameters {

# }

model {
	vector[K] mu[N];
	matrix[K, K] L_Sigma;
	for (n in 1:N)
	mu[n] = beta * x[n];
	L_Sigma = diag_pre_multiply(L_sigma, L_Omega);	

	to_vector(beta) ~ normal(0, 5);
	L_Omega ~ lkj_corr_cholesky(4);
	L_sigma ~ cauchy(0, 2.5);
	y ~ multi_normal_cholesky(mu, L_Sigma);
}
"""


regress_dat = {"K": 5, "J": 5, "N":300, "x": x, "y": y}

fit = pystan.stan(model_code=regress_cod, data=regress_dat, iter=1000, chains=4)

print fit