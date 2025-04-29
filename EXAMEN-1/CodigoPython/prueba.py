import numpy as np

error = 5.0  # por ejemplo
lambda_ = error**2  # porque sigma = sqrt(lambda)
x_con_ruido = np.random.poisson(lam=lambda_)

print(x_con_ruido)