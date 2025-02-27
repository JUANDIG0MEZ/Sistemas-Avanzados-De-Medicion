import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, kstest

def boxMuller(n):
    U1 = np.random.uniform(0, 1, size=n)
    U2 = np.random.uniform(0, 1, size=n)
    Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    return Z0, Z1

Z0, Z1 = boxMuller(500)
Z_100 = np.concatenate([Z0, Z1])

Z0, Z1 = boxMuller(5)
Z_10 = np.concatenate([Z0, Z1])

plt.figure(figsize=(9,6))
plt.hist(Z_100, bins=15, density=True, alpha=0.6, color='b', edgecolor='black')
plt.title("Histograma de 100 muestras (Box-Muller)")
plt.savefig('histograma100.png')

plt.figure(figsize=(9,6))
plt.hist(Z_10, bins=5, density=True, alpha=0.6, color='r', edgecolor='black')
plt.title("Histograma de 10 muestras (Box-Muller)")
plt.savefig('histograma10.png')


def kolmogorov_smirnov_test(Z, cdf):
    Z = np.sort(Z)
    n = len(Z)
    # Se calcula la función teórica en cada punto
    cdf_teorica = np.array([cdf(x) for x in Z])
    # Límite derecho: valor de la función empírica justo después del salto
    emp_der = np.arange(1, n + 1) / n
    # Límite izquierdo: valor de la función empírica justo antes del salto
    emp_izq = np.arange(0, n) / n
    d_der = np.abs(emp_der - cdf_teorica)
    d_izq = np.abs(cdf_teorica - emp_izq)
    d_n = np.max(np.concatenate([d_der, d_izq]))
    return d_n

d_n_100_norm = kolmogorov_smirnov_test(Z_100, norm.cdf)
d_n_100_expon = kolmogorov_smirnov_test(Z_100, expon.cdf)

d_n_10_norm = kolmogorov_smirnov_test(Z_10, norm.cdf)
d_n_10_expon = kolmogorov_smirnov_test(Z_10, expon.cdf)

print("RESULTADOS")
print("D_n_100_norm  : ", d_n_100_norm)
print("D_n_100_expon : ", d_n_100_expon)
print("D_n_10_norm   : ", d_n_10_norm)
print("D_n_10_expon  : ",d_n_10_expon)
