import numpy as np

def generarLineaRecta(range, puntos, m, b):
    x = np.linspace(range[0], range[1], puntos)
    y = m * x + b
    return x, y

def agregarRuidoUniforme(y, min, max, outliers=0.1):
    for i in range(len(y)):
        if np.random.rand() < outliers:
            y[i] = y[i] + np.random.uniform(min, max)

def agregarRuidoGaussiano(y, mu, sigma):
    for i in range(len(y)):
        y[i] = y[i] + np.random.normal(mu, sigma)
