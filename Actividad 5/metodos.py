import numpy as np

def generarY(x, a, b , c, rango, puntos):
    # ax + by + c = 0
    m = -a / b
    b = -c / b
    y = m* x + b
    return y


def agregarOutliers(y, min, max, outliers=0.1):
    for i in range(len(y)):
        if np.random.rand() < outliers:
            y[i] = y[i] + np.random.uniform(min, max)

def agregarRuidoGaussiano(y, mu, sigma):
    for i in range(len(y)):
        y[i] = y[i] + np.random.normal(mu, sigma)

def normalizacion(data):
    media = np.mean(data)
    desviacion = np.std(data)
    return (data - media) / desviacion