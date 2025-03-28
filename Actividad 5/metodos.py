import numpy as np

def generarY(x, a, b , c, rango, puntos):
    # ax + by + c = 0
    m = -a / b
    b = -c / b
    y = m* x + b
    return y


def agregarOutliers(y, min, max, outliers=0.1):
    y_outliers = y.copy()
    for i in range(len(y)):
        if np.random.rand() < outliers:
            y_outliers[i] = y_outliers[i] + np.random.uniform(min, max)
    return y_outliers

def agregarRuidoGaussiano(y, mu, sigma):
    y_ruido = y.copy()
    for i in range(len(y)):
        y_ruido[i] = y_ruido[i] + np.random.normal(mu, sigma)
    return y_ruido

def normalizacion(data):
    media = np.mean(data)
    desviacion = np.std(data)
    return (data - media) / desviacion