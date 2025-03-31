import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def generarY(x, a, b , c):
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

# ax + by + c = 0 
a, b, c = 1, -5, 0
# Range de valores de x
rango = (-10, 10)
# Numero de puntos a generar
numPuntos = 40

x = np.linspace(rango[0], rango[1], numPuntos)
y = generarY(x, a, b, c)
y_ideal = y.copy()

# Agrega ruido pequeño a los datos
y = agregarRuidoGaussiano(y, 0, 2)
# Agrega outliers a los datos
y =agregarOutliers(y, -20, 20, 0.1)

# Se le agrega una dimension a x para que pueda ser utilizado en RANSAC
x = x.reshape(-1, 1)

# Se crea una instancia de RANSAC
ransac = RANSACRegressor(
    estimator= LinearRegression(),
    min_samples = 0.6, # Porcentaje de muestras necesarias para estimar el modelo
    residual_threshold = 5.0, # Umbral de residuo
    random_state = 42
)
ransac.fit(x, y)

# Es un vector de dimensiones (100,) con valores booleanos
# True si es un inlier, False si es un outlier 
mascara = ransac.inlier_mask_
y_ransac = ransac.predict(x)

plt.figure(figsize=(7, 5))
# Graficar resultados
plt.plot(x, y_ideal, "b--", linewidth=1.5, label="Ideal")
plt.scatter(x, y, color="orangered", edgecolors="black", label="Datos", zorder=2)
plt.scatter(x[~mascara], y[~mascara], facecolor="gold", edgecolors="black", s=100, label="Outliers", zorder=3)
plt.plot(x, y_ransac, color="purple", linewidth=2, label="Ajuste RANSAC")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
# Leyenda con fondo blanco y borde negro
plt.legend(facecolor='white', edgecolor='black', fontsize= 10)
plt.title("Regresión robusta con RANSAC")
plt.savefig("RANSAC.svg")