"""
JUAN DIEGO GOMEZ Y JUAN CAMILO VASCO

Actividad Sesión 05
ENUNCIADO 1
Seleccione la implementación de RANSAC de su predilección y cree un ejemplo de cómo se usa la función con sus propios datos. 

Sistemas Avanzados de medición
Universidad Tecnológica de Pereira
Maestria en Ingeniería Electrica
2025-1
"""

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import metodos

# ax + by + c = 0 
a, b, c = 1, -5, 0
rango = (-10, 10)
numPuntos = 40

x = np.linspace(rango[0], rango[1], numPuntos)
y = metodos.generarY(x, a, b, c, rango, numPuntos)
y_ideal = y.copy()

# Agrega ruido pequeño a los datos
metodos.agregarRuidoGaussiano(y, 0, 2)
# Agrega outliers a los datos
metodos.agregarOutliers(y, -20, 20, 0.1)


# Se le agrega una dimension a x para que pueda ser utilizado en RANSAC
x = x.reshape(-1, 1)


# Se crea una isntancia de RANSAC
ransac = RANSACRegressor(
    estimator= LinearRegression(),
    min_samples = 0.5,
    residual_threshold = 5.0,
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