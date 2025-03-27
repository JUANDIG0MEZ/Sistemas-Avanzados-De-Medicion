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
from sklearn.linear_model import RANSACRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
b, c = -7, 10
rango = (0, 5)
xlinspace = np.linspace(rango[0], rango[1], 100)

X = np.random.uniform(rango[0], rango[1], 100).reshape(-1, 1)

y2 = b * X + c

ruido = np.random.uniform(-5, 5, len(y2)).reshape(-1, 1)

y1 = y2 + ruido

outliers = np.zeros_like(y1)
num_outliers = len(y1) // 10  # Por ejemplo, 10% de los datos serán outliers
outlier_indices = np.random.choice(len(y1), num_outliers, replace=False)
outliers[outlier_indices] = np.random.uniform(-100, 100, num_outliers).reshape(-1, 1)

y = y1 + outliers

# Normalización tipo max-min
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_y.fit_transform(y)

# Implementación de RANSAC
ransac = RANSACRegressor()
ransac.fit(X_scaled, Y_scaled.ravel())

# Obtener inliers y outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

# Estimación final
if hasattr(ransac.estimator_, 'coef_'):
    x_estimated = ransac.estimator_.coef_
else:
    x_estimated = None
    print("The base estimator does not have a 'coef_' attribute.")

# Crear subgráficas
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Proceso de RANSAC paso a paso")

# Gráfica 1: Datos originales
axs[0, 0].scatter(X, y, color='blue', label='Datos originales')
axs[0, 0].set_title("Datos originales")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("y")

# Gráfica 2: Datos normalizados
axs[0, 1].scatter(X_scaled, Y_scaled, color='green', label='Datos normalizados')
axs[0, 1].set_title("Datos normalizados")
axs[0, 1].set_xlabel("X (normalizado)")
axs[0, 1].set_ylabel("y")
axs[0, 1].set_xlim(0, 1)
axs[0, 1].set_ylim(0, 1)

# Gráfica 3: Inliers y outliers
axs[1, 0].scatter(X_scaled[inlier_mask], Y_scaled[inlier_mask], color='blue', label='Inliers')
axs[1, 0].scatter(X_scaled[outlier_mask], Y_scaled[outlier_mask], color='red', label='Outliers')
axs[1, 0].set_title("Inliers y Outliers")
axs[1, 0].set_xlabel("X (normalizado)")
axs[1, 0].set_ylabel("y")
axs[1, 0].legend()
axs[0, 1].set_xlim(0, 1)
axs[0, 1].set_ylim(0, 1)

# Gráfica 4: Línea ajustada por RANSAC
line_X = np.linspace(0, 1, 100).reshape(-1, 1)
line_y = ransac.predict(line_X)
axs[1, 1].scatter(X_scaled[inlier_mask], Y_scaled[inlier_mask], color='blue', label='Inliers')
axs[1, 1].scatter(X_scaled[outlier_mask], Y_scaled[outlier_mask], color='red', label='Outliers')
axs[1, 1].plot(line_X, line_y, color='black', label='Línea ajustada')
axs[1, 1].set_title("Línea ajustada por RANSAC")
axs[1, 1].set_xlabel("X (normalizado)")
axs[1, 1].set_ylabel("y")
axs[1, 1].legend()
axs[0, 1].set_xlim(0, 1)
axs[0, 1].set_ylim(0, 1)

# Ajustar diseño y mostrar
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Coeficiente estimado:", x_estimated)


print("Enunciado 1")