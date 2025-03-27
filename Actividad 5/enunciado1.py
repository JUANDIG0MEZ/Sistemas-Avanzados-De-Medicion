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
from scikit-learn.linear_model import RANSACRegressor
from scikit-learn.datasets import make_regression
from scikit-learn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# Generar datos de ejemplo
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=1, noise=10)
outliers = np.random.uniform(low=-3, high=3, size=(20, 1))
X = np.vstack([X, outliers])
y = np.append(y, np.random.uniform(low=-200, high=200, size=20))

# Normalización tipo max-min
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Implementación de RANSAC
ransac = RANSACRegressor()
ransac.fit(X_scaled, y)

# Obtener inliers y outliers
inlier_mask = ransac.inlier_mask_
outlier_mask = ~inlier_mask

# Estimación final
x_estimated = ransac.estimator_.coef_

# Crear subgráficas
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Proceso de RANSAC paso a paso")

# Gráfica 1: Datos originales
axs[0, 0].scatter(X, y, color='blue', label='Datos originales')
axs[0, 0].set_title("Datos originales")
axs[0, 0].set_xlabel("X")
axs[0, 0].set_ylabel("y")
axs[0, 0].set_xlim(-3, 3)
axs[0, 0].set_ylim(-250, 250)

# Gráfica 2: Datos normalizados
axs[0, 1].scatter(X_scaled, y, color='green', label='Datos normalizados')
axs[0, 1].set_title("Datos normalizados")
axs[0, 1].set_xlabel("X (normalizado)")
axs[0, 1].set_ylabel("y")
axs[0, 1].set_xlim(0, 1)
axs[0, 1].set_ylim(-250, 250)

# Gráfica 3: Inliers y outliers
axs[1, 0].scatter(X_scaled[inlier_mask], y[inlier_mask], color='blue', label='Inliers')
axs[1, 0].scatter(X_scaled[outlier_mask], y[outlier_mask], color='red', label='Outliers')
axs[1, 0].set_title("Inliers y Outliers")
axs[1, 0].set_xlabel("X (normalizado)")
axs[1, 0].set_ylabel("y")
axs[1, 0].legend()
axs[1, 0].set_xlim(0, 1)
axs[1, 0].set_ylim(-250, 250)

# Gráfica 4: Línea ajustada por RANSAC
line_X = np.linspace(0, 1, 100).reshape(-1, 1)
line_y = ransac.predict(line_X)
axs[1, 1].scatter(X_scaled[inlier_mask], y[inlier_mask], color='blue', label='Inliers')
axs[1, 1].scatter(X_scaled[outlier_mask], y[outlier_mask], color='red', label='Outliers')
axs[1, 1].plot(line_X, line_y, color='black', label='Línea ajustada')
axs[1, 1].set_title("Línea ajustada por RANSAC")
axs[1, 1].set_xlabel("X (normalizado)")
axs[1, 1].set_ylabel("y")
axs[1, 1].legend()
axs[1, 1].set_xlim(0, 1)
axs[1, 1].set_ylim(-250, 250)

# Ajustar diseño y mostrar
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("Coeficiente estimado:", x_estimated)


print("Enunciado 1")