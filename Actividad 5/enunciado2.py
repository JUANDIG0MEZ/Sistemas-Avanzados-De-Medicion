"""
JUAN DIEGO GOMEZ Y JUAN CAMILO VASCO

Actividad Sesión 05
ENUNCIADO 2
Combinación de Funciones Trigonométricas  
Considere el siguiente modelo:   

y = a * sin(b*x) + c * cos(d*x)

donde x es la variable independiente, y(x) la variable dependiente; y a, b, c y d son los parámetros del modelo.

Sistemas Avanzados de medición
Universidad Tecnológica de Pereira
Maestria en Ingeniería Electrica
2025-1
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

a, b, c, d = 1.5, 0.5, 10, 0.3
rango = (0, 1)

xlinspace = np.linspace(rango[0], rango[1], 100)

x = np.random.uniform(rango[0], rango[1], 100)


y = a * np.sin(b*x) + c * np.cos(d*x)

ruidonormal = np.random.normal(0, 5, len(y))
ruidouniforme = np.random.uniform(-5, 5, len(y))

y_ruido_normal = y + ruidonormal
y_ruido_uniforme = y + ruidouniforme

y_ruidosa = y + ruidonormal + ruidouniforme

A = np.array([np.sin(b*x), np.cos(d*x), np.ones(len(x))]).T


# Definición de la función modelo
def modelo(params, x):
    a, b, c, d = params
    return a * np.sin(b * x) + c * np.cos(d * x)

# Función de error
def error(params, x, y_obs):
    return modelo(params, x) - y_obs

# Implementación del algoritmo de Levenberg-Marquardt
def LM(x, y_obs, p0, mu0, tol=1e-6, max_iter=100):
    pk = np.array(p0)
    mu = mu0
    for _ in range(max_iter):
        # Resolver por δpk usando el método de mínimos cuadrados
        res = least_squares(error, pk, args=(x, y_obs), method='lm')
        delta_pk = res.x - pk

        # Calcular pk+1
        pk_next = pk + delta_pk

        # Test de calidad
        rho = np.linalg.norm(error(pk_next, x, y_obs)) < np.linalg.norm(error(pk, x, y_obs))
        if rho:  # Test exitoso
            pk = pk_next
            mu /= 2
        else:  # Test fallido
            mu *= 2

        # Condición de parada
        if np.linalg.norm(delta_pk) < tol:
            break

    return pk

# Parámetros iniciales y ejecución del algoritmo
p0 = [1, 1, 1, 1]  # Valores iniciales para a, b, c, d
mu0 = 1e-2
params_opt = LM(x, y_ruidosa, p0, mu0)

print("Parámetros óptimos:", params_opt)

# Usando curve_fit para comparar
def modelo_curve_fit(x, a, b, c, d):
    return a * np.sin(b * x) + c * np.cos(d * x)

# Ajuste con curve_fit
params_curve_fit, _ = curve_fit(modelo_curve_fit, x, y_ruidosa, p0=p0)

print("Parámetros óptimos con curve_fit:", params_curve_fit)
