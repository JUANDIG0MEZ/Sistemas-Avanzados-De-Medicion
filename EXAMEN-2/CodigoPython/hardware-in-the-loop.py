"""
Medidas Eléctricas e Instrumentación Electrónica
Facultad de Ingenierías - Universidad Tecnológica de Pereira
Programas de Ing. Eléctrica e Ing. Electrónica
2024-I
Yineth Martinez Armero, Germán A. Holguín L.

Modelo de un sensor de fuerza de orden 2.
"""

import math
import random
import matplotlib.pyplot as plt

# Parámetros de un modelo de segundo orden
omega_n = 2.0
zeta = 0.1
k_s = 4.0
masa = 1.0

# Parámetros de la simulación
dt = 0.01  # segundos
Tmax = 15  # segundos
muestras = int(Tmax / dt)
tiempo = []
fuerza = []

# Parámetros de la señal de entrada Fuerza(t)
amplitud = 0.8  # V
freq = 1  # Hz

# Generación de la señal de entrada Fuerza(t)
for ts in range(muestras + 1):
    tiempo.append(ts * dt)
    ruido = 0.05 * (2 * random.random() - 1)  # ruido ~ 0.1 * U[-1, 1)
    fuerza.append(amplitud * math.sin(2 * math.pi * freq * tiempo[ts]) + ruido)

# Estado inicial
x1 = 0
x2 = 0
x1_vector = []
x2_vector = []

# Aplicar la entrada al modelo de segundo orden
for fz in fuerza:
    x1 = x1 + dt * x2
    x2 = x2 + dt * (-(omega_n ** 2) * x1 - 2 * zeta * omega_n * x2 + k_s * fz / masa)
    x1_vector.append(x1)
    x2_vector.append(x2)

# Reescalar la fuerza
factor_escala = 1.0
fuerza_es = [factor_escala * fs for fs in fuerza]

# Graficar las señales
plt.figure()
plt.plot(tiempo, fuerza_es, label='Fuerza')
plt.plot(tiempo, x1_vector, label='x1(t)=y(t)')
plt.plot(tiempo, x2_vector, label='x2(t)', color='limegreen')

plt.legend(fontsize=14)
plt.xlabel('Tiempo [s]', fontsize=14)
plt.ylabel('Amplitud [V]', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.show()
