import random
import numpy as np
import matplotlib.pyplot as plt

a, b, c, d = 2.5, 1.5, 1.0, 0.8
rango = (0, 10)
lr = 0.01
epochs = 2000

def modelo(x, a, b, c, d):
    return a * np.sin(b * x) + c * np.cos(d * x)

# Gradiente descendente
def gradiente_descendente(x, y, a, b, c, d, learning_rate=0.001, epochs=1000):
    for epoch in range(epochs):
        y_pred = modelo(x, a, b, c, d)
        error = y_pred - y

        # Calcular gradientes
        grad_a = np.sum(2 * error * np.sin(b * x)) / len(x)
        grad_b = np.sum(2 * error * a * x * np.cos(b * x)) / len(x)
        grad_c = np.sum(2 * error * np.cos(d * x)) / len(x)
        grad_d = np.sum(-2 * error * c * x * np.sin(d * x)) / len(x)

        # Actualizar parámetros
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        d -= learning_rate * grad_d
    return a, b, c, d


def agregarRuidoUniform(y, min, max):
    y_uniforme = y.copy()
    for i in range(len(y)):
        y_uniforme[i] = y_uniforme[i] + np.random.uniform(min, max)
    return y_uniforme


def agregarRuidoGaussiano(y, mu, sigma):
    y_ruido = y.copy()
    for i in range(len(y)):
        y_ruido[i] = y_ruido[i] + np.random.normal(mu, sigma)
    return y_ruido


# Generar datos x
x = np.linspace(rango[0], rango[1], 200)
# Generar datos y
y = modelo(x, a, b, c, d)
# Agrega ruido gaussiano
yRuidoGaussiano = agregarRuidoGaussiano(y, 0, 0.1)
# Agrega ruido uniforme
yRuidoUniforme = agregarRuidoUniform(y, -1.0, 1.0)

# Valores iniciales aleatorios para los parámetros
a_inicial =  random.uniform(-5, 5)
b_inicial =  random.uniform(-5, 5)
c_inicial =  random.uniform(-5, 5)
d_inicial =  random.uniform(-5, 5)

parametros_iniciales = (a_inicial, b_inicial, c_inicial, d_inicial)
parametros_gauss = gradiente_descendente(x, yRuidoGaussiano, a_inicial, b_inicial, c_inicial, d_inicial, lr, epochs)
parametros_uniformes = gradiente_descendente(x, yRuidoUniforme, a_inicial, b_inicial, c_inicial, d_inicial, lr, epochs)

print("\n -------Resultados-------\n")

print("Parámetros iniciales:", *parametros_iniciales)
print("Parámetros óptimos con ruido gaussiano:", *parametros_gauss)
print("Parámetros óptimos con ruido uniforme:", *parametros_uniformes)

# Generar curvas ajustadas
y_ajuste_gauss = modelo(x, *parametros_gauss)
y_ajuste_uniforme = modelo(x, *parametros_uniformes)

# Error cuadrático medio
error_gauss = np.mean((yRuidoGaussiano - y_ajuste_gauss) ** 2)
error_uniforme = np.mean((yRuidoUniforme - y_ajuste_uniforme) ** 2)
print(f'Error cuadrático medio (ruido gaussiano): {error_gauss:.4f}')
print(f'Error cuadrático medio (ruido uniforme): {error_uniforme:.4f}')

# Graficar
plt.figure(figsize=(12, 8))

#Datos originales
plt.plot(x, y, label="Datos originales (sin ruido)", color="blue", linewidth=2)

# Datos con ruido gaussiano
plt.scatter(x, yRuidoGaussiano, label="Datos con ruido gaussiano", color="orange", alpha=0.6)
plt.plot(x, y_ajuste_gauss, label="Ajuste (ruido gaussiano)", color="red", linestyle="--")

# Datos con ruido uniforme
plt.scatter(x, yRuidoUniforme, label="Datos con ruido uniforme", color="green", alpha=0.6)
plt.plot(x, y_ajuste_uniforme, label="Ajuste (ruido uniforme)", color="purple", linestyle="--")

# Configuración de la gráfica
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de modelo con ruido gaussiano y uniforme")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

# Mostrar la gráfica
plt.savefig("AJUSTE_NO_LINEAL.svg")
print()
# Interprete los resultados obtenidos y discuta la influencia los parámetros del ruido en la estimación de los parámetros del modelo.
print("----------Interpretación de resultados:----------\n")
print("Los resultados muestran que los parámetros obtenidos se mantienen relativamente constantes a pesar de la variación del ruido.\n"
      "Esto se observa porque los valores estimados a partir de los datos con ruido gaussiano y ruido uniforme son similares a pesar\n"
      "que el ruido uniforme tiene una magnitud mas grande. En consecuencia, se puede concluir que el modelo es robusto frente a  \ndiferentes tipos de ruido.\n\n"
      "Por otro lado, uno de los factores que más influye en la estimación de los parámetros es la elección de los valores iniciales.\n"
      "Si estos están demasiado alejados de los valores reales, el modelo puede no converger a la solución óptima,\n"
      "posiblemente debido a que se queda atrapado en mínimos locales. \n")
