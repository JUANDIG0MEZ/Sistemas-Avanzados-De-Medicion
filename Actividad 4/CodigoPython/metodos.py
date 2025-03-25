import numpy as np
import matplotlib.pyplot as plt
def X(rango, numeroPuntos):
    # los puntos se obtienen de manera aleatoria en el rango especificado
    # utilizando probabilidades uniformes
    x = np.random.uniform(rango[0], rango[1], numeroPuntos)
    return x


x = X((0, 5), 30)


def agregarRuidoNormal(y, media, desviacion):
    # se agrega ruido normal a los puntos
    ruido = np.random.normal(media, desviacion, len(y))
    return y + ruido


def agregarRuidoUniforme(y, minimo, maximo):
    # se agrega ruido uniforme a los puntos
    ruido = np.random.uniform(minimo, maximo, len(y))
    return y + ruido


def sistemaFuncionCuadratica(a, b, c, x):
    return a * x * x + b * x + c

def construirMatrizA(x):
    A = np.array([x ** 2, x, np.ones(len(x))]).T
    return A

def graficar(x, y, x2, y2, titulo):
    plt.figure(figsize=(14, 8))
    plt.scatter(x, y, marker="8", color="orange")
    plt.plot(x2, y2, color="blue")
    plt.title(titulo)
    plt.savefig(titulo + '.png')
    plt.show()

def graficar2(x, y, y1, y2, titulo, label1, label2, label3):
    plt.figure(figsize=(14, 8))
    plt.plot(x, y, color="blue", label=label1)
    plt.plot(x, y1, color="black", label=label2)
    plt.plot(x, y2, color="red", label = label3)
    plt.legend
    plt.title(titulo)
    plt.savefig(titulo + '.png')
    plt.show()

def display_fit_comparisons(xlinspace, x, y, y_ruido_normal, y_ruido_uniforme, y_real, y_ajustada_normal, y_ajustada_uniforme):
    """
    Muestra comparaciones de ajustes de una función cuadrática con y sin ruido.
    Parámetros:
    xlinspace (array-like): Valores de x para la línea continua de ajuste.
    x (array-like): Valores de x de los datos originales.
    y (array-like): Valores de y de los datos originales sin ruido.
    y_ruido_normal (array-like): Valores de y de los datos con ruido normal.
    y_ruido_uniforme (array-like): Valores de y de los datos con ruido uniforme.
    y_real (array-like): Valores de y de la función cuadrática sin ruido.
    y_ajustada_normal (array-like): Valores de y de la función ajustada con ruido normal.
    y_ajustada_uniforme (array-like): Valores de y de la función ajustada con ruido uniforme.
    Esta función genera una figura con cuatro subgráficos:
    1. Datos originales y ajuste sin ruido.
    2. Datos con ruido normal y ajuste correspondiente.
    3. Datos con ruido uniforme y ajuste correspondiente.
    4. Comparación de los ajustes sin ruido, con ruido normal y con ruido uniforme.
    La figura se guarda como 'Grafica Completa.png' y se muestra en pantalla.
    """
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'o', label='Datos originales')
    plt.plot(xlinspace, y_real, label='Ajuste sin ruido')
    plt.title('Función cuadrática sin ruido')
    plt.legend()
    plt.ylim(min(y) - 1, max(y) + 1)

    plt.subplot(2, 2, 2)
    plt.plot(x, y_ruido_normal, 'o', label='Datos con ruido normal')
    plt.plot(xlinspace, y_ajustada_normal, label='Ajuste con ruido normal')
    plt.title('Función cuadrática con ruido normal')
    plt.legend()
    plt.ylim(min(y) - 1, max(y) + 1)

    plt.subplot(2, 2, 3)
    plt.plot(x, y_ruido_uniforme, 'o', label='Datos con ruido uniforme')
    plt.plot(xlinspace, y_ajustada_uniforme, label='Ajuste con ruido uniforme')
    plt.title('Función cuadrática con ruido uniforme')
    plt.legend()
    plt.ylim(min(y) - 1, max(y) + 1)

    plt.subplot(2, 2, 4)
    plt.plot(xlinspace, y_real, label='Ideal')
    plt.plot(xlinspace, y_ajustada_normal, label='Normal')
    plt.plot(xlinspace, y_ajustada_uniforme, label='Uniforme')
    plt.title('Comparación de ajustes')
    plt.legend()
    plt.ylim(min(y) - 1, max(y) + 1)
    plt.savefig('Grafica Completa.png')
    plt.tight_layout()
    plt.show()