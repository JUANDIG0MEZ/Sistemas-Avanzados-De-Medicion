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