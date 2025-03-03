import math
import random

def boxMuller(mu= 0, sigma=1):
    u1 = random.random()
    u2 = random.random()

    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

    z0 = sigma * z0 + mu
    z1 = sigma * z1 + mu

    return z0, z1


# Metodo de Runge-Kutta de cuarto orden (RK4)
def rk4_step(f, t, x, h, u):
    k1 = f(t, x, u)
    print(k1)
    print(u)
    x_temp = [[x[i][0] + (h/2) * k1[i][0]] for i in range(3)]
    print(x_temp)
    k2 = f(t + h/2, x_temp, u)

    x_temp = [[x[i][0] + (h/2) * k2[i][0]] for i in range(3)]
    k3 = f(t + h/2, x_temp, u)
    x_temp = [[x[i][0] + h * k3[i][0]] for i in range(3)]
    k4 = f(t + h, x_temp, u)
    # Actualizar x
    x_new = [[x[i][0] + (h/6) * (k1[i][0] + 2*k2[i][0] + 2*k3[i][0] + k4[i][0])] for i in range(3)]
    return x_new

# Funcion para multiplicar dos matrices
def mult(A, B):
    # Obtener las dimensiones de las matrices
    filas_A = len(A)
    columnas_A = len(A[0])
    filas_B = len(B)
    columnas_B = len(B[0])

    # Verificar que las dimensiones sean compatibles
    if columnas_A != filas_B:
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")

    # Inicializar la matriz resultante con ceros
    resultado = [[0 for _ in range(columnas_B)] for _ in range(filas_A)]

    # Realizar la multiplicación
    for i in range(filas_A):
        for j in range(columnas_B):
            for k in range(columnas_A):
                resultado[i][j] += A[i][k] * B[k][j]

    return resultado


def sumM(A, B):
    # Obtener las dimensiones de las matrices
    filas_A = len(A)
    columnas_A = len(A[0])
    filas_B = len(B)
    columnas_B = len(B[0])

    # Verificar que las dimensiones sean iguales
    if filas_A != filas_B or columnas_A != columnas_B:
        raise ValueError("Las dimensiones de las matrices no son iguales.")

    # Inicializar la matriz resultante con ceros
    resultado = [[0 for _ in range(columnas_A)] for _ in range(filas_A)]

    # Realizar la suma
    for i in range(filas_A):
        for j in range(columnas_A):
            resultado[i][j] = A[i][j] + B[i][j]
    return resultado