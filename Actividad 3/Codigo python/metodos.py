import math
import random
import matplotlib.pyplot as plt

def boxMuller(mu= 0, sigma=1):
    u1 = random.random()
    u2 = random.random()

    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    z0 = sigma * z0 + mu
    z1 = sigma * z1 + mu

    return z0, z1

# Metodo de Runge-Kutta de cuarto orden (RK4)
def rk4_step(f, t, x, h, u, A, B):
    k1 = f(t, x, u, A, B)
    x_temp = [x[i] + (h / 2) * k1[i] for i in range(3)]
    k2 = f(t + h / 2, x_temp, u, A, B)
    x_temp = [x[i] + (h / 2) * k2[i] for i in range(3)]
    k3 = f(t + h / 2, x_temp, u, A, B)
    x_temp = [x[i] + h * k3[i] for i in range(3)]
    k4 = f(t + h, x_temp, u, A, B)
    # Actualizar x
    x_new = [x[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]
    return x_new


def generar_elementos(R_1, R_2, C_1, C_2, L):
    r_1, _ = boxMuller(R_1[0], R_1[1])
    r_2, _ = boxMuller(R_2[0], R_2[1])
    c_1, _ = boxMuller(C_1[0], C_1[1])
    c_2, _ = boxMuller(C_2[0], C_2[1])
    L, _ = boxMuller(L[0], L[1])
    return r_1, r_2, c_1, c_2, L


def simulacion_rk4(f, t, x, h, u, A, B, tiempo_total, tolerancia=1e-3):
    """ Ejecuta una simulación con Runge-Kutta 4 y criterio de parada. """

    x_sim = x
    t_sim = t
    x_vals = []

    for i in range(tiempo_total):
        x_nuevo = rk4_step(f, t_sim, x_sim, h, u, A, B)
        x_vals.append(x_nuevo)
        x_sim = x_nuevo
        t_sim += h

    return x_vals

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


def graficar_x(resultados):
    # Colores personalizados
    colores = ["#A0A0A0", "#E53935", "orange"]  # Negro, Rojo, Naranja

    plt.figure(figsize=(14, 8))

    # Cantidad de simulaciones y iteraciones
    num_simulaciones = len(resultados)
    num_iteraciones = len(resultados[0])

    # Inicializar listas para almacenar los promedios de cada variable
    promedio_x1 = [0] * num_iteraciones
    promedio_x2 = [0] * num_iteraciones
    promedio_x3 = [0] * num_iteraciones

    # Iterar sobre todas las simulaciones para graficarlas
    for i in range(num_simulaciones):
        for j in range(3):
            x_vals = range(num_iteraciones)
            y_vals = [fila[j] for fila in resultados[i]]  # Obtener valores de la variable j
            plt.plot(x_vals, y_vals, color=colores[j], alpha=0.08, linewidth=0.8)  # Líneas individuales

            # Acumular valores para el promedio
            if j == 0:
                promedio_x1 = [promedio_x1[k] + y_vals[k] for k in range(num_iteraciones)]
            elif j == 1:
                promedio_x2 = [promedio_x2[k] + y_vals[k] for k in range(num_iteraciones)]
            elif j == 2:
                promedio_x3 = [promedio_x3[k] + y_vals[k] for k in range(num_iteraciones)]

    # Calcular los promedios dividiendo por el número de simulaciones
    promedio_x1 = [v / num_simulaciones for v in promedio_x1]
    promedio_x2 = [v / num_simulaciones for v in promedio_x2]
    promedio_x3 = [v / num_simulaciones for v in promedio_x3]

    # Graficar los promedios con líneas más gruesas, punteadas y con marcadores
    plt.plot(range(num_iteraciones), promedio_x1, color="k", linewidth=1, alpha=1, marker="o", markersize=1, markeredgewidth=1, markeredgecolor="white", label="Promedio X1")
    plt.plot(range(num_iteraciones), promedio_x2, color="k", linewidth=1, alpha=1, marker="s", markersize=1, markeredgewidth=1, markeredgecolor="white", label="Promedio X2")
    plt.plot(range(num_iteraciones), promedio_x3, color="k", linewidth=1, alpha=1, marker="D", markersize=1, markeredgewidth=1, markeredgecolor="white", label="Promedio X3")

    plt.title("Evolución de X en múltiples simulaciones de Monte Carlo")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Valor de x")
    import matplotlib.lines as mlines
    # Definir los elementos de la leyenda manualmente
    legend_x1 = mlines.Line2D([], [], color="r", label="X1 Promedio")
    legend_x2 = mlines.Line2D([], [], color="silver", label="X2 Promedio")
    legend_x3 = mlines.Line2D([], [], color="orange", label="X3 Promedio")

    # Agregar la leyenda sin depender de los datos
    plt.legend(handles=[legend_x1, legend_x2, legend_x3], loc="best")  # Agregar leyenda para los promedios
    plt.grid()
    plt.savefig("grafica.png")
