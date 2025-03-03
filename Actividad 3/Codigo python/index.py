from metodos import *
import matplotlib.pyplot as plt

# Definir la función f(t, x) = A x + B u(t)
def f(t, x, u, A, B):
    # Convertir x en una lista de listas para la multiplicación
    x_matrix = [[x[0]], [x[1]], [x[2]]]
    # Multiplicar A por x
    Ax = mult(A, x_matrix)
    # Convertir u en una lista de listas para la multiplicación
    u_matrix = [[u[0]], [u[1]]]
    # Multiplicar B por u
    Bu = mult(B, u_matrix)
    # Sumar Ax y Bu
    resultado = sumM(Ax, Bu)
    # Convertir el resultado de nuevo a una lista simple
    return [resultado[i][0] for i in range(3)]


# Condiciones iniciales
x = [0.0, 0.0, 0.0]  # Vector de estado inicial
u = [0.2, 0.1]  # Vector de entradas (constante)
h = 0.001  # Tamaño del paso
t = 0.0  # Tiempo inicial
tiempo_total = int(0.5/h)  # Tiempo total de simulación

# Método de Runge-Kutta de cuarto orden (RK4)
def rk4_step(t, x, h, u, A, B):
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


datos =dict()

# Almacenamiento de datos
tiempos = []  # Lista para almacenar los tiempos
x1_vals = []  # Lista para almacenar x1
x2_vals = []  # Lista para almacenar x2
x3_vals = []  # Lista para almacenar x3

# Método de Monte Carlo
num_simulaciones = 100  # Número de simulaciones Monte Carlo
resultados = []  # Almacenar resultados de cada simulación


import matplotlib.pyplot as plt

def simulacion_rk4(t, x, h, u, A, B, tiempo_total, tolerancia=1e-6):
    """ Ejecuta una simulación con Runge-Kutta 4 y criterio de parada. """
    x_sim = x.copy()
    t_sim = t
    x_anterior = None  # Para verificar convergencia
    x_vals = []
    
    for i in range(tiempo_total):
        x_nuevo = rk4_step(t_sim, x_sim, h, u, A, B)

        # Criterio de parada: Si el cambio entre iteraciones es menor que la tolerancia, detener
        if x_anterior is not None:
            cambio = max(abs(x_nuevo[j] - x_anterior[j]) for j in range(len(x_nuevo)))
            if cambio < tolerancia:
                print(f"Convergencia alcanzada en iteración {i + 1} con cambio {cambio:.2e}")
                break
        
        x_vals.append(x_nuevo)
        x_anterior = x_sim  # Guardar estado anterior
        x_sim = x_nuevo
        t_sim += h

    return x_vals


def metodo_monte_carlo(num_simulaciones, t, x, h, u, tiempo_total):
    """ Ejecuta múltiples simulaciones de Monte Carlo y grafica los resultados. """
    resultados = []
    
    for sim in range(num_simulaciones):
        print(f"Simulación {sim + 1}/{num_simulaciones}")
        
        # Parámetros de los componentes (media y desviación estándar)
        R_1 = (1, 0.1)  # (media, desviación estándar)
        R_2 = (2, 0.2)
        C_1 = (1000e-6, 50e-6)
        C_2 = (500e-6, 25e-6)
        L = (15e-3, 1e-3)

        # Generar valores aleatorios para los componentes
        r_1, r_2, c_1, c_2, l = elementos(R_1, R_2, C_1, C_2, L)

        # Definir las matrices A y B
        A = [[ -1/(r_1*c_1), 0, -1/c_1],
             [0, 0, 1/c_2],
             [1/l, -1/l, -r_2/l]]

        B = [[1/(r_1*c_1), 1/c_1],
             [0, 0],
             [0, 0]]
        
        # Ejecutar simulación con RK4
        x_vals = simulacion_rk4(t, x, h, u, A, B, tiempo_total)

        # Guardar resultados de esta simulación
        resultados.append(x_vals)

     # Graficar resultados de Monte Carlo
    plt.figure(figsize=(10, 6))
    for i, x1_vals in enumerate(resultados):
        plt.plot(range(len(x1_vals)), x1_vals, alpha=0.5, label=f'Simulación {i + 1}' if i < 5 else None)

    plt.title("Evolución de X en múltiples simulaciones de Monte Carlo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Valor de x")
    plt.legend()
    plt.grid()
    plt.show()

    print(resultados)

    # # Graficar resultados de Monte Carlo con colores para cada componente de x
    # plt.figure(figsize=(10, 6))
    # colores = ['r', 'g', 'b']  # Colores para x1, x2 y x3

    # for i, x_vals in enumerate(resultados):
    #     x_vals = list(zip(*x_vals))  # Transponer para separar x1, x2 y x3
    #     for j in range(3):  # Para cada componente de x (x1, x2, x3)
    #         plt.plot(range(len(x_vals[j])), x_vals[j], alpha=0.5, color=colores[j], label=f'Simulación {i + 1} - x{j + 1}' if i < 5 else None)

    # plt.title("Evolución de X en múltiples simulaciones de Monte Carlo")
    # plt.xlabel("Tiempo (s)")
    # plt.ylabel("Valor de x")
    # plt.legend()
    # plt.grid()
    # plt.show()



metodo_monte_carlo(num_simulaciones=100, 
                   t=0, x=[0, 0, 0], h=0.001, u=[1, 0], tiempo_total=100)