from metodos import *
# Tiempo de simulacion en segundos
tiempo_simulacion = 0.07

# Metodo de Monte Carlo
num_simulaciones = 200  # Número de simulaciones Monte Carlo

# Condiciones iniciales
x = [0.0, 0.0, 0.0]  # Vector de estado inicial
u = [1, 0]  # Vector de entradas (constante)
h = 0.001  # Tamaño del paso
t = 0.0  # Tiempo inicial

## Parámetros de los componentes
# (media, Desviación estándar)
R_1 = (1, 0.1)
R_2 = (2, 0.2)
C_1 = (1000e-6, 50e-6)
C_2 = (500e-6, 25e-6)
L = (15e-3, 1e-3)
elementos = (R_1, R_2, C_1, C_2, L)

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

def monte_carlo(f, num_simulaciones, t, x, h, u, iteraciones, elementos):
    """ Ejecuta múltiples simulaciones de Monte Carlo y grafica los resultados. """

    R_1, R_2, C_1, C_2, L = elementos
    result_X = []
    result_Y =[]

    for i in range(num_simulaciones):
        # Generar valores de tal manera que sigan una distribucion normal
        r_1, r_2, c_1, c_2, l = generar_elementos(R_1, R_2, C_1, C_2, L)

        A = [[-1 / (r_1 * c_1), 0, -1 / c_1],
             [0, 0, 1 / c_2],
             [1 / l, -1 / l, -r_2 / l]]

        B = [[1 / (r_1 * c_1), 1 / c_1],
             [0, 0],
             [0, 0]]

        # Ejecutar simulación con RK4
        x_vals = simulacion_rk4(f, t, x, h, u, A, B, iteraciones)
        # Calcular y
        y_vals = [x1 - x2 - r_2 * x3 for x1, x2, x3 in x_vals]
        # Guardar resultados de esta simulación
        result_X.append(x_vals)
        result_Y.append(y_vals)
    return result_X, result_Y

result_X, result_Y = monte_carlo(f,
                     num_simulaciones=num_simulaciones,
                     elementos=elementos,
                     t=t,
                     x=x,
                     h=h,
                     u=u,
                     iteraciones=int(tiempo_simulacion/h))

graficar_x(result_X)
graficar_y(result_Y)