from metodos import mult, sumM
import matplotlib.pyplot as plt

# Parametros

R_1 = 5
R_2 = 20

C_1 = 100e-6
C_2 = 50e-6

L = 15e-3

C= [1, -1, -R_2]



# Definir las matrices A y B
A = [[ -1/(R_1*C_1), 0, -1/C_1],
     [0, 0, 1/C_2],
     [1/L, -1/L, -R_2/L]]

B = [[1/(R_1*C_1), 1/C_1],
     [0, 0],
     [0, 0]]

# Definir la función f(t, x) = A x + B u(t)
def f(t, x, u):
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
h = 0.01  # Tamaño del paso
t = 0.0  # Tiempo inicial
tiempo_total = int(0.8/h)  # Tiempo total de simulación

# Método de Runge-Kutta de cuarto orden (RK4)
def rk4_step(t, x, h, u):
    k1 = f(t, x, u)
    x_temp = [x[i] + (h / 2) * k1[i] for i in range(3)]
    k2 = f(t + h / 2, x_temp, u)
    x_temp = [x[i] + (h / 2) * k2[i] for i in range(3)]
    k3 = f(t + h / 2, x_temp, u)
    x_temp = [x[i] + h * k3[i] for i in range(3)]
    k4 = f(t + h, x_temp, u)
    # Actualizar x
    x_new = [x[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(3)]
    return x_new


datos =dict()

# Almacenamiento de datos
tiempos = []  # Lista para almacenar los tiempos
x1_vals = []  # Lista para almacenar x1
x2_vals = []  # Lista para almacenar x2
x3_vals = []  # Lista para almacenar x3

# Iterar el método RK4
for i in range(tiempo_total):
    
    print(f"Iteración {i + 1}: t = {t:.2f}, x = {x}")
    
    # Almacenar datos
    tiempos.append(t)
    x1_vals.append(x[0])
    x2_vals.append(x[1])
    x3_vals.append(x[2])
    
    # Calcular el siguiente paso
    x_nuevo = rk4_step(t, x, h, u)
    
    
    # Actualizar x y t
    x = x_nuevo
    t += h



# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(tiempos, x1_vals, label='x1')
plt.plot(tiempos, x2_vals, label='x2')
plt.plot(tiempos, x3_vals, label='x3')

# Personalizar la gráfica
plt.title("Evolución de las variables de estado")
plt.xlabel("Tiempo (s)")
plt.ylabel("Valor de x")
plt.legend()
plt.grid()
plt.show()

delta_x = f(t, x, u)
datos['x'] = x
datos['delta_x'] = delta_x
