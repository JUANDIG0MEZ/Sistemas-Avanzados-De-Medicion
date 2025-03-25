import numpy as np
import metodos
import matplotlib.pyplot as plt
import metodos


def solucionPunto1():

    # Punto 1
    A = [[1, 2, 3, 4],
         [2, 3, 5, 7],
         [4, 1, 2, 6],
         [3, 4, 1, 5],
         [5, 6, 4, 3],
         [7, 8, 6, 2]]

    # Descomposición en valores singulares (SVD)
    U, S, Vt = np.linalg.svd(A)

    # Resolver Ax = 0 usando los valores singulares
    # La solución no trivial está en el último vector de V (correspondiente al menor valor singular)
    x_svd = Vt[-1]

    print("Dado una matriz V resultado de una descomposición SDV de A se obtiene que:\n")
    print(Vt)


    print("\nx1     x2     x3     x4")
    print(f"{x_svd[0]:<6.2f} {x_svd[1]:<6.2f} {x_svd[2]:<6.2f} {x_svd[3]:<6.2f}")


def solucionPunto2():
    a, b, c = 1.5, -7, 10
    rango = (0, 5)
    xlinspace = np.linspace(rango[0], rango[1], 100)

    x = metodos.X(rango, 100)
    y = metodos.sistemaFuncionCuadratica(a, b, c, x)

    y_ruido_normal = metodos.agregarRuidoNormal(y, 0, 0.5)
    y_ruido_uniforme = metodos.agregarRuidoUniforme(y, -0.5, 0.5)

    # Se contruye la matriz A
    A = metodos.construirMatrizA(x)

    # Descomposicion de valores singulares
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # A pseudo inversa
    S_inv = np.diag(1/S)
    Ap = Vt.T @ S_inv @ U.T

    # Solucion del sistema
    abc_real = Ap @ y
    abc_ruido_normal = Ap @ y_ruido_normal
    abc_ruido_uniforme = Ap @ y_ruido_uniforme


    y_real = metodos.sistemaFuncionCuadratica(abc_real[0], abc_real[1], abc_real[2], xlinspace)
    y_ajustada_normal = metodos.sistemaFuncionCuadratica(abc_ruido_normal[0], abc_ruido_normal[1], abc_ruido_normal[2], xlinspace)
    y_ajustada_uniforme = metodos.sistemaFuncionCuadratica(abc_ruido_uniforme[0], abc_ruido_uniforme[1], abc_ruido_uniforme[2], xlinspace)

    # metodos.graficar(x, y, xlinspace, y_real,'Function cuadrática sin ruido')
    # metodos.graficar(x, y_ruido_normal, xlinspace,  y_ajustada_normal,'Función cuadrática con ruido normal')
    # metodos.graficar(x, y_ruido_uniforme, xlinspace, y_ajustada_uniforme,'Función cuadrática con ruido uniforme')

    # metodos.graficar2(xlinspace, y_real, y_ajustada_normal, y_ajustada_uniforme, 'Comparación de ajustes', 'Ideal', 'Normal', 'Uniforme')


    metodos.display_fit_comparisons(xlinspace, x, y, y_ruido_normal, y_ruido_uniforme, y_real, y_ajustada_normal, y_ajustada_uniforme)

    mse_real = np.mean((y - y_real) ** 2)
    mse_normal = np.mean((y_ruido_normal - y_ajustada_normal) ** 2)
    mse_uniforme = np.mean((y_ruido_uniforme - y_ajustada_uniforme) ** 2)

    print(f"{'Parámetros':<20} {'a':<5} {'b':<5} {'c':<5}")
    print(f"{'sin ruido:':<20} {abc_real[0]:<5.1f} {abc_real[1]:<5.1f} {abc_real[2]:<5.1f}")
    print(f"{'con ruido normal:':<20} {abc_ruido_normal[0]:<5.1f} {abc_ruido_normal[1]:<5.1f} {abc_ruido_normal[2]:<5.1f}")
    print(f"{'con ruido uniforme:':<20} {abc_ruido_uniforme[0]:<5.1f} {abc_ruido_uniforme[1]:<5.1f} {abc_ruido_uniforme[2]:<5.1f}")

    print("\n")
    print("MSE:")
    print(f"{'Ideal:':<10}{mse_real:<10.5f}")
    print(f"{'Normal:':<10}{mse_normal:<10.5f}")
    print(f"{'Uniforme:':<10}{mse_uniforme:<10.5f}")

    print("\nAnálisis:\n")

    print("Los resultados obtenidos en el MSE, son relativamente cercanos entre si,")

    print("se puede apreciar que, el error cuadrático medio de los datos con ruido,\n"
    "uniforme es mayor que el error cuadrático medio de los datos con ruido normal.")

    print("\nEsto se debe a que el ruido uniforme tiene una mayor dispersión que el ruido normal,"
    "lo que hace que los datos se alejen más de la función ideal.")

    print("\nPor otro lado, en algunas iteración el MSE de los datos con ruido normal"
    "es menor que el MSE de los datos sin ruido, \nesto se debe a que el ruido normal puede tener "
    "valores negativos, lo que puede hacer que los datos se acerquen a la función ideal en algunos casos.")

    print("Tambien, se puede observar que los párametros obtenidos con los datos con ruido uniforme son los "
    "que mas se acercan a los parametros reales,\nen comparación a la distribución uniforme.")

    print("\nEs clave resaltar la importancia del número de muestras en este caso 100 para un rango de 0 a 5, es más que suficiente,"
    "ya que con un número\nde muestras mayor, se obtienen resultados más precisos, dado que se tienen más datos para"
    "ajustar la función cuadrática.")

print("Solución punto 1:")
solucionPunto1()
print("\n------------------------------------\n")
print("Solucion punto 2:\n")
solucionPunto2() 