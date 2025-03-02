from metodos import *

R_1 = 1
R_2 = 2

C_1 = 1e-10
C_2 = 20e-6

L = 5e-3

h = 1/10

u = [[2], [3]]

A = [[ -1/(R_1*C_1), 0, -1/C_1],
     [0, 0, 1/C_2],
     [1/L, -1/L, -R_2/L]]

B = [[1/(R_1*C_1), 1/C_1],
     [0, 0],
     [0, 0]]

C= [1, -1, -R_2]

x_0 = [[0], [0], [0]]

# Definir la función f(t, x) = A x + B u(t)
def f(t, x, u):
    # Multiplicar matriz A por vector x
    Ax = mult(A,x)
    # Multiplicar matriz B por vector u
    Bu = mult(B, u)
    # Sumar Ax y Bu
    return sumM(Ax, Bu)

t=0
for i in range(1000):
    print("Iteracion", i )
    x_0 = rk4_step(f, t, x_0, h, u)
    t += h



#metodos.boxMuller()
#metodos.rungeKutta4()

for i in t:
    print(i)