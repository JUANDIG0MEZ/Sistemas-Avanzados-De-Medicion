import numpy as np
import matplotlib.pyplot as plt


def generar_fuerza(tiempo, amplitud, freq, ruido = 0.05):
    return amplitud * np.sin(2 * np.pi * freq * tiempo) + np.random.normal(0.0, ruido,len(tiempo))

class SegundoOrden:
    def __init__(self, omega_n, zeta, Ks, masa, dt):
        self.omega_n = omega_n
        self.zeta = zeta
        self.Ks = Ks
        self.masa = masa
        self.dt = dt
        self.x1 = 0.0 # y(t)
        self.x2 = 0.0 # dy(t)/dt

    def Actualizar(self, fuerza):
        x1 = self.x1 + self.dt * self.x2
        x2 = self.x2 + self.dt * (-(self.omega_n ** 2) * self.x1 - 2 * self.zeta * self.omega_n * self.x2 + self.Ks * fuerza / self.masa)

        self.x1 = x1
        self.x2 = x2

        return x1, x2
    
def graficar_señales(tiempo, fuerza, x1_vector, x2_vector, save=False):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(tiempo, fuerza, label='Fuerza')
    plt.plot(tiempo, x1_vector, label='x1(t)=y(t)')
    plt.plot(tiempo, x2_vector, label='x2(t)', color='limegreen')
    plt.title('Modelo de Segundo Orden')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    if save:
        plt.savefig('./imagenes/SegundoOrden.png')
    plt.show()
    


omega_n = 2.0
zeta = 0.01
k_s = 4.0
masa = 1.0

# Parámetros de la simulación
dt = 0.01  # segundos
Tmax = 15  # segundos
muestras = int(Tmax / dt)
tiempo = np.linspace(0, Tmax, muestras + 1)

# Parámetros de la señal de entrada Fuerza(t)
amplitud = 0.8  # V
freq = 1.5  # Hz

fuerza = generar_fuerza(tiempo, amplitud, freq)

# Estado inicial
x1_vector = []
x2_vector = []

# Aplicar la entrada al modelo de segundo orden
modelo = SegundoOrden(omega_n, zeta, k_s, masa, dt)

for fz in fuerza:
    x1, x2 = modelo.Actualizar(fz)
    x1_vector.append(x1)
    x2_vector.append(x2)

# graficar las señales
graficar_señales(tiempo, fuerza, x1_vector, x2_vector, save=True)