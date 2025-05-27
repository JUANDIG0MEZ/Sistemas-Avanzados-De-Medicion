import numpy as np
import matplotlib.pyplot as plt


def generar_fuerza(tiempo, amplitud, freq, ruido = 0.05):
    return amplitud * np.sin(2 * np.pi * freq * tiempo) + np.random.normal(0.0, ruido,len(tiempo))

def generar_u(tiempo, amplitud, freq, ruido= 0.05):
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

        # Actualiza
        self.x1 = x1
        self.x2 = x2

        return x1, x2
    
class PrimerOrden:
    def __init__(self, tau, k_s, dt):
        self.tau = tau
        self.k_s = k_s
        self.dt = dt
        self.x1 = 0.0

    
    def actualizar(self, u):
        x1 = self.x1 + self.dt * ( self.k_s * u -self.x1)/ self.tau

        # Actualiza
        self.x1 = x1

        return self.x1
    


class Parametros_2:
    omega_n = 2.0
    zeta = 0.01
    k_s = 4.0
    masa = 1.0
    
    amplitud = 0.8
    freq = 0.1

    dt = 0.03
    Tmax =10
    muestras = int(Tmax / dt)


class Parametros_1:
    tau = 1.0
    k_s = 1.0

    amplitud= 0.8
    freq = 0.1

    dt = 0.02
    Tmax =10
    muestras = int(Tmax / dt)



#######################################################################
####################### MODELO PRIMER ORDEN ###########################
#######################################################################

tiempo_1 = np.linspace(0, Parametros_1.Tmax, Parametros_1.muestras + 1)
u_1 = generar_u(tiempo_1, Parametros_1.amplitud, Parametros_1.freq)

x1_modelo1 = []

# Aplicar la entrada al modelo de primer orden
modelo_1 = PrimerOrden(Parametros_1.tau, Parametros_1.k_s, Parametros_1.dt)

for u in u_1:
    x1 = modelo_1.actualizar(u)
    x1_modelo1.append(x1)


plt.figure()
plt.plot(tiempo_1, u_1, label='Fuerza')
plt.plot(tiempo_1, x1_modelo1, label='x1(t)=y(t)')
plt.title('Modelo de Primer Orden')
plt.xlabel('Tiempo (s)')
plt.ylabel('Respuesta')
plt.legend()
plt.grid()
plt.savefig('PrimerOrden.png')
plt.show()


#######################################################################
######################## MODELO SEGUNDO ORDEN #########################
#######################################################################

tiempo_2 = np.linspace(0, Parametros_2.Tmax, Parametros_2.muestras + 1)
fuerza_2 = generar_fuerza(tiempo_2, Parametros_2.amplitud, Parametros_2.freq)

# Estado inicial
x1_modelo2 = []
x2_modelo2 = []

# Aplicar la entrada al modelo de segundo orden
modelo_2 = SegundoOrden(Parametros_2.omega_n, Parametros_2.zeta, Parametros_2.k_s, Parametros_2.masa, Parametros_2.dt)

for fz in fuerza_2:
    x1, x2 = modelo_2.Actualizar(fz)
    x1_modelo2.append(x1)
    x2_modelo2.append(x2)

plt.figure()
plt.plot(tiempo_2, fuerza_2, label='Fuerza')
plt.plot(tiempo_2, x1_modelo2, label='x1(t)=y(t)')
plt.plot(tiempo_2, x2_modelo2, label='x2(t)', color='limegreen')
plt.title('Modelo de Segundo Orden')
plt.xlabel('Tiempo (s)')
plt.ylabel('Respuesta')
plt.legend()
plt.grid()
plt.savefig('SegundoOrden.png')
plt.show()