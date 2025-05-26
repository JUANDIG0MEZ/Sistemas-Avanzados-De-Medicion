import numpy as np
import matplotlib.pyplot as plt


# Parámetros del proceso
dt = 1.0          # tiempo de muestreo [s]
Tmax = 5400       # duración total en segundos
tiempo = np.arange(0, Tmax+1, dt)


# Perfil de referencia (ejemplo térmico similar al código Octave)
def dT(t):
    perfiles = [1/3, 0, 1/3, 0, 1/6, 0, -5/18, -5/18, -5/18, -5/18]
    return perfiles[min(t // 600, len(perfiles)-1)]

T_real = [0]
for t in tiempo[:-1]:
    T_real.append(T_real[-1] + dT(int(t)))

T_real = np.array(T_real)

plt.figure()
plt.plot(tiempo, T_real, label="Proceso Real")
plt.title("Proceso de Referencia (p.ej. Temperatura)")
plt.xlabel("Tiempo [s]")
plt.ylabel("T(t)")
plt.grid()
plt.legend()
plt.show()

# def generar_fuerza(tiempo, amplitud, freq, ruido = 0.05):
#     return amplitud * np.sin(2 * np.pi * freq * tiempo) + np.random.normal(0.0, ruido,len(tiempo))

# # Generador de fuerza con ruido y posibles outliers
# def generar_fuerza(tiempo, amplitud, freq, ruido=0.05):
#     señal = amplitud * np.sin(2 * np.pi * freq * tiempo)
#     ruido_aditivo = np.random.normal(0.0, ruido, len(tiempo))
#     outliers = np.random.rand(len(tiempo)) < 0.001
#     señal[outliers] += np.random.normal(5, 2, np.sum(outliers))  # Outliers grandes
#     return señal + ruido_aditivo


# Sensor de primer orden (RTD)
class SensorPrimerOrden:
    def __init__(self, tau, dt):
        self.tau = tau
        self.dt = dt
        self.y = 0.0


    def actualizar(self, entrada):
        self.y += self.dt * (-(self.y - entrada) / self.tau)
        return self.y
    

# Sensor de segundo orden
class SensorSegundoOrden:
    def __init__(self, omega_n, zeta, dt):
        self.omega_n = omega_n
        self.zeta = zeta
        self.dt = dt
        self.x1 = 0.0
        self.x2 = 0.0

    def actualizar(self, entrada):
        dx1 = self.x2
        dx2 = -2 * self.zeta * self.omega_n * self.x2 - self.omega_n**2 * (self.x1 - entrada)
        self.x1 += self.dt * dx1
        self.x2 += self.dt * dx2
        return self.x1

    
# Inicialización de sensores
sensor1 = SensorPrimerOrden(tau=60.0, dt=dt)
sensor2 = SensorSegundoOrden(omega_n=0.15, zeta=0.2, dt=dt)

# Aplicar a todo el proceso
T_sensor1 = []
T_sensor2 = []

for T in T_real:
    T_sensor1.append(sensor1.actualizar(T))
    T_sensor2.append(sensor2.actualizar(T))

T_sensor1 = np.array(T_sensor1)
T_sensor2 = np.array(T_sensor2)

# Graficar
plt.figure()
plt.plot(tiempo, T_real, label="T_real")
plt.plot(tiempo, T_sensor1, label="Sensor 1er orden")
plt.plot(tiempo, T_sensor2, label="Sensor 2do orden")
plt.title("Lecturas de los Sensores")
plt.xlabel("Tiempo [s]")
plt.ylabel("Temperatura simulada [°C]")
plt.grid()
plt.legend()
plt.show()


def ruido_sensor(T, tipo="rtd"):
    if tipo == "rtd":
        return np.random.normal(0, 0.01) + np.random.normal(0, 0.005 * abs(T))
    elif tipo == "ntc":
        return np.random.normal(0, 5.5)
    elif tipo == "termopar":
        return np.random.normal(0, 1.5) + np.random.normal(0, 0.004 * abs(T))
    return 0

medida_sensor1 = T_sensor1 + np.array([ruido_sensor(T, "rtd") for T in T_sensor1])
medida_sensor2 = T_sensor2 + np.array([ruido_sensor(T, "termopar") for T in T_sensor2])

plt.figure()
plt.plot(tiempo, medida_sensor1, label="Sensor 1er orden + ruido")
plt.plot(tiempo, medida_sensor2, label="Sensor 2do orden + ruido")
plt.title("Lecturas de Sensores con Ruido")
plt.xlabel("Tiempo [s]")
plt.ylabel("Señal con ruido")
plt.grid()
plt.legend()
plt.show()


def acondicionar_voltaje(T, sensor="rtd"):
    if sensor == "rtd":
        Vcc = 10
        Ro = 100
        alpha = 0.0039
        R = Ro + alpha * Ro * T
        R_lim = 4000
        V = (Vcc * R) / (R + R_lim)
        return np.clip(-10 + 20 * V, -10, 10)
    elif sensor == "termopar":
        # Simplificación del voltaje de un termopar tipo J
        mV = 0.05 * T  # coeficiente ficticio
        return np.clip(-9 + 500 * mV * 1e-3, -10, 10)

Vdaq_sensor1 = acondicionar_voltaje(medida_sensor1, "rtd")
Vdaq_sensor2 = acondicionar_voltaje(medida_sensor2, "termopar")

plt.figure()
plt.plot(tiempo, Vdaq_sensor1, label="DAQ Sensor 1")
plt.plot(tiempo, Vdaq_sensor2, label="DAQ Sensor 2")
plt.title("Voltajes de Entrada para DAQ")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid()
plt.legend()
plt.show()



