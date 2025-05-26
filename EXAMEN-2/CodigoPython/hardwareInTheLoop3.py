import numpy as np
import matplotlib.pyplot as plt

# ========= [1] Parámetros del tanque (el proceso físico) =========
# Este sistema representa el nivel de líquido en un tanque alimentado por una válvula.
# La dinámica es de segundo orden por la inercia del fluido + retardo de control.

omega_n = 2.0        # Frecuencia natural del sistema hidráulico (rad/s)
zeta = 0.01          # Amortiguamiento del sistema (casi sin fricción)
k_s = 4.0            # Ganancia del proceso (cuánto sube el nivel por unidad de flujo)
masa = 0.5           # Representa la "masa equivalente" o inercia hidráulica del sistema

# ========= [2] Simulación en tiempo discreto =========
dt = 0.01            # Paso de simulación (100 Hz)
Tmax = 15            # Tiempo total de simulación (s)
muestras = int(Tmax / dt)
tiempo = np.linspace(0, Tmax, muestras + 1)

# ========= [3] Entrada al sistema: Flujo por válvula (Fuerza) =========
# La válvula entrega un flujo sinusoidal con algo de ruido (simula un actuador real)
def generar_flujo_válvula(tiempo, amplitud, freq, ruido=0.05):
    return amplitud * np.sin(2 * np.pi * freq * tiempo) + np.random.normal(0.0, ruido, len(tiempo))

amplitud = 0.8       # Amplitud del flujo (L/s)
freq = 1.5           # Frecuencia de oscilación del flujo
flujo = generar_flujo_válvula(tiempo, amplitud, freq)  # Entrada al tanque

# ========= [4] Modelo del tanque: Sistema de Segundo Orden =========
# Simula cómo varía el nivel del líquido en función del flujo que entra.

class TanqueSegundoOrden:
    def __init__(self, omega_n, zeta, Ks, masa, dt):
        self.omega_n = omega_n
        self.zeta = zeta
        self.Ks = Ks
        self.masa = masa
        self.dt = dt
        self.nivel = 0.0       # x1(t): Nivel del líquido en el tanque
        self.derivada = 0.0    # x2(t): Velocidad de cambio del nivel

    def actualizar(self, flujo):
        # Ecuaciones discretas del sistema masa-resorte (fluido en tanque)
        nuevo_nivel = self.nivel + self.dt * self.derivada
        nueva_derivada = self.derivada + self.dt * (
            -(self.omega_n ** 2) * self.nivel
            - 2 * self.zeta * self.omega_n * self.derivada
            + self.Ks * flujo / self.masa
        )

        self.nivel = nuevo_nivel
        self.derivada = nueva_derivada
        return nuevo_nivel, nueva_derivada

# ========= [5] Sensores =========

# Sensor de primer orden (ej. sensor de nivel capacitivo con retardo)
class SensorPrimerOrden:
    def __init__(self, tau, dt):
        self.tau = tau
        self.dt = dt
        self.valor = 0.0

    def medir(self, proceso):
        self.valor += self.dt / self.tau * (proceso - self.valor)
        return self.valor

# Sensor de segundo orden (ej. sensor ultrasónico con filtro dinámico interno)
class SensorSegundoOrden:
    def __init__(self, omega_n, zeta, dt):
        self.omega_n = omega_n
        self.zeta = zeta
        self.dt = dt
        self.salida = 0.0
        self.derivada = 0.0

    def medir(self, proceso):
        nueva_salida = self.salida + self.dt * self.derivada
        nueva_derivada = self.derivada + self.dt * (
            -2 * self.zeta * self.omega_n * self.derivada
            - self.omega_n ** 2 * self.salida
            + self.omega_n ** 2 * proceso
        )
        self.salida = nueva_salida
        self.derivada = nueva_derivada
        return self.salida

# ========= [6] Simulación completa =========
tanque = TanqueSegundoOrden(omega_n, zeta, k_s, masa, dt)
sensor1 = SensorPrimerOrden(tau=0.4, dt=dt)
sensor2 = SensorSegundoOrden(omega_n=5.0, zeta=0.8, dt=dt)

nivel_referencia = []
sensor1_lectura = []
sensor2_lectura = []

for flujo_v in flujo:
    nivel_real, _ = tanque.actualizar(flujo_v)
    nivel_referencia.append(nivel_real)
    sensor1_lectura.append(sensor1.medir(nivel_real))
    sensor2_lectura.append(sensor2.medir(nivel_real))

# ========= [7] Visualización =========
plt.figure(figsize=(10,6))
plt.plot(tiempo, flujo, label='Flujo por válvula (entrada)', linestyle='dotted', alpha=0.7)
plt.plot(tiempo, nivel_referencia, label='Nivel real del tanque (proceso)', linewidth=2)
plt.plot(tiempo, sensor1_lectura, label='Sensor 1 - Primer orden (ej. capacitivo)')
plt.plot(tiempo, sensor2_lectura, label='Sensor 2 - Segundo orden (ej. ultrasónico)', linestyle='--')
plt.title("Simulación de nivel en tanque y respuesta de sensores")
plt.xlabel("Tiempo (s)")
plt.ylabel("Nivel de líquido (unidades arbitrarias)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========= [8] Guardar resultados =========
# para leer en "DAQ"


# Función para escalar a voltaje [-10, 10]
def escalar_a_voltaje(senal):
    minimo = np.min(senal)
    maximo = np.max(senal)
    return -10 + 20 * (senal - minimo) / (maximo - minimo)

# Escalar las señales
Vdaq_proceso = escalar_a_voltaje(nivel_referencia)
Vdaq_sensor1 = escalar_a_voltaje(sensor1_lectura)
Vdaq_sensor2 = escalar_a_voltaje(sensor2_lectura)

# Frecuencia de muestreo (Hz)
fs = 100  # Ya que dt = 0.01s → 100 muestras por segundo

#Grafica los voltajes
plt.figure(figsize=(10,6))
plt.plot(tiempo, Vdaq_proceso, label='Nivel tanque (proceso)', linewidth=2)
plt.plot(tiempo, Vdaq_sensor1, label='Sensor 1 - Primer orden (capacitivos)')
plt.plot(tiempo, Vdaq_sensor2, label='Sensor 2 - Segundo orden (ultrasónico)', linestyle='--')
plt.title("Señales de DAQ escaladas a voltaje")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


import scipy.io as sio  # Para guardar como archivo .mat compatible con Octave/MATLAB


# Guardar en archivo .mat (compatible con Octave)
sio.savemat('senales_daq.mat', {
    'Vdaq_proceso': Vdaq_proceso,
    'Vdaq_sensor1': Vdaq_sensor1,
    'Vdaq_sensor2': Vdaq_sensor2,
    'fs': fs,
    'descripcion': 'Nivel tanque (proceso), Sensor capacitivo, Sensor ultrasónico'
})
