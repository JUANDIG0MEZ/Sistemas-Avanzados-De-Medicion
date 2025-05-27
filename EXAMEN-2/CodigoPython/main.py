import matplotlib.pyplot as plt
import numpy as np
from hardwareInTheLoop import Parametros_2, x1_modelo2, x2_modelo2, tiempo_2, fuerza_2
from hardwareInTheLoop import Parametros_1, x1_modelo1, tiempo_1, u_1



class KalmanFilter:
    def __init__(self, F, H, Q, R, P0):
        self.F = F  # Matriz de transición
        self.H = H  # Matriz de medición
        self.Q = Q  # Covarianza del proceso
        self.R = R  # Covarianza del sensor
        self.P = P0  # Covarianza del error
        self.x = np.array([[0.0], [0.0], [0.0]])  # Estado inicial

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(3) - K @ self.H) @ self.P


def fourier_2(y):
    dt = Parametros_2.dt

    Y = np.fft.fft(y)
    N = len(tiempo_2)

    omega = 2 * np.pi * np.fft.fftfreq(N, dt)
    omega_n = Parametros_2.omega_n
    zeta = Parametros_2.zeta
    k_s = Parametros_2.k_s
    masa = Parametros_2.masa


    F_omega = (masa)/k_s * ((-omega**2 + 2j*zeta*omega_n*omega + omega_n)* Y)

    F = np.fft.ifft(F_omega).real

    return F




class Modelo2:
    # Varianza de la tasa de cambio de la fuerza 
    q_F = 0.001
    # Varianza en las mediciones del sensor
    r_F = 0.039
    # Covarianza inicial
    P0 = np.eye(3) * 0.001  


    # Matrices necesarias para kalman
    A = np.array([
        [0, 1, 0],
        [-Parametros_2.omega_n**2, -2*Parametros_2.zeta*Parametros_2.omega_n, Parametros_2.k_s/ Parametros_2.masa],
        [0, 0, 0]
        ])
    F = np.eye(3) + A * Parametros_2.dt
    H = np.array([[1.0, 0.0, 0.0]])
    Q = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, q_F]]) * Parametros_2.dt
    R = np.array([[r_F]])  # varianza del sensor


    


class Modelo1:
    # Varianza de la tasa de cambio de u(t)
    q_u = 0.001
    # Varianza en las mediciones del sensor
    r_u = 0.039

    A = np.array([
        [-1/Parametros_1.k_s, Parametros_1.k_s/Parametros_1.tau],
        [0, 0]
    ])

    F = np.eye(2) + A *Parametros_1.dt
    H = np.array([[1.0, 0.0]])
    Q = np.array([
        [0, 0],
        [0, q_u * Parametros_1.dt]
    ])
    R = np.array([[r_u]])


def aplicar_fft(tiempo, señal):
    """
    Aplica la Transformada Rápida de Fourier (FFT) a una señal dada.

    Parámetros:
    - tiempo: vector de tiempo (numpy array)
    - señal: señal a analizar (numpy array)

    Retorna:
    - freqs: frecuencias correspondientes (Hz)
    - amplitud: espectro de amplitud (magnitud normalizada)
    """
    N = len(señal)                      # Número de puntos
    dt = tiempo[1] - tiempo[0]         # Paso de tiempo
    fs = 1 / dt                        # Frecuencia de muestreo

    # FFT
    fft_result = np.fft.fft(señal)
    fft_result = fft_result[:N // 2]  # Solo parte positiva

    # Frecuencias
    freqs = np.fft.fftfreq(N, dt)[:N // 2]

    # Amplitud normalizada
    amplitud = np.abs(fft_result) * 2 / N

    return freqs, amplitud


if __name__ == "__main__":
    

    ##################################################################
    ###################### MODELO PRIMER ORDEN #######################
    ##################################################################


    # kf_1 = KalmanFilter(
    #     Modelo1.F,
    #     Modelo1.H,
    #     Modelo1.Q,
    #     Modelo1.R,
    #     Modelo1.P0
    # )

    # x1_kalman_1 = []
    # u_kalman = []

    # for i in range(len(Parametros_1.tiempo)):
    #     z = Parametros_1.x1_vector[i] + np.random.normal(0, Modelo1.r_u ** (1/2))

    #     kf_1.predict()
    #     kf_1.update(np.array([[z]]))


    #     x1_kalman_1.append(kf_1.x[0, 0])
    #     u_kalman.append([1, 0])



    ##################################################################
    ####################### MODELO SEGUNDO ORDEN #####################
    ##################################################################

    ########################################
    ########### KALMAN #####################
    kf_2 = KalmanFilter(
        Modelo2.F,
        Modelo2.H,
        Modelo2.Q,
        Modelo2.R,
        Modelo2.P0)

    x1_kalman_2 = []
    x2_kalman_2 = []
    fuerza_kalman = []
    
    for i in range(len(tiempo_2)):

        z = x1_modelo2[i] #+ np.random.normal(0, Modelo2.r_F ** (1/2))

        kf_2.predict() # Paso de prediccion
        kf_2.update(np.array([[z]])) # z debe ser un vector columna y representa la meedicion de x1(t)


        x1_kalman_2.append(kf_2.x[0, 0]) # x1 estimado 
        x2_kalman_2.append(kf_2.x[1, 0]) # x2 estimado
        fuerza_kalman.append(kf_2.x[2,0])

    #######################################
    ############ FOURIER ##################

    fuerza_fourier = fourier_2(x1_modelo2)

    plt.plot(tiempo_2, x1_modelo2, label='x1(t)')
    plt.plot(tiempo_2, x1_kalman_2, label='x1(t) estimado')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('EstimacionX1.png')
    plt.show()

    plt.plot(tiempo_2, fuerza_2, label='Fuerza')
    plt.plot(tiempo_2, fuerza_kalman, label='Fuerza kalman')
    plt.plot(tiempo_2, fuerza_fourier, label='Fuerza Fourier')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('EstimacionF.png')
    plt.show()




    # Aplicar FFT a la fuerza real y estimada
    freqs_f, amp_f = aplicar_fft(tiempo_2, fuerza_2)
    freqs_f_est, amp_f_est = aplicar_fft(tiempo_2, fuerza_kalman)

    # Graficar
    plt.figure()
    plt.plot(freqs_f, amp_f, label="Fuerza real")
    plt.plot(freqs_f_est, amp_f_est, label="Fuerza estimada (Kalman)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.title("Espectro de la fuerza")
    plt.grid(True)
    plt.legend()
    plt.savefig("FFT_Fuerza.png")
    plt.show()

