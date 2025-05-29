import matplotlib.pyplot as plt
import numpy as np
from hardwareInTheLoop import Parametros_2, x1_modelo2, tiempo_2, fuerza_2
from hardwareInTheLoop import Parametros_1, x1_modelo1, tiempo_1, u_1



class KalmanFilter3:
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


class KalmanFilter2:
    def __init__(self, F, H, Q, R, P0):
        self.F = F  # Matriz de transición
        self.H = H  # Matriz de medición
        self.Q = Q  # Covarianza del proceso
        self.R = R  # Covarianza del sensor
        self.P = P0  # Covarianza del error
        self.x = np.array([[0.0], [0.0]])  # Estado inicial

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x += K @ (z - self.H @ self.x)
        self.P = (np.eye(2) - K @ self.H) @ self.P



def fourier_2(y):
    dt = Parametros_2.dt

    Y = np.fft.fft(y)
    N = len(tiempo_2)

    omega = 2 * np.pi * np.fft.fftfreq(N, dt)
    omega_n = Parametros_2.omega_n
    zeta = Parametros_2.zeta
    k_s = Parametros_2.k_s
    masa = Parametros_2.masa


    F_omega = (masa)/k_s * ((-omega**2 + 2j*zeta*omega_n*omega + omega_n**2)* Y)

    F = np.fft.ifft(F_omega).real

    return F


def fourier_1(y):
    dt = Parametros_1.dt
    N = len(tiempo_1)


    Y = np.fft.fft(y)
    

    tau = Parametros_1.tau
    k_s = Parametros_1.k_s
    omega = 2 * np.pi * np.fft.fftfreq(N, dt)

    U_omega = (tau * 1j * omega + 1) * Y / k_s

    u = np.fft.ifft(U_omega).real
    return u



class Modelo2:
    # Varianza de la tasa de cambio de la fuerza 
    q_F = 2.0
    # Varianza en las mediciones del sensor
    #r_F = 0.039
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
    q_u = 0.1
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

    P0 = np.eye(2) * 0.001  # Covarianza inicial


if __name__ == "__main__":
    

    ##################################################################
    ###################### MODELO PRIMER ORDEN #######################
    ##################################################################


    kf_1 = KalmanFilter2(
        Modelo1.F,
        Modelo1.H,
        Modelo1.Q,
        Modelo1.R,
        Modelo1.P0
    )

    x1_kalman_1 = []
    u_kalman = []
    


    for i in range(len(tiempo_1)):
        z = x1_modelo1[i]

        kf_1.predict()
        kf_1.update(np.array([[z]]))


        x1_kalman_1.append(kf_1.x[0, 0])
        u_kalman.append(kf_1.x[1, 0])

    u_fourier = fourier_1(x1_modelo1)

 


    ##################################################################
    ####################### MODELO SEGUNDO ORDEN #####################
    ##################################################################

    #######################################
    ########## KALMAN #####################
    kf_2 = KalmanFilter3(
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




    plt.plot(tiempo_1, u_1, label='u(t) ideal')
    plt.plot(tiempo_1, u_fourier, label='u(t) fourier', alpha=0.5)
    plt.ylim(-2, 2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.title('Estimacion de u(t) modelo 1 con Fourier')
    plt.legend()
    plt.grid()
    plt.savefig('estimacion_u_fourier_modelo_1.png')
    plt.show()

    
    plt.plot(tiempo_1, u_1, label='u(t) ideal')
    plt.plot(tiempo_1, u_kalman, label='u(t) kalman')
    plt.ylim(-2, 2)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.title('Estimacion de u(t) modelo 1 con Kalman')
    plt.legend()
    plt.grid()
    plt.savefig('estimacion_u_kalman_modelo_1.png')
    plt.show()
    plt.figure()

    plt.plot(tiempo_2, fuerza_kalman, label='F(t) kalman')
    plt.plot(tiempo_2, fuerza_2, label='F(t) ideal')
    plt.ylim(-2, 2)
    plt.title('Estimacion de F(t) modelo 2 con Kalman')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('estimacion_f_kalman_modelo_2.png')
    plt.show()


    plt.plot(tiempo_2, fuerza_2, label='F(t) ideal')
    plt.plot(tiempo_2, fuerza_fourier, label='F(t) Fourier', alpha=0.5)
    plt.ylim(-2, 2)
    plt.xlabel('Tiempo (s)')
    plt.title('Estimacion de F(t) modelo 2 con Fourier')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('estimacion_f_fourier_modelo_2.png')
    plt.show()
    
    ########################################
    ############### ERRORES ################

    error_kalman_1 = np.array(u_1) - np.array(u_kalman)
    error_kalman_2 = np.array(fuerza_2) - np.array(fuerza_kalman)
    error_fourier_1 = np.array(u_1) - np.array(u_fourier)
    error_fourier_2 = np.array(fuerza_2) - np.array(fuerza_fourier)

    # Elevan al cuadrado los errores
    error_kalman_1 = np.square(error_kalman_1)
    error_kalman_2 = np.square(error_kalman_2)
    error_fourier_1 = np.square(error_fourier_1)
    error_fourier_2 = np.square(error_fourier_2)

    print("Error cuadratico medio modelo 1 (Kalman):", np.mean(error_kalman_1))
    print("Error cuadratico medio modelo 2 (Kalman):", np.mean(error_kalman_2))
    print("Error cuadratico medio modelo 1 (Fourier):", np.mean(error_fourier_1))
    print("Error cuadratico medio modelo 2 (Fourier):", np.mean(error_fourier_2))

