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



class Modelo2:
    # Varianza de la tasa de cambio de la fuerza 
    q_F = 0.001
    # Varianza en las mediciones del sensor
    r_F = 0.0003
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
    r_u = 0.003

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


    plt.plot(tiempo_2, x1_modelo2, label='x1(t)')
    plt.plot(tiempo_2, x1_kalman_2, label='x1(t) estimado')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('EstimacionX1.png')
    plt.show()


    plt.plot(tiempo_2, x2_modelo2, label='x2(t)')
    plt.plot(tiempo_2, x2_kalman_2, label='x2(t) estimado')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('EstimacionX2.png')
    plt.show()


    plt.plot(tiempo_2, fuerza_2, label='Fuerza')
    plt.plot(tiempo_2, fuerza_kalman, label='Fuerza estimada')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('EstimacionF.png')
    plt.show()
