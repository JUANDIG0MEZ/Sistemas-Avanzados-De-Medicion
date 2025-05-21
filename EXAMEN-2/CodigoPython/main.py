import matplotlib.pyplot as plt
import numpy as np
from hardwareInTheLoop import x1_vector, x2_vector, tiempo, fuerza
from hardwareInTheLoop import omega_n, zeta, k_s, masa, dt
class KalmanFilter:
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




if __name__ == "__main__":
    F_continua = np.array([[0, 1], [-omega_n**2, -2*zeta*omega_n]])  # Matriz de transición continua
    F = np.eye(2) + F_continua * dt  # Matriz de transición discreta

    B_continua = np.array([[0], [k_s/masa]])  # Matriz de entrada continua
    B = B_continua * dt  # Matriz de entrada discreta

    H = np.array([[1.0, 0.0]])  # Matriz de medición
    Q = np.eye(2) * 0.05  # Covarianza del proceso
    R = np.array([[0.05]])  # Covarianza del sensor
    P0 = np.eye(2) * 1.0  # Covarianza inicial
    
    
    kf = KalmanFilter(F, H, Q, R, P0)

    x1_kalman = []
    x2_kalman = []
    
    for i in range(len(tiempo)):

        z = x1_vector[i]

        kf.predict() # Paso de prediccion
        kf.update(np.array([[z]])) # z debe ser un vector columna y representa la meedicion de x1(t)


        x1_kalman.append(kf.x[0, 0]) # x1 estimado 
        x2_kalman.append(kf.x[1, 0]) # x2 estimado

    
    def recuperar_fuerza(x1, x2, dt, omega_n, zeta, k_s, masa):
        # Convertir las listas a arrays de numpy
        x1 = np.array(x1)
        x2 = np.array(x2)
        dx2 = np.gradient(x2, dt)

        F_estimada = (masa / k_s) * (dx2 + omega_n**2 * x1 + 2.0 * zeta * omega_n * x2)
        return F_estimada
    
    
    fuerza_kalman = recuperar_fuerza(x1_kalman, x2_kalman, dt, omega_n, zeta, k_s, masa)


    plt.plot(tiempo, x1_vector, label='x1(t)')
    plt.plot(tiempo, x1_kalman, label='x1(t) estimado', color='limegreen')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('./imagenes/EstimacionX1.png')
    plt.show()


    plt.plot(tiempo, x2_vector, label='x2(t)')
    plt.plot(tiempo, x2_kalman, label='x2(t) estimado', color='limegreen')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('./imagenes/EstimacionX2.png')
    plt.show()


    plt.plot(tiempo, fuerza, label='Fuerza')
    plt.plot(tiempo, fuerza_kalman, label='Fuerza estimada', color='limegreen')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('./imagenes/EstimacionF.png')
    plt.show()

    plt.plot(tiempo, x2_kalman, label='x2(t) estimado')
    plt.plot(tiempo, np.gradient(x1_kalman, dt), label='x2(t) estimado', color='limegreen')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Respuesta')
    plt.legend()
    plt.grid()
    plt.savefig('./imagenes/EstimacionX1Fuerza.png')
    plt.show()

