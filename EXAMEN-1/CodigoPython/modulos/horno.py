import matplotlib.pyplot as plt
import numpy as np


class Horno:
    def __init__(self, X, Y, Z, W, T0=0):
        """
        Este horno sigue un perfil de temperatura más dinámico tipo 'montaña rusa'.
        La temperatura incrementa, luego cae rápidamente, vuelve a subir y oscila.
        """
        self.X = X  # amplitud de subida
        self.Y = Y  # duración de cada fase
        self.Z = Z  # amplitud de bajada
        self.W = W  # duración de cada bajada
        self.T0 = T0
        self.temperaturas = self.generar_temperaturas()

    def generar_temperaturas(self):
        """
        Genera un perfil de temperatura con forma de montaña rusa.
        """
        temperaturas = []

        # Tiempo total: subida1 + bajada1 + subida2 + bajada2 + oscilaciones
        tiempo_subida1 = np.linspace(0, self.Y, self.Y)
        tiempo_bajada1 = np.linspace(0, self.W, self.W)
        tiempo_subida2 = np.linspace(0, self.Y, self.Y)
        tiempo_bajada2 = np.linspace(0, self.W, self.W)
        tiempo_osc = np.linspace(0, self.Y, self.Y)

        # Subida 1: seno creciente
        temp_subida1 = self.T0 + self.X * np.sin(0.5 * np.pi * tiempo_subida1 / self.Y)
        temperaturas.extend(temp_subida1)

        # Bajada 1: parábola invertida
        temp_bajada1 = temperaturas[-1] - self.Z * (tiempo_bajada1 / self.W) ** 2
        temperaturas.extend(temp_bajada1)

        # Subida 2: cuadrática
        temp_subida2 = temperaturas[-1] + (self.X / 2) * (tiempo_subida2 / self.Y) ** 2
        temperaturas.extend(temp_subida2)

        # Bajada 2: raíz cuadrada
        temp_bajada2 = temperaturas[-1] - (self.Z * 1.5) * np.sqrt(tiempo_bajada2 / self.W)
        temperaturas.extend(temp_bajada2)

        # Oscilaciones finales
        temp_osc = temperaturas[-1] + 0.1 * self.X * np.sin(10 * np.pi * tiempo_osc / self.Y)
        temperaturas.extend(temp_osc)

        return np.array(temperaturas)
    
    