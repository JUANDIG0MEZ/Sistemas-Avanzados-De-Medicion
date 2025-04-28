import matplotlib.pyplot as plt
import numpy as np


class Horno:
    def __init__(self, X, Y, Z, W, T0=0):
        """
        Este horno sigue un perfil de temperatura conocido. La temperatura
        incrementa X grados en Y segundo, para luego decrecer Z grados en W segundos"""
        self.X = X
        self.Y = Y
        self.Z = Z
        self.W = W
        self.T0 = T0
        self.temperaturas = self.generar_temperaturas()
    
    def generar_temperaturas(self):
        """
        Esta funcion genera las temperaturas del horno en un tiempo determinado
        """
        tiempo = np.arange(0, self.Y + self.W, 1)
        temperaturas = []
        for t in tiempo:
            if t < self.Y:
                temperatura = self.T0 + (self.X / self.Y) * t
            else:
                temperatura = self.T0 + self.X - (self.Z / self.W) * (t - self.Y)
            temperaturas.append(temperatura)
        return np.array(temperaturas)
    
    def simular_sensor(self, sensor, outlier=0.005):
        dict_sensor = {}
        for temperatura in self.temperaturas:
            valor = sensor.calcularValores(temperatura)
            # Agregar Outliers
            if np.random.rand() < outlier:
                if np.random.rand() < 0.5:
                    outlier_value = np.random.uniform(1.5, 3)
                else:
                    outlier_value = np.random.uniform(-1.5, -3)
                valor = valor * outlier_value
            

            dict_sensor[temperatura] = float(valor)
        return dict_sensor