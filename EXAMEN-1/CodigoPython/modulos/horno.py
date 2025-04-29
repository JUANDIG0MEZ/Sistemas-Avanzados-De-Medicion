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
    
    