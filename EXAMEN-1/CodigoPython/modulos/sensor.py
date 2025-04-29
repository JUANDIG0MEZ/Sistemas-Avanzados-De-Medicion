"""
Codigo mas organizado para el examen 1
"""

import numpy as np
from .metodos import Metodos


class Sensor():
    def __init__(self, diccionario, nombre_sensor, unidades_valores, tipo_curva="lineal"):
        self.temperaturas, self.valores = self.obtener_datos(diccionario)
        self.length = len(self.valores)
        self.nombre_sensor = nombre_sensor
        self.unidades_valores = unidades_valores
        self.tipo_curva = tipo_curva
        self.parametros = self.calcularParametros()
    

    def obtener_datos(self, diccionario):
        """
        Esta funcion extrae los datos del sensor que se encuentran en el diccionario
        """
        T = []
        valores = []
        for temperatura, valor in diccionario.items():
            T.append(temperatura)
            valores.append(valor)

        # Convertimos las listas a arrays de numpy
        T = np.array(T)
        valores = np.array(valores)
        return T, valores

    def calcularParametros(self):
        """
        Esta funcion calcula los parametros de la curva
        """
        b = np.array(self.valores).T
        if self.tipo_curva == "lineal":
            """
                R(T) = m * (T + deltaError) + b ==> [T, 1] * [[m], [b]]
                R(T) = m *T + m*deltaError + b
                R(T) = m * T + b  +- m*deltaError
            """
            A = np.array([self.temperaturas, np.ones(self.length)]).T

        elif self.tipo_curva == "exponencial":
            """
                R(T) = A * e^(B/T) ==> ln(R(T)) = ln(A) + B/T ==> ln(R(T)) = [1/T, 1] * [[B], [ln(A)]]
            """
            A = np.array([1 / (self.temperaturas + 273.15), np.ones(len(self.valores))]).T
            b = np.log(b)

        elif self.tipo_curva == "polinomial":
            """
                V(T) = A + B * T + C * T^2 + D * T^3 
            """
            A = np.array([np.ones(self.length), self.temperaturas, self.temperaturas**2, self.temperaturas ** 3]).T

        else: 
            raise ValueError("Tipo de curva no soportada")
        
        parametros = Metodos.svd_obtenerParametros(A, b)
        return parametros
    
    def calcularValores(self, temperaturas):
       
        if self.tipo_curva == "lineal":
            """
            V(T) = m*T + b
            """
            m = self.parametros[0]
            b = self.parametros[1]
            return m * temperaturas + b
        elif self.tipo_curva == "exponencial":
            """
            R(T) = A * e^(B/T)
            """
            A = np.exp(self.parametros[1])
            B = self.parametros[0]
            R = A * np.exp(B / (temperaturas + 273.15))
            return R
        elif self.tipo_curva == "polinomial":
            """
            V(T0) = A + B * T + C * T^2 + D * T^3
            """
            A = self.parametros[0]
            B = self.parametros[1]
            C = self.parametros[2]
            D = self.parametros[3]
            return A + B * temperaturas + C * temperaturas**2 + D * temperaturas**3
        else:
            raise ValueError("Tipo de curva no soportada")

    @staticmethod
    def calcularTemperatura(sensor, valores):
        """
        Esta funcion calcula la temperatura a partir de los valores
        """
        if sensor.tipo_curva == "lineal":
            """
            V(T) = m*T + b ==> T = (V(T) - b) / m
            """
            m = sensor.parametros[0]
            b = sensor.parametros[1]
            return (valores - b) / m

        elif sensor.tipo_curva == "exponencial":
            """
            R(T) = A * e^(B/T) ==> T = B / ln(R(T)/A)
            """
            A = np.exp(sensor.parametros[1])
            B = sensor.parametros[0]
            T = B / np.log(valores / A) - 273.15
            return T

        elif sensor.tipo_curva == "polinomial":
            """
            V(T) = A + B * T + C * T^2 + D * T^3
            """
            A = sensor.parametros[0]
            B = sensor.parametros[1]
            C = sensor.parametros[2]
            D = sensor.parametros[3]

            temperaturas = []
            for valor in valores:
                coef = [D, C, B, A - valor]
                # raices
                raices = np.roots(coef)
                # Filtramos las raices reales
                reales = raices[np.isreal(raices)].real
                # Elegimos la raiz valida y positiva
                posibles = reales[reales > 0]
                if len(posibles) == 0:
                    temperaturas.append(np.nan)
                else:
                    temperaturas.append(posibles.min())
            return np.array(temperaturas)

        else: 
            raise ValueError("Tipo de curva no soportada")
