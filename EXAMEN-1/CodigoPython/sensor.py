"""
Codigo mas organizado para el examen 1
"""

import matplotlib.pyplot as plt
import numpy as np
from tablas import *


class Metodos:
    @staticmethod
    def svd_obtenerParametros(A, b):
        """
        Esta funcion obtiene los parametros de la curva utilizando la descomposicion SVD
        """
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1 / S)
        A_inv = VT.T @ S_inv @ U.T
        return A_inv @ b

    @staticmethod
    def calcularErrores(valores, valores_calculados):
        """
        Esta funcion calcula los errores entre los valores y los valores calculados
        """
        errores = valores - valores_calculados
        return errores

    # @staticmethod
    # def calcularIncertidumbre(valores, )

class Sensor():
    def __init__(self, diccionario, nombre_sensor, unidades_valores, tipo_curva="lineal"):
        self.temperaturas, self.valores = self.obtener_datos(diccionario)
        self.length = len(self.valores)
        self.nombre_sensor = nombre_sensor
        self.unidades_valores = unidades_valores
        self.tipo_curva = tipo_curva
        self.parametros = None

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
        T = np.array(T) + 273.15
        valores = np.array(valores)
        return T, valores

    def calcularParametros(self):
        """
        Esta funcion calcula los parametros de la curva
        """
        if self.tipo_curva == "lineal":
            """
            R(T) = m*T + b ==> [T, 1] * [[m], [b]]
            """
            A = np.array([self.temperaturas, np.ones(self.length)]).T

        elif self.tipo_curva == "exponencial":
            """
            R(T) = A * e^(B/T) ==> ln(R(T)) = ln(A) + B/T ==> ln(R(T)) = [1/T, 1] * [[B], [ln(A)]]
            """
            A = np.array([1 / self.temperaturas, np.ones(len(self.valores))]).T

        elif self.tipo_curva == "polinomial":
            """
            V(T) = A + B * T + C * T^2 + D * T^3
            """
            A = np.array([np.ones(self.length), self.temperaturas, self.temperaturas**2, self.temperaturas ** 3]).T

        else: 
            raise ValueError("Tipo de curva no soportada")
        
        
        b = np.array(self.valores).T

        parametros =Metodos.svd_obtenerParametros(A, b)
        return parametros
    
    def obtenerParametros(self):
        """
        Esta funcion obtiene los parametros de la curva
        """
        if self.parametros is None:
            self.parametros = self.calcularParametros()
        return self.parametros

    def calcularValores(self, temperaturas):
        if self.parametros is None:
            self.parametros = self.calcularParametros()
        
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
            return A * np.exp(B / (temperaturas + 273.15))
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
        if sensor.parametros is None:
            sensor.parametros = sensor.calcularParametros()

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
            return B / np.log(valores / A)

        elif sensor.tipo_curva == "polinomial":
            """
            V(T) = A + B * T + C * T^2 + D * T^3
            """
            A = sensor.parametros[0]
            B = sensor.parametros[1]
            C = sensor.parametros[2]
            D = sensor.parametros[3]

            temperaturas = []
            print("voltajes para calcular las temperaturas")
            print(valores)
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

    def calcularIncertidumbre(self):
        """
        Esta funcion calcula la incertidumbre del sensor
        """
        pass


class Graficas():
    @staticmethod
    def graficar_sensor(sensor):
        """
        Esta funcion grafica los datos del sensor
        """
        plt.plot(sensor.temperaturas, sensor.valores, "o")
        plt.xlabel("Temperatura (°C)")   
        plt.ylabel(f"{sensor.unidades_valores}")
        plt.title(f"Grafica de {sensor.nombre_sensor}")
        plt.show()
    
    @staticmethod
    def graficar_sensor_con_curva(sensor):
        """
        Esta funcion grafica los datos del sensor junto con la curva ajustada
        """
        T = sensor.temperaturas
        if sensor.parametros is None: 
            sensor.parametros = sensor.calcularParametros()
        

        if sensor.tipo_curva == "lineal":
            y = sensor.parametros[0] * T + sensor.parametros[1]
        elif sensor.tipo_curva == "exponencial":
            y = np.exp(sensor.parametros[1] ) * np.exp(sensor.parametros[0] / (T))
        elif sensor.tipo_curva == "polinomial":
            y = sensor.parametros[0] + sensor.parametros[1] * T + sensor.parametros[2] * T**2 + sensor.parametros[3] * T**3
        else:
            raise ValueError("Tipo de curva no soportado")

        plt.plot(sensor.temperaturas, sensor.valores, "o", label="Datos")
        plt.plot(sensor.temperaturas, y, label="Curva ajustada")
        plt.xlabel("Temperatura (°K)")
        plt.ylabel(f"{sensor.unidades_valores}")
        plt.title(f"Grafica de {sensor.nombre_sensor} con curva ajustada")
        plt.legend()
    
    @staticmethod
    def grafica_simple(temperaturas, valores):
        plt.plot(temperaturas, valores, "o")


class Ruido():
    @staticmethod
    def ruidoGaussiano(sensor, desviacion):
        """
        Esta funcion crear un nuevo sensor con ruido gaussiano
        """
        valores = sensor.valores
        ruido = np.random.normal(0, desviacion, len(valores))
        valores_ruido = valores + ruido
        # crear el diccionario con los valores y temperaturas
        return valores_ruido
    


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
        return np.array(temperaturas) + 273.15

    def graficar_perfil_temperatura(self):
        """
        Esta funcion grafica el perfil de temperatura del horno
        """
        plt.plot(self.temperaturas)




sensor_PT1000 = Sensor(pt_1000, "PT1000", "Resistencia (Ohmios)", "lineal")
valores_PT1000_ruido = Ruido.ruidoGaussiano(sensor_PT1000, 1.5)
temperaturas_PT1000_ruido = Sensor.calcularTemperatura(sensor_PT1000, valores_PT1000_ruido)

Graficas.graficar_sensor_con_curva(sensor_PT1000)
Graficas.grafica_simple(temperaturas_PT1000_ruido, valores_PT1000_ruido)
print("valores del PT_1000_ con ruido", valores_PT1000_ruido)
plt.show()
