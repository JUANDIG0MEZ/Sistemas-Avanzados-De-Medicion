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
            R(T) = m*T + b ==> [T, 1] * [[m], [b]]
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
            return A * np.exp(B / (temperaturas))
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


if __name__ == "__main__":
    metodo1 = Metodos.svd_obtenerParametros
    print(metodo1)



# class Simulacion():
#     def __init__(self,  horno,  lista_sensores, num_iteraciones= 5):
#         self.num_iteraciones = num_iteraciones
#         self.horno = horno
#         self.lista_sensores = lista_sensores

#     def simularSensor(self, sensor, ruido):
#         """
#         """
#         temperaturas = self.horno.temperaturas
#         valores = sensor.calcularValores(temperaturas)
#         # ruido = valores * ruido
#         # valores_ruido = valores + ruido
#         # temperaturas_ruido = sensor.calcularTemperatura(sensor, valores_ruido)

#         # return valores_ruido, temperaturas_ruido
#         mean = np.mean(valores)
#         std = np.std(valores)
#         valores_ruido_normalizados = (valores - mean)/std + ruido 
#         valores_ruido_desnormalizados = valores_ruido_normalizados * std + mean
        
#         temperaturas_ruido = sensor.calcularTemperatura(sensor, valores_ruido_desnormalizados)
#         return valores_ruido_desnormalizados, temperaturas_ruido
        
    
#     def simulacionGaussianos(self):
#         lista_sensores_horno = self.lista_sensores.copy()
#         for nombreSensor, sensor in self.lista_sensores.items():
#             ruido = self.generarDiccionarioRuidos()["gaussiano"]
#             valores_ruido, temperata_ruido = self.simularSensor(sensor, ruido)
#             lista_sensores_horno[nombreSensor] = {
#                 "valores_ruido": valores_ruido,
#                 "temperaturas_ruido": temperata_ruido
#             }
#         return lista_sensores_horno

#     def simulacionVariosRuidos(self):
#         lista_sensores_horno = self.lista_sensores.copy()
#         for i, (nombreSensor, sensor) in enumerate(self.lista_sensores.items()):
#             keys_ruido = list(self.generarDiccionarioRuidos().keys())
#             ruido = self.generarDiccionarioRuidos()[keys_ruido[i]]

#             valores_ruido, temperata_ruido = self.simularSensor(sensor, ruido)
#             lista_sensores_horno[nombreSensor] = {
#                 "valores_ruido": valores_ruido,
#                 "temperaturas_ruido": temperata_ruido
#             }
#         return lista_sensores_horno

#     def monteCarlo(self, tipo_simulacion):
#         """
#         Esta funcion simula los sensores utilizando Monte Carlo
#         """
#         simulaciones = {} 
#         if tipo_simulacion == "gaussiano":
#             for i in range(self.num_iteraciones):
#                 simulaciones[i] = self.simulacionGaussianos()
#         elif tipo_simulacion == "variosRuidos":
#             for i in range(self.num_iteraciones):
#                 simulaciones[i] = self.simulacionVariosRuidos()
#         return simulaciones

    
#     def generarDiccionarioRuidos(self):

#         listaRuidos = {
#         "gaussiano": Ruido("gaussiano", 0, 0.1, longitud).valores,
#         "uniforme": Ruido("uniforme", 0, 0.1, longitud).valores,
#         "exponencial": Ruido("exponencial", 0.1, None, longitud).valores,
#         "poisson": Ruido("poisson", 0.1, None ,longitud).valores
#         }
#         return listaRuidos



